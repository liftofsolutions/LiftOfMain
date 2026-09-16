"""Btest for the config component.

Logic only: every test controls the environment directly and checks what
`getSettings` makes of it. Nothing here reads the real `.env`, opens a
connection or touches a database - `loadEnvironment` is stubbed out in the
fixture so a developer's own `.env` cannot change the result.
"""

import os
from pathlib import Path

import pytest

from config import settings as configModule
from config.settings import (
    DEFAULT_FIRESTORE_DATABASE_ID,
    PROJECT_ROOT,
    ConfigurationError,
    Settings,
    getSettings,
    resolveCredentialsPath,
)

# Every variable the component reads, so a test can clear the lot and set back
# only what it is actually exercising.
CONFIG_ENV_VARS = [
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "PINECONE_INDEX_HOST",
    "GCP_PROJECT_ID",
    "FIRESTORE_DATABASE_ID",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "REDIS_URL",
    "MONGODB_URI",
    "MONGODB_DATABASE",
    "RAG_UNRAR_TOOL",
]


@pytest.fixture
def cleanEnvironment(monkeypatch):
    """An empty configuration environment, with `.env` loading disabled.

    `getSettings` caches, so the cache is cleared either side of the test: once
    so this test does not inherit an earlier one's snapshot, once so it does not
    leave its own behind.
    """
    monkeypatch.setattr(configModule, "loadEnvironment", lambda: False)
    for envVarName in CONFIG_ENV_VARS:
        monkeypatch.delenv(envVarName, raising=False)
    getSettings.cache_clear()
    yield monkeypatch
    getSettings.cache_clear()


def buildEnvironment(monkeypatch, **values) -> None:
    """Set the named configuration variables and nothing else."""
    for envVarName, value in values.items():
        monkeypatch.setenv(envVarName, value)


class TestResolveCredentialsPath:
    """Where a service account file is looked for."""

    def testRelativePathIsAnchoredToTheProjectRoot(self):
        resolved = resolveCredentialsPath("keys/service-account.json")

        assert Path(resolved).is_absolute()
        assert Path(resolved) == PROJECT_ROOT / "keys" / "service-account.json"

    def testAbsolutePathIsLeftAlone(self):
        # Spelled per-platform: a container mounts its key at a POSIX path, a
        # developer machine holds it on a drive letter, and neither is rewritten.
        absolutePath = "C:\\secrets\\key.json" if os.name == "nt" else "/secrets/key.json"

        assert resolveCredentialsPath(absolutePath) == str(Path(absolutePath))

    def testEmptyStaysEmpty(self):
        # Absent credentials are legitimate: on Cloud Run the runtime service
        # account is picked up from the metadata server instead.
        assert resolveCredentialsPath("") == ""

    def testHomeShorthandIsExpanded(self):
        resolved = resolveCredentialsPath("~/key.json")

        assert "~" not in resolved
        assert Path(resolved).is_absolute()


class TestSettingsFromEnvironment:
    """What the component makes of the variables this project actually sets."""

    def testProjectVariableNamesAreTranslated(self, cleanEnvironment):
        buildEnvironment(
            cleanEnvironment,
            GCP_PROJECT_ID="chatbot-9b8d3",
            FIRESTORE_DATABASE_ID="default",
            PINECONE_API_KEY="pc-key",
            PINECONE_INDEX_NAME="rag-chunks",
            REDIS_URL="redis://localhost:6379/0",
        )

        settings = getSettings()

        # The adapters spell these GOOGLE_CLOUD_PROJECT / FIRESTORE_DATABASE;
        # this project spells them its own way and the gap closes here.
        assert settings.gcpProjectId == "chatbot-9b8d3"
        assert settings.firestoreDatabaseId == "default"
        assert settings.pineconeIndexName == "rag-chunks"
        assert settings.redisUrl == "redis://localhost:6379/0"

    def testNamedDefaultDatabaseIsNotTheUnnamedOne(self, cleanEnvironment):
        # This project's database is *named* "default". Firestore reserves
        # "(default)" for a project's unnamed one, and they are not the same
        # database - collapsing them would read from the wrong place.
        buildEnvironment(cleanEnvironment, FIRESTORE_DATABASE_ID="default")

        assert getSettings().firestoreDatabaseId == "default"
        assert getSettings().firestoreDatabaseId != DEFAULT_FIRESTORE_DATABASE_ID

    def testUnsetDatabaseIdFallsBackToFirestoresOwnDefault(self, cleanEnvironment):
        assert getSettings().firestoreDatabaseId == DEFAULT_FIRESTORE_DATABASE_ID

    def testCredentialsPathIsResolvedAndPublishedBackToTheEnvironment(self, cleanEnvironment):
        buildEnvironment(
            cleanEnvironment, GOOGLE_APPLICATION_CREDENTIALS="keys/service-account.json"
        )

        settings = getSettings()

        assert Path(settings.googleCredentialsPath).is_absolute()
        # google-auth reads the variable itself, so anything in this process
        # building its own client has to see the same absolute path.
        assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == settings.googleCredentialsPath

    def testMissingValuesAreEmptyRatherThanAnImportTimeFailure(self, cleanEnvironment):
        settings = getSettings()

        assert settings == Settings()
        assert settings.pineconeApiKey == ""
        assert settings.mongoUri == ""

    def testSettingsAreCachedUntilExplicitlyCleared(self, cleanEnvironment):
        buildEnvironment(cleanEnvironment, GCP_PROJECT_ID="first")
        assert getSettings().gcpProjectId == "first"

        buildEnvironment(cleanEnvironment, GCP_PROJECT_ID="second")
        assert getSettings().gcpProjectId == "first", "the snapshot must not drift mid-process"

        getSettings.cache_clear()
        assert getSettings().gcpProjectId == "second"

    def testSettingsAreFrozen(self, cleanEnvironment):
        with pytest.raises(Exception):
            getSettings().gcpProjectId = "rewritten"


class TestConfiguredFlags:
    """Whether a store is configured well enough to be worth connecting to."""

    @pytest.mark.parametrize(
        "environment, expectedFlags",
        [
            ({}, {"pinecone": False, "firestore": False, "redis": False, "mongo": False}),
            (
                {"PINECONE_API_KEY": "k", "PINECONE_INDEX_NAME": "i"},
                {"pinecone": True, "firestore": False, "redis": False, "mongo": False},
            ),
            # A host alone identifies the index, so a name is not also required.
            (
                {"PINECONE_API_KEY": "k", "PINECONE_INDEX_HOST": "https://h"},
                {"pinecone": True, "firestore": False, "redis": False, "mongo": False},
            ),
            # An index without a key is not usable.
            (
                {"PINECONE_INDEX_NAME": "i"},
                {"pinecone": False, "firestore": False, "redis": False, "mongo": False},
            ),
            (
                {"GCP_PROJECT_ID": "p"},
                {"pinecone": False, "firestore": True, "redis": False, "mongo": False},
            ),
            (
                {"REDIS_URL": "redis://localhost:6379/0"},
                {"pinecone": False, "firestore": False, "redis": True, "mongo": False},
            ),
            # Mongo needs both halves: a URI without a database names nothing.
            (
                {"MONGODB_URI": "mongodb://localhost:27017"},
                {"pinecone": False, "firestore": False, "redis": False, "mongo": False},
            ),
            (
                {"MONGODB_URI": "mongodb://localhost:27017", "MONGODB_DATABASE": "rag"},
                {"pinecone": False, "firestore": False, "redis": False, "mongo": True},
            ),
        ],
    )
    def testFlagsReflectWhatIsActuallyUsable(
        self, cleanEnvironment, environment, expectedFlags
    ):
        buildEnvironment(cleanEnvironment, **environment)

        settings = getSettings()

        assert {
            "pinecone": settings.hasPinecone,
            "firestore": settings.hasFirestore,
            "redis": settings.hasRedis,
            "mongo": settings.hasMongo,
        } == expectedFlags


class TestAdapterFactories:
    """What the factories hand the adapters, and how they fail when they cannot.

    Constructing an adapter opens no connection, so these stay logic tests.
    """

    def testPineconeAdapterGetsTheConfiguredIndex(self, cleanEnvironment):
        buildEnvironment(
            cleanEnvironment,
            PINECONE_API_KEY="pc-key",
            PINECONE_INDEX_NAME="rag-chunks",
            PINECONE_INDEX_HOST="https://rag-chunks.svc.pinecone.io",
        )

        adapter = configModule.buildPineconeAdapter()

        assert adapter._indexName == "rag-chunks"
        # The host is passed as well as the name, which saves the describe round
        # trip the adapter would otherwise spend resolving one from the other.
        assert adapter._indexHost == "https://rag-chunks.svc.pinecone.io"

    def testFirestoreAdapterGetsTheNamedDatabase(self, cleanEnvironment):
        buildEnvironment(
            cleanEnvironment, GCP_PROJECT_ID="chatbot-9b8d3", FIRESTORE_DATABASE_ID="default"
        )

        adapter = configModule.buildFirestoreAdapter()

        assert adapter.databaseId == "default"

    def testOverridesWinOverConfiguration(self, cleanEnvironment):
        buildEnvironment(cleanEnvironment, REDIS_URL="redis://configured:6379/0")

        adapter = configModule.buildRedisAdapter(connectionUrl="redis://override:6379/1")

        assert adapter._connectionUrl == "redis://override:6379/1"

    @pytest.mark.parametrize(
        "factoryName, expectedVariable",
        [
            ("buildPineconeAdapter", "PINECONE_API_KEY"),
            ("buildFirestoreAdapter", "GCP_PROJECT_ID"),
            ("buildRedisAdapter", "REDIS_URL"),
            ("buildMongoAdapter", "MONGODB_URI"),
        ],
    )
    def testAnUnconfiguredStoreNamesTheVariableItNeeds(
        self, cleanEnvironment, factoryName, expectedVariable
    ):
        with pytest.raises(ConfigurationError) as raised:
            getattr(configModule, factoryName)()

        assert expectedVariable in str(raised.value)

    def testPineconeWithAKeyButNoIndexSaysSo(self, cleanEnvironment):
        buildEnvironment(cleanEnvironment, PINECONE_API_KEY="pc-key")

        with pytest.raises(ConfigurationError) as raised:
            configModule.buildPineconeAdapter()

        assert "PINECONE_INDEX_NAME" in str(raised.value)

    def testMongoWithAUriButNoDatabaseSaysSo(self, cleanEnvironment):
        buildEnvironment(cleanEnvironment, MONGODB_URI="mongodb://localhost:27017")

        with pytest.raises(ConfigurationError) as raised:
            configModule.buildMongoAdapter()

        assert "MONGODB_DATABASE" in str(raised.value)
