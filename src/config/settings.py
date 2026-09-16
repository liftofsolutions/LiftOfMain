"""Where this service's configuration comes from, resolved once at import.

Not a route component - there is nothing to expose over HTTP here - so this
folder has no `routes.py` or `schemas.py`. It sits under `src/` because every
component needs it and nothing should reach past it to `os.environ` directly.

Two jobs:

1. **Load `.env`.** Nothing else does. Values already in the real environment win,
   so a container's own configuration is never overwritten by a stray file.
2. **Translate.** The adapters in `DbAdapters` name their environment variables
   generically (`GOOGLE_CLOUD_PROJECT`, `FIRESTORE_DATABASE`), while this project
   spells them its own way (`GCP_PROJECT_ID`, `FIRESTORE_DATABASE_ID`). The
   vocabulary is reconciled here and passed to the adapters explicitly, so the
   adapters stay reusable and the project keeps its own names.

The `build*Adapter` helpers are the intended entry point. Build one adapter per
process and share it - each carries a connection pool that only pays off if it
outlives the request.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from DbAdapters import FirestoreAdapter, MongoAdapter, PineconeAdapter, RedisAdapter

# src/config/settings.py -> src/config -> src -> the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE_PATH = PROJECT_ROOT / ".env"

PINECONE_API_KEY_ENV_VAR = "PINECONE_API_KEY"
PINECONE_INDEX_NAME_ENV_VAR = "PINECONE_INDEX_NAME"
PINECONE_INDEX_HOST_ENV_VAR = "PINECONE_INDEX_HOST"
GCP_PROJECT_ID_ENV_VAR = "GCP_PROJECT_ID"
FIRESTORE_DATABASE_ID_ENV_VAR = "FIRESTORE_DATABASE_ID"
GOOGLE_CREDENTIALS_ENV_VAR = "GOOGLE_APPLICATION_CREDENTIALS"
REDIS_URL_ENV_VAR = "REDIS_URL"
MONGODB_URI_ENV_VAR = "MONGODB_URI"
MONGODB_DATABASE_ENV_VAR = "MONGODB_DATABASE"
UNRAR_TOOL_ENV_VAR = "RAG_UNRAR_TOOL"

# Firestore reserves "(default)" for a project's unnamed database. This project's
# database is *named* "default", which is a different thing and only resolves
# because FIRESTORE_DATABASE_ID says so explicitly.
DEFAULT_FIRESTORE_DATABASE_ID = "(default)"


class ConfigurationError(RuntimeError):
    """A setting a caller asked for is missing or unusable.

    Raised at the point of use rather than at import, so a service that never
    touches Redis is not stopped by an unset `REDIS_URL`.
    """


@dataclass(frozen=True, slots=True)
class Settings:
    """Every external endpoint this service talks to, as configured.

    Frozen: configuration is read at startup and not rewritten afterwards, so a
    value read in one place cannot drift from the same value read in another.
    An empty string means "not configured", which the `require*` helpers turn
    into a clear error at the moment it actually matters.
    """

    pineconeApiKey: str = ""
    pineconeIndexName: str = ""
    pineconeIndexHost: str = ""
    gcpProjectId: str = ""
    firestoreDatabaseId: str = DEFAULT_FIRESTORE_DATABASE_ID
    googleCredentialsPath: str = ""
    redisUrl: str = ""
    mongoUri: str = ""
    mongoDatabase: str = ""
    unrarTool: str = ""

    @property
    def hasPinecone(self) -> bool:
        """Whether Pinecone is configured well enough to connect."""
        return bool(self.pineconeApiKey and (self.pineconeIndexName or self.pineconeIndexHost))

    @property
    def hasFirestore(self) -> bool:
        """Whether Firestore is configured well enough to connect.

        Credentials may legitimately be absent: on Cloud Run the runtime service
        account is picked up from the metadata server, so a project id is enough.
        """
        return bool(self.gcpProjectId or self.googleCredentialsPath)

    @property
    def hasRedis(self) -> bool:
        """Whether a Redis URL is configured."""
        return bool(self.redisUrl)

    @property
    def hasMongo(self) -> bool:
        """Whether MongoDB is configured well enough to connect."""
        return bool(self.mongoUri and self.mongoDatabase)


def resolveCredentialsPath(rawPath: str) -> str:
    """Turn a credentials path from the environment into an absolute one.

    `.env` holds it relative to the repository root, which only resolves while
    the process happens to be running from there - true for `fastapi dev`, not
    true for a worker started elsewhere or for pytest run from a subdirectory.
    Anchoring it to the root makes the working directory stop mattering.

    An absolute path is returned unchanged, so a container that mounts its key
    somewhere else is unaffected.
    """
    if not rawPath:
        return ""

    path = Path(rawPath).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def loadEnvironment() -> bool:
    """Read `.env` into the process environment. Returns whether a file was found.

    Values already set in the real environment are left alone: deployment
    configuration outranks a file that only exists on a developer's machine.
    Safe to call repeatedly.
    """
    return load_dotenv(ENV_FILE_PATH, override=False)


@lru_cache(maxsize=1)
def getSettings() -> Settings:
    """The configuration for this process, read once and cached.

    Cached because the values cannot change without a restart, and because a
    frozen snapshot means two components cannot disagree about what is
    configured. Call `getSettings.cache_clear()` in a test that needs to reload.
    """
    loadEnvironment()

    credentialsPath = resolveCredentialsPath(os.environ.get(GOOGLE_CREDENTIALS_ENV_VAR, ""))
    if credentialsPath:
        # google-auth reads this variable itself, from anything in this process
        # that builds its own client. Rewriting it absolute keeps those in step
        # with the adapters built here.
        os.environ[GOOGLE_CREDENTIALS_ENV_VAR] = credentialsPath

    return Settings(
        pineconeApiKey=os.environ.get(PINECONE_API_KEY_ENV_VAR, ""),
        pineconeIndexName=os.environ.get(PINECONE_INDEX_NAME_ENV_VAR, ""),
        pineconeIndexHost=os.environ.get(PINECONE_INDEX_HOST_ENV_VAR, ""),
        gcpProjectId=os.environ.get(GCP_PROJECT_ID_ENV_VAR, ""),
        firestoreDatabaseId=(
            os.environ.get(FIRESTORE_DATABASE_ID_ENV_VAR) or DEFAULT_FIRESTORE_DATABASE_ID
        ),
        googleCredentialsPath=credentialsPath,
        redisUrl=os.environ.get(REDIS_URL_ENV_VAR, ""),
        mongoUri=os.environ.get(MONGODB_URI_ENV_VAR, ""),
        mongoDatabase=os.environ.get(MONGODB_DATABASE_ENV_VAR, ""),
        unrarTool=os.environ.get(UNRAR_TOOL_ENV_VAR, ""),
    )


def requireSetting(value: str, envVarName: str, purpose: str) -> str:
    """Return `value`, or explain which variable has to be set and what for."""
    if not value:
        raise ConfigurationError(
            f"{purpose} needs ${envVarName}, which is not set. Add it to "
            f"{ENV_FILE_PATH.name} or to the environment."
        )
    return value


def buildPineconeAdapter(**overrides) -> PineconeAdapter:
    """A Pinecone adapter pointed at the configured index.

    Passing the host as well as the name saves a describe round trip on connect,
    so both are handed over when both are known.

    Raises:
        ConfigurationError: If the API key or the index is not configured.
    """
    settings = getSettings()
    requireSetting(settings.pineconeApiKey, PINECONE_API_KEY_ENV_VAR, "Pinecone")
    if not (settings.pineconeIndexName or settings.pineconeIndexHost):
        raise ConfigurationError(
            f"Pinecone needs ${PINECONE_INDEX_NAME_ENV_VAR} or "
            f"${PINECONE_INDEX_HOST_ENV_VAR}, and neither is set."
        )

    return PineconeAdapter(
        **{
            "apiKey": settings.pineconeApiKey,
            "indexName": settings.pineconeIndexName,
            "indexHost": settings.pineconeIndexHost,
            **overrides,
        }
    )


def buildFirestoreAdapter(**overrides) -> FirestoreAdapter:
    """A Firestore adapter pointed at the configured database.

    Raises:
        ConfigurationError: If neither a project id nor credentials are set,
            leaving nothing to identify the database with.
    """
    settings = getSettings()
    if not settings.hasFirestore:
        raise ConfigurationError(
            f"Firestore needs ${GCP_PROJECT_ID_ENV_VAR} or "
            f"${GOOGLE_CREDENTIALS_ENV_VAR}, and neither is set."
        )

    return FirestoreAdapter(
        **{
            "projectId": settings.gcpProjectId,
            "databaseId": settings.firestoreDatabaseId,
            "credentialsPath": settings.googleCredentialsPath,
            **overrides,
        }
    )


def buildRedisAdapter(**overrides) -> RedisAdapter:
    """A Redis adapter pointed at the configured server.

    Raises:
        ConfigurationError: If `REDIS_URL` is not set.
    """
    settings = getSettings()
    requireSetting(settings.redisUrl, REDIS_URL_ENV_VAR, "Redis")
    return RedisAdapter(**{"connectionUrl": settings.redisUrl, **overrides})


def buildMongoAdapter(**overrides) -> MongoAdapter:
    """A MongoDB adapter pointed at the configured database.

    Raises:
        ConfigurationError: If the URI or the database name is not set. Neither
            has a default worth guessing at: a wrong guess writes real documents
            into the wrong place.
    """
    settings = getSettings()
    requireSetting(settings.mongoUri, MONGODB_URI_ENV_VAR, "MongoDB")
    requireSetting(settings.mongoDatabase, MONGODB_DATABASE_ENV_VAR, "MongoDB")
    return MongoAdapter(
        **{
            "connectionUri": settings.mongoUri,
            "databaseName": settings.mongoDatabase,
            **overrides,
        }
    )
