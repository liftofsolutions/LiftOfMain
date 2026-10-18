"""Btest for the id checks in `documentProcessing.utilities`.

Runs against the real Firestore from `test.env`, and is skipped when that
is not configured. The `dummyRecords` fixture writes the records the checks
read - an account, a live and a deleted project, memberships, a rag and a
rag job - before the first test, and deletes every one of them after the
last, whether or not the tests passed. Ids carry a `uuid4`, so two runs
never collide.

The adapter's gRPC channel is bound to the event loop that opened it, so
setup, every check and teardown run on one loop, `runAsync`. Each test gets
a fresh, empty memoryDb, and `firestoreReads` records every Firestore read,
so a cached answer can be told from a fetched one.

    uv run pytest Btest/documentProcessing -q
"""

import asyncio
import uuid
from collections.abc import Callable, Iterator
from datetime import datetime, timezone

import pytest

from config import getServerConfig
from DbAdapters import Document, deleteOperation, setOperation
from DbAdapters.memoryDb import MemoryDb
from documentProcessing import utilities

pytestmark = pytest.mark.skipif(
    not getServerConfig().hasFirestore,
    reason="Firestore is not configured for this environment.",
)

ACCOUNT_ID = str(uuid.uuid4())
OTHER_ACCOUNT_ID = str(uuid.uuid4())
PROJECT_ID = str(uuid.uuid4())
DELETED_PROJECT_ID = str(uuid.uuid4())
OTHER_PROJECT_ID = str(uuid.uuid4())
RAG_ID = str(uuid.uuid4())
OTHER_RAG_ID = str(uuid.uuid4())
RAG_JOB_ID = str(uuid.uuid4())
MISSING_ID = str(uuid.uuid4())

DELETED_AT = datetime(2026, 10, 10, tzinfo=timezone.utc)


def buildAccount(**overrides) -> dict:
    """A user record, shaped as `db/user-schema` stores one."""
    account = {"_id": ACCOUNT_ID, "display_name": "Btest", "identities": []}
    account.update(overrides)
    return account


def buildProject(**overrides) -> dict:
    """A live project record, shaped as `db/project-schema` stores one."""
    project = {
        "_id": PROJECT_ID,
        "name": "Btest project",
        "scope": "personal",
        "status": "active",
        "created_by": ACCOUNT_ID,
    }
    project.update(overrides)
    return project


def buildProjectMember(**overrides) -> dict:
    """The account's membership of the project, keyed as the schema keys it."""
    member = {"project_id": PROJECT_ID, "user_id": ACCOUNT_ID}
    member.update(overrides)
    member["_id"] = f"{member['project_id']}_{member['user_id']}"
    return member


def buildRag(**overrides) -> dict:
    """A rag of the project, shaped as `db/knowledge-schema` stores one."""
    rag = {"_id": RAG_ID, "project_id": PROJECT_ID, "status": "ready"}
    rag.update(overrides)
    return rag


def buildRagJob(**overrides) -> dict:
    """A job of the rag, shaped as `db/knowledge-schema` stores one."""
    ragJob = {
        "_id": RAG_JOB_ID,
        "project_id": PROJECT_ID,
        "rag_id": RAG_ID,
        "status": "done",
    }
    ragJob.update(overrides)
    return ragJob


# Every record the module writes, as (collection, record).
DUMMY_RECORDS = [
    ("users", buildAccount()),
    ("projects", buildProject()),
    (
        "projects",
        buildProject(_id=DELETED_PROJECT_ID, deleted_at=DELETED_AT),
    ),
    ("project_members", buildProjectMember()),
    ("project_members", buildProjectMember(project_id=DELETED_PROJECT_ID)),
    ("rags", buildRag()),
    ("rag_jobs", buildRagJob()),
]
# Written by a test rather than up front, and deleted with the rest.
LATE_RECORDS = [("users", OTHER_ACCOUNT_ID)]


@pytest.fixture(scope="module")
def runAsync() -> Iterator[Callable]:
    """Run a coroutine to the end on the module's one event loop."""
    loop = asyncio.new_event_loop()
    yield loop.run_until_complete
    loop.close()


@pytest.fixture(scope="module", autouse=True)
def dummyRecords(runAsync: Callable) -> Iterator[None]:
    """Write the dummy records, then delete them after the last test."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(utilities, "_FIRESTORE_ADAPTER", None)
        adapter = utilities.getFirestoreAdapter()
        try:
            runAsync(
                adapter.applyAtomically(
                    [
                        setOperation(
                            collection, Document(record["_id"], record)
                        )
                        for collection, record in DUMMY_RECORDS
                    ]
                )
            )
            yield
        finally:
            runAsync(
                adapter.applyAtomically(
                    [
                        deleteOperation(collection, record["_id"])
                        for collection, record in DUMMY_RECORDS
                    ]
                    + [
                        deleteOperation(collection, recordId)
                        for collection, recordId in LATE_RECORDS
                    ]
                )
            )
            runAsync(adapter.close())


@pytest.fixture(autouse=True)
def freshCache(monkeypatch) -> MemoryDb:
    """An empty memoryDb with the caches registered, for each test."""
    memoryDb = MemoryDb()
    monkeypatch.setattr(utilities, "getMemoryDb", lambda: memoryDb)
    utilities.registerCaches()
    return memoryDb


@pytest.fixture
def firestoreReads(monkeypatch) -> list[tuple[str, str]]:
    """Every Firestore read the test makes, as (collection, record id)."""
    adapter = utilities.getFirestoreAdapter()
    realFetch = adapter.fetchDocument
    reads = []

    async def countingFetch(collection: str, documentId: str):
        reads.append((collection, documentId))
        return await realFetch(collection, documentId)

    monkeypatch.setattr(adapter, "fetchDocument", countingFetch)
    return reads


class TestIsAccountIdValid:
    """An account id is valid when its user record exists."""

    def testKnownAccountIsValid(self, runAsync: Callable) -> None:
        assert runAsync(utilities.isAccountIdValid(ACCOUNT_ID)) is True

    def testUnknownAccountIsNotValid(self, runAsync: Callable) -> None:
        assert runAsync(utilities.isAccountIdValid(MISSING_ID)) is False


class TestIsProjectIdValid:
    """A project id is valid when the project exists and is not deleted."""

    def testLiveProjectIsValid(self, runAsync: Callable) -> None:
        assert runAsync(utilities.isProjectIdValid(PROJECT_ID)) is True

    def testDeletedProjectIsNotValid(self, runAsync: Callable) -> None:
        assert (
            runAsync(utilities.isProjectIdValid(DELETED_PROJECT_ID)) is False
        )

    def testUnknownProjectIsNotValid(self, runAsync: Callable) -> None:
        assert runAsync(utilities.isProjectIdValid(MISSING_ID)) is False


class TestIsProjectIdAssociatedWithAccountId:
    """A project belongs to an account the account is a member of."""

    def testMemberIsAssociated(self, runAsync: Callable) -> None:
        associated = runAsync(
            utilities.isProjectIdAssociatedWithAccountId(
                PROJECT_ID, ACCOUNT_ID
            )
        )

        assert associated is True

    def testNonMemberIsNotAssociated(self, runAsync: Callable) -> None:
        associated = runAsync(
            utilities.isProjectIdAssociatedWithAccountId(
                PROJECT_ID, OTHER_ACCOUNT_ID
            )
        )

        assert associated is False

    def testMemberOfDeletedProjectIsNotAssociated(
        self, runAsync: Callable
    ) -> None:
        associated = runAsync(
            utilities.isProjectIdAssociatedWithAccountId(
                DELETED_PROJECT_ID, ACCOUNT_ID
            )
        )

        assert associated is False


class TestIsRagJobIdAssociatedWithRagId:
    """A rag job belongs to the rag its record names."""

    def testJobOfTheRagIsAssociated(self, runAsync: Callable) -> None:
        associated = runAsync(
            utilities.isRagJobIdAssociatedWithRagId(RAG_JOB_ID, RAG_ID)
        )

        assert associated is True

    def testJobOfAnotherRagIsNotAssociated(self, runAsync: Callable) -> None:
        associated = runAsync(
            utilities.isRagJobIdAssociatedWithRagId(RAG_JOB_ID, OTHER_RAG_ID)
        )

        assert associated is False

    def testUnknownJobIsNotAssociated(self, runAsync: Callable) -> None:
        associated = runAsync(
            utilities.isRagJobIdAssociatedWithRagId(MISSING_ID, RAG_ID)
        )

        assert associated is False


class TestIsRagIdAssociatedWithProjectId:
    """A rag belongs to the project its record names."""

    def testRagOfTheProjectIsAssociated(self, runAsync: Callable) -> None:
        associated = runAsync(
            utilities.isRagIdAssociatedWithProjectId(RAG_ID, PROJECT_ID)
        )

        assert associated is True

    def testRagOfAnotherProjectIsNotAssociated(
        self, runAsync: Callable
    ) -> None:
        associated = runAsync(
            utilities.isRagIdAssociatedWithProjectId(RAG_ID, OTHER_PROJECT_ID)
        )

        assert associated is False

    def testUnknownRagIsNotAssociated(self, runAsync: Callable) -> None:
        associated = runAsync(
            utilities.isRagIdAssociatedWithProjectId(MISSING_ID, PROJECT_ID)
        )

        assert associated is False


class TestCaching:
    """Each check reads Firestore once, then answers from the cache."""

    @pytest.mark.parametrize(
        ("callCheck", "expectedReads"),
        [
            (
                lambda: utilities.isAccountIdValid(ACCOUNT_ID),
                [("users", ACCOUNT_ID)],
            ),
            (
                lambda: utilities.isProjectIdValid(PROJECT_ID),
                [("projects", PROJECT_ID)],
            ),
            (
                lambda: utilities.isProjectIdAssociatedWithAccountId(
                    PROJECT_ID, ACCOUNT_ID
                ),
                [
                    ("projects", PROJECT_ID),
                    ("project_members", f"{PROJECT_ID}_{ACCOUNT_ID}"),
                ],
            ),
            (
                lambda: utilities.isRagJobIdAssociatedWithRagId(
                    RAG_JOB_ID, RAG_ID
                ),
                [("rag_jobs", RAG_JOB_ID)],
            ),
            (
                lambda: utilities.isRagIdAssociatedWithProjectId(
                    RAG_ID, PROJECT_ID
                ),
                [("rags", RAG_ID)],
            ),
        ],
        ids=["account", "project", "membership", "ragJob", "rag"],
    )
    def testRepeatCallMakesNoNewRead(
        self,
        runAsync: Callable,
        firestoreReads: list,
        callCheck: Callable,
        expectedReads: list,
    ) -> None:
        first = runAsync(callCheck())
        readsAfterFirst = list(firestoreReads)
        second = runAsync(callCheck())
        third = runAsync(callCheck())

        assert (first, second, third) == (True, True, True)
        assert readsAfterFirst == expectedReads
        assert firestoreReads == expectedReads

    def testFetchedRecordIsStoredInTheCache(
        self, runAsync: Callable, freshCache: MemoryDb
    ) -> None:
        runAsync(utilities.isRagJobIdAssociatedWithRagId(RAG_JOB_ID, RAG_ID))

        assert freshCache.getRecord("rag_jobs", RAG_JOB_ID) == buildRagJob()

    def testCachedRecordIsUsedBeforeFirestore(
        self,
        runAsync: Callable,
        freshCache: MemoryDb,
        firestoreReads: list,
    ) -> None:
        # Only the cache holds this account; Firestore has never heard of it.
        freshCache.addRecord("users", MISSING_ID, buildAccount(_id=MISSING_ID))

        assert runAsync(utilities.isAccountIdValid(MISSING_ID)) is True
        assert firestoreReads == []

    def testCacheCountsOneMissThenHits(
        self, runAsync: Callable, freshCache: MemoryDb
    ) -> None:
        for _ in range(3):
            runAsync(
                utilities.isRagIdAssociatedWithProjectId(RAG_ID, PROJECT_ID)
            )

        stats = freshCache.getCollectionStats("rags")
        assert (stats.missCount, stats.hitCount, stats.recordCount) == (1, 2, 1)

    def testChecksShareTheCachedProject(
        self, runAsync: Callable, firestoreReads: list
    ) -> None:
        runAsync(utilities.isProjectIdValid(PROJECT_ID))
        runAsync(
            utilities.isProjectIdAssociatedWithAccountId(
                PROJECT_ID, ACCOUNT_ID
            )
        )

        # The project once, then only the membership.
        assert firestoreReads == [
            ("projects", PROJECT_ID),
            ("project_members", f"{PROJECT_ID}_{ACCOUNT_ID}"),
        ]

    def testMissingRecordIsNotCached(
        self,
        runAsync: Callable,
        freshCache: MemoryDb,
        firestoreReads: list,
    ) -> None:
        before = runAsync(utilities.isAccountIdValid(OTHER_ACCOUNT_ID))
        cachedAfterMiss = freshCache.getCollectionStats("users").recordCount
        runAsync(
            utilities.getFirestoreAdapter().upsertDocument(
                "users",
                Document(OTHER_ACCOUNT_ID, buildAccount(_id=OTHER_ACCOUNT_ID)),
            )
        )
        after = runAsync(utilities.isAccountIdValid(OTHER_ACCOUNT_ID))

        assert (before, after) == (False, True)
        assert cachedAfterMiss == 0
        # Nothing was cached for the miss, so the second call read again.
        assert firestoreReads == [("users", OTHER_ACCOUNT_ID)] * 2

    def testEachCollectionHasItsOwnLruCache(
        self, freshCache: MemoryDb
    ) -> None:
        for collection in (
            "users",
            "projects",
            "project_members",
            "rags",
            "rag_jobs",
        ):
            stats = freshCache.getCollectionStats(collection)
            assert stats.evictionPolicy == "lru"
            assert stats.maxRecords == 10_000
