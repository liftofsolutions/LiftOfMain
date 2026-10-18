"""Checks that the ids a request names exist and belong together.

Every check reads its records through `fetchRecord`:

    call -> cache check -> cached: return it
                        -> not cached: fetch from Firestore, cache, return

Firestore holds the record of truth, in the collections of
`db/user-schema`, `db/project-schema` and `db/knowledge-schema`. A record
Firestore does not have is not cached, so one made a moment later is still
found. Each collection gets its own cache in memoryDb, registered when this
module is imported.
"""

from config import buildFirestoreAdapter
from DbAdapters import FirestoreAdapter
from DbAdapters.memoryDb import EVICTION_LRU, CollectionConfig, getMemoryDb

USERS = "users"
PROJECTS = "projects"
PROJECT_MEMBERS = "project_members"
RAGS = "rags"
RAG_JOBS = "rag_jobs"
CACHED_COLLECTIONS = (USERS, PROJECTS, PROJECT_MEMBERS, RAGS, RAG_JOBS)

# Records each collection's cache keeps before it drops the least used.
CACHE_MAX_RECORDS = 10_000
CACHE_CONFIG = CollectionConfig(
    evictionPolicy=EVICTION_LRU, maxRecords=CACHE_MAX_RECORDS
)

_FIRESTORE_ADAPTER: FirestoreAdapter | None = None


def registerCaches() -> None:
    """Give each cached collection its own cache in memoryDb."""
    memoryDb = getMemoryDb()
    for collection in CACHED_COLLECTIONS:
        memoryDb.createCollection(collection, CACHE_CONFIG)


def getFirestoreAdapter() -> FirestoreAdapter:
    """The one Firestore adapter every fetch goes through."""
    global _FIRESTORE_ADAPTER
    if _FIRESTORE_ADAPTER is None:
        _FIRESTORE_ADAPTER = buildFirestoreAdapter()
    return _FIRESTORE_ADAPTER


async def fetchRecord(collection: str, recordId: str) -> dict | None:
    """The record under `recordId`, from the cache or else Firestore."""
    memoryDb = getMemoryDb()
    record = memoryDb.getRecord(collection, recordId)
    if record is not None:
        return record

    stored = await getFirestoreAdapter().fetchDocument(collection, recordId)
    if stored is None:
        return None
    memoryDb.addRecord(collection, recordId, stored.data)
    return stored.data


async def fetchAccount(accountId: str) -> dict | None:
    """The account's user record."""
    return await fetchRecord(USERS, accountId)


async def fetchProject(projectId: str) -> dict | None:
    """The project's record."""
    return await fetchRecord(PROJECTS, projectId)


async def fetchProjectMember(projectId: str, accountId: str) -> dict | None:
    """The account's membership of the project, keyed as the schema keys it."""
    return await fetchRecord(PROJECT_MEMBERS, f"{projectId}_{accountId}")


async def fetchRag(ragId: str) -> dict | None:
    """The rag's record."""
    return await fetchRecord(RAGS, ragId)


async def fetchRagJob(ragJobId: str) -> dict | None:
    """The rag job's record."""
    return await fetchRecord(RAG_JOBS, ragJobId)


async def isAccountIdValid(accountId: str) -> bool:
    """Whether `accountId` names an account."""
    return await fetchAccount(accountId) is not None


async def isProjectIdValid(projectId: str) -> bool:
    """Whether `projectId` names a project that has not been deleted."""
    project = await fetchProject(projectId)
    return project is not None and project.get("deleted_at") is None


async def isProjectIdAssociatedWithAccountId(
    projectId: str, accountId: str
) -> bool:
    """Whether `accountId` is a member of `projectId`, a valid project."""
    if not await isProjectIdValid(projectId):
        return False
    return await fetchProjectMember(projectId, accountId) is not None


async def isRagJobIdAssociatedWithRagId(ragJobId: str, ragId: str) -> bool:
    """Whether `ragJobId` names a job of the rag `ragId`."""
    ragJob = await fetchRagJob(ragJobId)
    return ragJob is not None and ragJob.get("rag_id") == ragId


async def isRagIdAssociatedWithProjectId(ragId: str, projectId: str) -> bool:
    """Whether `ragId` names a rag of the project `projectId`."""
    rag = await fetchRag(ragId)
    return rag is not None and rag.get("project_id") == projectId


registerCaches()
