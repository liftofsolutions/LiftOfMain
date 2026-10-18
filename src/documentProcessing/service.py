"""Rag jobs: create one, submit a document to it, follow it, delete it.

Creating a job is written; the rest is still to be designed, so those calls
report that they are not implemented yet. Records follow
`db/knowledge-schema` and are read and written through `utilities`.

Errors are returned, never raised: `{"error": message, "errorKind": kind}`,
logged, with `errorKind` one of the `ERROR_*` constants for the route to
turn into a status code.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone

from DbAdapters import Document, setOperation
from documentProcessing import utilities

LOGGER = logging.getLogger(__name__)

ERROR_NOT_FOUND = "notFound"
ERROR_FORBIDDEN = "forbidden"
ERROR_NOT_IMPLEMENTED = "notImplemented"

RAG_STATUS_INDEXING = "indexing"
JOB_TYPE_INDEX = "index"
JOB_STATUS_CREATED = "created"

# A job's `expires_at` while it is open, for the TTL that deletes it.
OPEN_JOB_RETENTION = timedelta(days=1)


def utcNow() -> datetime:
    """Current UTC time, kept behind a function so tests can freeze it."""
    return datetime.now(timezone.utc)


def newId() -> str:
    """A fresh UUID v4, as the schemas' ids are."""
    return str(uuid.uuid4())


async def generateUniqueId(collection: str) -> str:
    """A new id that no record in `collection` already has.

    Each id drawn is looked up first; one already taken is drawn again.
    """
    recordId = newId()
    while await utilities.fetchRecord(collection, recordId) is not None:
        recordId = newId()
    return recordId


def errorResult(errorKind: str, message: str) -> dict:
    """An error to hand back to the route, logged first."""
    LOGGER.error(message)
    return {"error": message, "errorKind": errorKind}


def notImplemented(operation: str) -> dict:
    """The error for an operation that has no logic yet, logged first."""
    return errorResult(
        ERROR_NOT_IMPLEMENTED, f"{operation} is not implemented yet."
    )


async def createRagJob(accountUuid: str, projectId: str) -> dict:
    """Open a rag and a job for it under `accountUuid`'s `projectId`.

    Database reads and writes only, in this order: the account must exist,
    the project must be valid, and the account must be a member of it.
    Then a rag id and a job id are drawn, each checked against what is
    already stored, and both records are written in one batch, so neither
    exists without the other.
    """
    if not await utilities.isAccountIdValid(accountUuid):
        return errorResult(
            ERROR_NOT_FOUND, f"Account {accountUuid!r} does not exist."
        )
    if not await utilities.isProjectIdValid(projectId):
        return errorResult(
            ERROR_NOT_FOUND, f"Project {projectId!r} does not exist."
        )
    if not await utilities.isProjectIdAssociatedWithAccountId(
        projectId, accountUuid
    ):
        return errorResult(
            ERROR_FORBIDDEN,
            f"Account {accountUuid!r} is not a member of project "
            f"{projectId!r}.",
        )

    ragId = await generateUniqueId(utilities.RAGS)
    jobId = await generateUniqueId(utilities.RAG_JOBS)
    now = utcNow()
    rag = {
        "_id": ragId,
        "project_id": projectId,
        "namespace": ragId,
        "status": RAG_STATUS_INDEXING,
        "created_by": accountUuid,
        "created_at": now,
        "updated_at": now,
    }
    job = {
        "_id": jobId,
        "project_id": projectId,
        "rag_id": ragId,
        "document_id": None,
        "type": JOB_TYPE_INDEX,
        "status": JOB_STATUS_CREATED,
        "attempts": 0,
        "expires_at": now + OPEN_JOB_RETENTION,
        "created_at": now,
        "updated_at": now,
    }
    await utilities.getFirestoreAdapter().applyAtomically(
        [
            setOperation(utilities.RAGS, Document(ragId, rag)),
            setOperation(utilities.RAG_JOBS, Document(jobId, job)),
        ]
    )
    LOGGER.info("Created rag %s and job %s", rag, job)
    return {"jobId": jobId, "ragId": ragId, "status": JOB_STATUS_CREATED}


async def submitRagJob(
    accountUuid: str,
    projectId: str,
    jobId: str,
    documentId: str,
    countryId: str | None,
    metadata: dict,
) -> dict:
    """Give a created job its document, country and metadata, and start it.

    No country means knowledge that applies to every country.
    """
    return notImplemented("Submitting a rag job")


async def getRagJobStatus(
    accountUuid: str, projectId: str, jobId: str
) -> dict:
    """Where the job has got to."""
    return notImplemented("Reading a rag job's status")


async def deleteRagJob(accountUuid: str, projectId: str, jobId: str) -> dict:
    """Stop the job if it is still going, then delete it."""
    return notImplemented("Deleting a rag job")


async def deleteRag(accountUuid: str, projectId: str, ragId: str) -> dict:
    """Delete the rag itself."""
    return notImplemented("Deleting a rag")
