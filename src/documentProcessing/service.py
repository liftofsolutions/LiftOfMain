"""Rag jobs and rags: create, submit, follow, list and delete them.

Records follow `db/knowledge-schema` and are read and written through
`utilities`. A submitted job is run by the `JobManager`.

Errors are returned, never raised: `{"error": message, "errorKind": kind}`,
logged, with `errorKind` one of the `ERROR_*` constants for the route to
turn into a status code.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone

from DbAdapters import Document, deleteOperation, setOperation
from DbAdapters.pineconeDb import getPineconeDb
from DbAdapters.pineconeDb.pineconeDb import IDS_PER_REQUEST
from documentProcessing import ragStrategy, utilities
from documentProcessing.jobManager import getJobManager
from documentProcessing.ragJob import (
    JOB_STATUS_DONE,
    JOB_STATUS_FAILED,
    JOB_STATUS_RUNNING,
    RagJob,
)

LOGGER = logging.getLogger(__name__)

ERROR_NOT_FOUND = "notFound"
ERROR_FORBIDDEN = "forbidden"
ERROR_CONFLICT = "conflict"
# Pinecone did not finish its part; doing it again picks up what is left.
ERROR_INCOMPLETE = "incomplete"

RAG_STATUS_INDEXING = "indexing"
# Being deleted: searches use ready rags only, so it is out of them.
RAG_STATUS_DELETING = "deleting"
JOB_TYPE_INDEX = "index"
JOB_STATUS_CREATED = "created"

# A job's `expires_at` while it is open, for the TTL that deletes it.
OPEN_JOB_RETENTION = timedelta(days=1)
# Runs a job may have, as `db/knowledge-schema` says: one that has failed
# this many times is not run again.
MAX_JOB_ATTEMPTS = 3


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


async def checkAccess(accountUuid: str, projectId: str) -> dict | None:
    """The error for an account that may not use `projectId`, else None.

    In this order: the account must exist, the project must be valid, and
    the account must be a member of it.
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
    return None


async def fetchProjectRecord(
    collection: str, projectId: str, recordId: str
) -> dict | None:
    """The record under `recordId`, if it belongs to `projectId`.

    Read from Firestore itself, not the cache, as a job's or a rag's
    status moves.
    """
    stored = await utilities.getFirestoreAdapter().fetchDocument(
        collection, recordId
    )
    if stored is None or stored.data.get("project_id") != projectId:
        return None
    return stored.data


async def createRagJob(accountUuid: str, projectId: str) -> dict:
    """Open a rag and a job for it under `accountUuid`'s `projectId`.

    Database reads and writes only: once `checkAccess` lets the account in,
    a rag id and a job id are drawn, each checked against what is already
    stored, and both records are written in one batch, so neither exists
    without the other.
    """
    denied = await checkAccess(accountUuid, projectId)
    if denied is not None:
        return denied

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
    documentLink: str,
    countryId: str | None,
    metadata: dict,
) -> dict:
    """Hand a job, with its document's link, to the job manager.

    Only the job is verified - it exists under `projectId` - since creating
    it checked the account, the project and the membership. Its rag must
    still be there and not being deleted, or the job would write into a
    rag on its way out. Then the job's state:
    a job its record says is done is answered as completed, one that has
    failed MAX_JOB_ATTEMPTS times as out of tries, and one the manager
    holds, waiting or running, as still processing; none is run again. Any
    other job runs: a new one, one that failed fewer times, or one a
    restart stopped part way, since the manager holds its jobs in memory
    only. Each run counts in `attempts`.

    The link is checked by the job when it downloads; a link it cannot take
    fails the job, with the reason as its error. The manager queues the job
    and runs it when a slot is free; this returns at once. No country means
    knowledge that applies to every country.
    """
    ragJob = await fetchProjectRecord(
        utilities.RAG_JOBS, projectId, jobId
    )
    if ragJob is None:
        return errorResult(
            ERROR_NOT_FOUND, f"Project {projectId!r} has no job {jobId!r}."
        )
    ragId = ragJob["rag_id"]
    rag = await fetchProjectRecord(utilities.RAGS, projectId, ragId)
    if rag is None:
        return errorResult(
            ERROR_NOT_FOUND, f"The rag {ragId!r} of job {jobId!r} is gone."
        )
    if rag.get("status") == RAG_STATUS_DELETING:
        return errorResult(
            ERROR_CONFLICT,
            f"Rag {ragId!r} is being deleted; job {jobId!r} cannot run.",
        )
    status = ragJob.get("status")
    if status == JOB_STATUS_DONE:
        return errorResult(
            ERROR_CONFLICT, f"Job {jobId!r} is already completed."
        )
    attempts = ragJob.get("attempts", 0)
    if status == JOB_STATUS_FAILED and attempts >= MAX_JOB_ATTEMPTS:
        return errorResult(
            ERROR_CONFLICT,
            f"Job {jobId!r} has failed {attempts} times, the most it may; "
            "create a new job.",
        )

    job = RagJob(
        jobId=jobId,
        accountUuid=accountUuid,
        projectId=projectId,
        ragId=ragId,
        documentLink=documentLink,
        countryId=countryId,
        metadata=metadata,
        createdAt=utcNow(),
    )
    if not getJobManager().createJob(job):
        return errorResult(
            ERROR_CONFLICT, f"Job {jobId!r} is still processing."
        )
    return {"jobId": jobId, "ragId": job.ragId, "status": job.status}


async def getRagJobStatus(
    accountUuid: str, projectId: str, jobId: str
) -> dict:
    """Where the job has got to, for a member of its project.

    `checkAccess` lets the account in first. A job of another project is
    answered as missing, the same as one that does not exist.
    """
    denied = await checkAccess(accountUuid, projectId)
    if denied is not None:
        return denied
    ragJob = await fetchProjectRecord(
        utilities.RAG_JOBS, projectId, jobId
    )
    if ragJob is None:
        return errorResult(
            ERROR_NOT_FOUND, f"Project {projectId!r} has no job {jobId!r}."
        )
    return {
        "jobId": jobId,
        "ragId": ragJob["rag_id"],
        "status": ragJob["status"],
    }


async def deleteRagJob(accountUuid: str, projectId: str, jobId: str) -> dict:
    """Delete a job that has not started, and the empty rag made with it.

    For a member of its project, as `checkAccess` checks. Only a job its
    record calls `created`, that the manager is not running, may go: one
    that runs, or has finished, done or failed, is refused. One waiting in
    the manager's queue is taken off it, so it never starts. Since it never
    ran, its rag holds nothing; the two are deleted in one batch, so no rag
    is left without its job.
    """
    denied = await checkAccess(accountUuid, projectId)
    if denied is not None:
        return denied
    ragJob = await fetchProjectRecord(utilities.RAG_JOBS, projectId, jobId)
    if ragJob is None:
        return errorResult(
            ERROR_NOT_FOUND, f"Project {projectId!r} has no job {jobId!r}."
        )
    jobManager = getJobManager()
    status = ragJob.get("status")
    if jobManager.isRunning(jobId):
        status = JOB_STATUS_RUNNING
    if status != JOB_STATUS_CREATED:
        return errorResult(
            ERROR_CONFLICT,
            f"Job {jobId!r} is {status}; only a job that has not started "
            "may be deleted.",
        )

    interrupted = jobManager.isHeld(jobId)
    if interrupted:
        jobManager.interruptAndDelete(jobId)
    ragId = ragJob["rag_id"]
    await utilities.getFirestoreAdapter().applyAtomically(
        [
            deleteOperation(utilities.RAG_JOBS, jobId),
            deleteOperation(utilities.RAGS, ragId),
        ]
    )
    LOGGER.info(
        "Deleted job %s and its rag %s; taken off the queue: %s",
        jobId,
        ragId,
        interrupted,
    )
    return {"jobId": jobId, "interrupted": interrupted}


def ragSummaryOf(ragId: str, rag: dict) -> dict:
    """A rag's record as it is listed: its state, without its text."""
    return {
        "ragId": ragId,
        "status": rag.get("status"),
        "strategy": rag.get("strategy"),
        "documentLink": rag.get("document_link"),
        "tokenCount": rag.get("token_count"),
        "chunkCount": rag.get("chunk_count"),
        "createdAt": rag.get("created_at"),
        "updatedAt": rag.get("updated_at"),
    }


async def listRags(accountUuid: str, projectId: str) -> dict:
    """Every rag of `projectId`, for a member of it.

    `checkAccess` lets the account in first. The rags are read from
    Firestore itself, not the cache, as their statuses move; a plain text
    rag's text is left out, as it can be long.
    """
    denied = await checkAccess(accountUuid, projectId)
    if denied is not None:
        return denied
    stored = await utilities.getFirestoreAdapter().findDocuments(
        utilities.RAGS, equals={"project_id": projectId}, limit=0
    )
    rags = [ragSummaryOf(document.id, document.data) for document in stored]
    LOGGER.info("Listed %s rags of project %s.", len(rags), projectId)
    return {"projectId": projectId, "rags": rags}


def heldJobIdOf(jobs: list[Document]) -> str | None:
    """The first of `jobs` the manager holds, waiting or running, if any."""
    return next(
        (job.id for job in jobs if getJobManager().isHeld(job.id)), None
    )


async def deleteChunkBatch(ragId: str, chunkIds: list[str]) -> bool:
    """Delete one batch of a rag's chunks in Pinecone; whether it worked."""
    try:
        await getPineconeDb().delete(ragId, chunkIds)
    except Exception:
        LOGGER.exception(
            "Deleting %s chunks of rag %s failed.", len(chunkIds), ragId
        )
        return False
    return True


async def deleteRag(accountUuid: str, projectId: str, ragId: str) -> dict:
    """Delete a rag: its chunks in Pinecone, then its records in Firestore.

    For a member of its project, as `checkAccess` checks. A rag whose job
    the manager still holds, waiting or running, is refused. Otherwise:

    1. The rag is marked `deleting`, so searches stop using it and its
       jobs can no longer be submitted; the jobs are checked once more,
       for a submit that read the rag just before.
    2. Its chunk ids are gathered: every id its namespace lists, and those
       its `chunk_count` gives, as a listing can lag a save just made.
    3. They are deleted in batches of IDS_PER_REQUEST, each its own task,
       all at once, and every task's outcome is awaited.
    4. Only if all succeed are the rag's record and its jobs' records
       deleted, in one batch. If any fails, the rag stays `deleting`, and
       deleting it again picks up what is left.
    """
    denied = await checkAccess(accountUuid, projectId)
    if denied is not None:
        return denied
    rag = await fetchProjectRecord(utilities.RAGS, projectId, ragId)
    if rag is None:
        return errorResult(
            ERROR_NOT_FOUND, f"Project {projectId!r} has no rag {ragId!r}."
        )
    jobs = await utilities.getFirestoreAdapter().findDocuments(
        utilities.RAG_JOBS, equals={"rag_id": ragId}
    )
    heldJobId = heldJobIdOf(jobs)
    if heldJobId is None:
        await ragStrategy.saveRag(ragId, {"status": RAG_STATUS_DELETING})
        # A submit that read the rag just before it was disabled may have
        # queued its job meanwhile. That job's end writes the rag's status
        # back, so the rag is left to it.
        heldJobId = heldJobIdOf(jobs)
    if heldJobId is not None:
        return errorResult(
            ERROR_CONFLICT,
            f"Rag {ragId!r} has job {heldJobId!r} still processing; "
            "delete the rag once the job has finished.",
        )

    try:
        listedIds = await getPineconeDb().listIds(ragId)
    except Exception:
        LOGGER.exception("Listing the chunks of rag %s failed.", ragId)
        return errorResult(
            ERROR_INCOMPLETE,
            f"The chunks of rag {ragId!r} could not be listed. It stays "
            "disabled; delete it again to finish.",
        )
    countedIds = {
        ragStrategy.recordIdOf(ragId, chunkIndex)
        for chunkIndex in range(rag.get("chunk_count", 0))
    }
    chunkIds = sorted(set(listedIds) | countedIds)
    batches = [
        chunkIds[start : start + IDS_PER_REQUEST]
        for start in range(0, len(chunkIds), IDS_PER_REQUEST)
    ]
    outcomes = await asyncio.gather(
        *(deleteChunkBatch(ragId, batch) for batch in batches)
    )
    failedCount = sum(
        len(batch)
        for batch, deleted in zip(batches, outcomes, strict=True)
        if not deleted
    )
    if failedCount:
        return errorResult(
            ERROR_INCOMPLETE,
            f"{failedCount} of {len(chunkIds)} chunks of rag {ragId!r} "
            "could not be deleted. It stays disabled; delete it again to "
            "finish.",
        )

    jobIds = [job.id for job in jobs]
    await utilities.getFirestoreAdapter().applyAtomically(
        [deleteOperation(utilities.RAGS, ragId)]
        + [deleteOperation(utilities.RAG_JOBS, jobId) for jobId in jobIds]
    )
    LOGGER.info(
        "Deleted rag %s: %s chunks, and jobs %s.",
        ragId,
        len(chunkIds),
        jobIds,
    )
    return {"ragId": ragId}
