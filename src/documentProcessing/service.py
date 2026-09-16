"""Rag database creation, document submission and job status.

Job records live in `store` (an in-memory dict mirrored into Firestore) -
see its docstring for why there is no Redis and no separate job store here.
A job is keyed by jobId and carries the account and project that opened it,
so a project can own many of them: one per country, or one shared under the
default country, `DEFAULT_COUNTRY`.

Only creation lives here for now; submission, the job runner and status
reads are still to be written.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone

from documentProcessing import store

LOGGER = logging.getLogger(__name__)

# Firestore paths, hung off one root so every record this service writes
# sits under it. A project's jobs get their own collection, so the ids
# taken under one account's project are read without filtering.
RAG_DATABASE_ROOT = "ragDatabase"
USER_PROJECTS_PATH = RAG_DATABASE_ROOT + "/userProjects/{accountUuid}"
JOBS_PATH = RAG_DATABASE_ROOT + "/Jobs/{accountUuid}/{projectId}/jobs"

DEFAULT_COUNTRY = "all"

# How long a job record is kept. Written onto every job as `expiresAt`,
# for Firestore's TTL policy to delete on - 20 minutes is the cap, so a job
# opened and never submitted does not linger.
JOB_TTL = timedelta(minutes=20)

JOB_STATUS_CREATED = "created"
JOB_STATUS_QUEUED = "queued"
JOB_STATUS_PROCESSING = "processing"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"


class RagDatabaseNotFoundError(LookupError):
    """No rag database is stored under the given ragDbId."""


def utcNow() -> datetime:
    """Current UTC time, kept behind a function so tests can freeze it."""
    return datetime.now(timezone.utc)


def getNewId(takenIds: set[str]) -> str:
    """A fresh `uuid4`, redrawn until it is none of `takenIds`."""
    candidate = uuid.uuid4().hex
    while candidate in takenIds:
        candidate = uuid.uuid4().hex
    return candidate


async def createRagDatabase(accountUuid: str, projectId: str) -> dict:
    """Open a job under `accountUuid`'s `projectId` and hand back its id.

    The account's project mapping is read first: an account asking for a
    project it does not own gets `{"error": ...}` back, logged and with no
    job opened, rather than an empty slice of jobs to add to. Only then are
    the job ids that project has already used read, once, so a collision
    costs another draw rather than another read.

    The job holds nothing but its ids and status; the document, country and
    metadata reach it through `submitDocument`, which is also what queues
    it - until then it sits at `created`.
    """
    projectsPath = USER_PROJECTS_PATH.format(accountUuid=accountUuid)
    if store.fetchRecord(projectsPath, projectId) is None:
        message = f"Account {accountUuid!r} has no project {projectId!r}."
        LOGGER.error(message)
        return {"error": message}

    jobsPath = JOBS_PATH.format(
        accountUuid=accountUuid, projectId=projectId
    )
    jobId = getNewId(store.fetchRecordIds(jobsPath, {}))

    now = utcNow()
    createdAt = now.isoformat()
    jobRecord = {
        "jobId": jobId,
        "accountUuid": accountUuid,
        "projectId": projectId,
        "status": JOB_STATUS_CREATED,
        "createdAt": createdAt,
        "updatedAt": createdAt,
        "expiresAt": (now + JOB_TTL).isoformat(),
    }
    await store.writeRecord(jobsPath, jobId, jobRecord)
    LOGGER.info("Opened job at %s: %s", jobsPath, jobRecord)
    return {
        "jobId": jobId,
        "projectId": projectId,
        "status": JOB_STATUS_CREATED,
        "createdAt": createdAt,
    }
