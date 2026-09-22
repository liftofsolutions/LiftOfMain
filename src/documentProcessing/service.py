"""Rag database creation, document submission and job status.

Job records live in `store` (an in-memory dict mirrored into Firestore) -
see its docstring for why there is no Redis and no separate job store here.
A job is keyed by jobId and carries the account and project that opened it,
so a project can own many of them: one per country, or one shared under the
default country, `DEFAULT_COUNTRY`.

Creating a job only reserves its id. Submitting one hands it the document
and puts it on the `JobManager`, which runs at most five at a time and owns
everything that happens after that; status reads are still to be written.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone

from documentProcessing import store
from jobManager import JOB_STATUS_QUEUED, getJobManager

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

# What a job is before it has been submitted anything to work on. Every
# status after this one is the job manager's, and a `Job` writes it itself.
JOB_STATUS_CREATED = "created"


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


async def submitDocument(
    accountUuid: str,
    projectId: str,
    jobId: str,
    documentLink: str,
    country: str | None,
    metadata: dict,
) -> dict:
    """Hand `jobId` its document and put it on the job manager.

    The job has to exist first - `createRagDatabase` opens it - so a jobId
    this account's project never opened gets a logged `{"error": ...}` back
    and nothing is queued. Otherwise the job data is stored, the job is
    scheduled, and this returns without waiting for it to run.

    Nothing in the reply depends on the stored record having reached
    Firestore, so the job data is written to memory and mirrored in the
    background rather than costing the caller a network round trip.
    """

    jobsPath = JOBS_PATH.format(
        accountUuid=accountUuid, projectId=projectId
    )
    stored = store.fetchRecord(jobsPath, jobId)
    if stored is None:
        message = (
            f"Account {accountUuid!r} has no job {jobId!r} under project "
            f"{projectId!r}."
        )
        LOGGER.error(message)
        return {"error": message}

    jobData = {
        "documentLink": documentLink,
        "country": country or DEFAULT_COUNTRY,
        "metadata": metadata,
    }
    store.writeRecordSoon(
        jobsPath,
        jobId,
        {
            **stored,
            "jobData": jobData,
            "status": JOB_STATUS_QUEUED,
            "updatedAt": utcNow().isoformat(),
        },
    )

    # The job is told where its record lives; the record itself has no use
    # for a path that is its own address.
    getJobManager().createJob(
        jobId, {**jobData, "jobsPath": jobsPath}, ttl=JOB_TTL
    )
    LOGGER.info("Submitted job %s with %s.", jobId, jobData)

    return {
        "jobId": jobId,
        "status": JOB_STATUS_QUEUED,
        "message": "Job submitted successfully.",
    }
