"""Btest for job creation, run against the real Firestore test database.

Unlike most Btests this one talks to Firestore rather than a patched store:
the point is that a created job really lands in the database, under the path
the service builds. `service.createRagDatabase` is called directly - no route,
no client - and the whole flow runs once, in one event loop, because the
adapter holds a gRPC channel bound to the loop that opened it.

Nothing is configured up there yet, so the test writes its own account to
project mapping first, then deletes everything it wrote - jobs and mapping
both - in a `finally`, so a failed assertion still leaves the database clean.
Ids carry a `uuid4` so two runs, or two machines, never collide.

    uv run pytest Btest/documentProcessing/routes -q
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest

from config import getSettings
from documentProcessing import service, store

LOGGER = logging.getLogger(__name__)

ACCOUNT_UUID = f"btest-{uuid.uuid4().hex}"
PROJECT_ID = f"project-{uuid.uuid4().hex}"
UNKNOWN_PROJECT_ID = f"missing-{uuid.uuid4().hex}"

ID_LENGTH = 32

# The cap the job is allowed to live for, spelled out rather than read from
# `service.JOB_TTL` - asserting against that constant would pass whatever it
# was changed to.
MAX_TTL = timedelta(minutes=20)


@dataclass
class CreateOutcome:
    """What the flow returned and what Firestore held, before cleanup."""

    created: dict
    storedJob: dict | None
    missingProject: dict
    missingProjectJobs: set[str]


def projectsPathOf(accountUuid: str) -> str:
    """Collection holding one document per project the account owns."""
    return service.USER_PROJECTS_PATH.format(accountUuid=accountUuid)


def jobsPathOf(accountUuid: str, projectId: str) -> str:
    """Collection holding the jobs opened under one account's project."""
    return service.JOBS_PATH.format(
        accountUuid=accountUuid, projectId=projectId
    )


async def cleanUp(adapter, projectsPath: str, jobsPath: str) -> None:
    """Delete everything the run wrote, in Firestore and in memory.

    The job ids come from the in-memory store, which mirrors exactly the
    writes this run made, so nothing else in the collection is touched.
    """
    for jobId in list(store.IN_MEMORY_STORE.get(jobsPath, {})):
        LOGGER.info("Deleting job %s/%s.", jobsPath, jobId)
        await adapter.deleteDocument(jobsPath, jobId)
    LOGGER.info("Deleting mapping %s/%s.", projectsPath, PROJECT_ID)
    await adapter.deleteDocument(projectsPath, PROJECT_ID)
    await adapter.close()
    store._FIRESTORE_ADAPTER = None
    store.IN_MEMORY_STORE.clear()


async def runCreateFlow() -> CreateOutcome:
    """Configure the mapping, create a job, read it back, then clean up."""
    projectsPath = projectsPathOf(ACCOUNT_UUID)
    jobsPath = jobsPathOf(ACCOUNT_UUID, PROJECT_ID)
    adapter = await store.getAdapter()

    LOGGER.info("Account %s, project %s.", ACCOUNT_UUID, PROJECT_ID)
    LOGGER.info("Jobs path %s.", jobsPath)

    try:
        await store.writeRecord(
            projectsPath, PROJECT_ID, {"projectId": PROJECT_ID}
        )
        LOGGER.info("Wrote mapping %s/%s.", projectsPath, PROJECT_ID)

        missingProject = await service.createRagDatabase(
            ACCOUNT_UUID, UNKNOWN_PROJECT_ID
        )
        missingProjectJobs = store.fetchRecordIds(
            jobsPathOf(ACCOUNT_UUID, UNKNOWN_PROJECT_ID), {}
        )
        LOGGER.info("Unknown project returned %s.", missingProject)
        LOGGER.info("Unknown project jobs %s.", missingProjectJobs)

        created = await service.createRagDatabase(ACCOUNT_UUID, PROJECT_ID)
        LOGGER.info("Create returned %s.", created)

        stored = await adapter.fetchDocument(jobsPath, created["jobId"])
        LOGGER.info(
            "Firestore holds %s.", None if stored is None else stored.data
        )

        return CreateOutcome(
            created=created,
            storedJob=None if stored is None else stored.data,
            missingProject=missingProject,
            missingProjectJobs=missingProjectJobs,
        )
    finally:
        await cleanUp(adapter, projectsPath, jobsPath)


@pytest.fixture(scope="module")
def createOutcome() -> CreateOutcome:
    """The flow, run once against Firestore for every test in this module."""
    if not getSettings().hasFirestore:
        pytest.skip("Firestore is not configured for this environment.")
    return asyncio.run(runCreateFlow())


class TestCreateJob:
    """A job is opened under an account's project and handed back."""

    def testReturnsNewJobId(self, createOutcome: CreateOutcome) -> None:
        jobId = createOutcome.created["jobId"]
        assert len(jobId) == ID_LENGTH
        assert jobId == uuid.UUID(jobId).hex

    def testReturnsCreatedStatus(
        self, createOutcome: CreateOutcome
    ) -> None:
        assert createOutcome.created["status"] == service.JOB_STATUS_CREATED
        assert createOutcome.created["projectId"] == PROJECT_ID
        assert datetime.fromisoformat(createOutcome.created["createdAt"])


class TestJobRecordInFirestore:
    """The job is really in the database, under the path built for it."""

    def testJobIsStored(self, createOutcome: CreateOutcome) -> None:
        assert createOutcome.storedJob is not None

    def testStoresEveryAttribute(
        self, createOutcome: CreateOutcome
    ) -> None:
        stored = createOutcome.storedJob
        assert stored["jobId"] == createOutcome.created["jobId"]
        assert stored["accountUuid"] == ACCOUNT_UUID
        assert stored["projectId"] == PROJECT_ID
        assert stored["status"] == service.JOB_STATUS_CREATED
        assert stored["createdAt"] == createOutcome.created["createdAt"]
        assert stored["updatedAt"] == stored["createdAt"]

    def testExpiresTwentyMinutesAfterCreation(
        self, createOutcome: CreateOutcome
    ) -> None:
        stored = createOutcome.storedJob
        expiresAt = datetime.fromisoformat(stored["expiresAt"])
        createdAt = datetime.fromisoformat(stored["createdAt"])
        assert expiresAt - createdAt == MAX_TTL
        assert expiresAt > createdAt


class TestUnknownProject:
    """An account asking for a project it does not own gets an error."""

    def testReturnsError(self, createOutcome: CreateOutcome) -> None:
        error = createOutcome.missingProject["error"]
        assert ACCOUNT_UUID in error
        assert UNKNOWN_PROJECT_ID in error

    def testOpensNoJob(self, createOutcome: CreateOutcome) -> None:
        assert "jobId" not in createOutcome.missingProject
        assert createOutcome.missingProjectJobs == set()
