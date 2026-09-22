"""Holds the jobs this process is running, behind a concurrency cap.

Three things are exposed: `createJob` registers a job and starts it running,
`getStatus` reports where it got to, and `interruptAndDelete` stops it and
forgets it. The job list, the tasks and the semaphore are internal.

A job created above `MAX_CONCURRENT_JOBS` is not rejected - it waits at
`queued` until a running job finishes and frees a slot, so five documents
process at a time and the sixth starts the moment one of them is done.

Running a job lives here, in `runJob`; a `Job` owns its own status, writing
each move onto the stored record as it happens.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from documentProcessing import downloader, store

LOGGER = logging.getLogger(__name__)

# No more than this many jobs run at the same time; the rest queue for a slot.
MAX_CONCURRENT_JOBS = 5

# How long a job is worth keeping once created.
DEFAULT_JOB_TTL = timedelta(minutes=20)

JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"


def utcNow() -> datetime:
    """Current UTC time, kept behind a function so tests can freeze it."""
    return datetime.now(timezone.utc)


@dataclass
class Job:
    """One unit of work: what it was given, and where it has got to.

    A job moves itself between statuses and writes each move onto its
    stored record, so the database says the same thing the manager does
    without a caller having to keep the two in step. It holds the manager
    that created it, to be cleaned up by it when it cannot go on.
    """

    jobId: str
    jobData: dict[str, Any]
    ttl: timedelta
    createdAt: datetime
    # The manager this job belongs to. Not in the repr: a job prints as its
    # own fields, not as the whole manager and every other job on it.
    manager: "JobManager" = field(repr=False)
    status: str = JOB_STATUS_QUEUED
    error: str | None = None

    @property
    def expiresAt(self) -> datetime:
        """When this job stops being worth keeping."""
        return self.createdAt + self.ttl

    @property
    def jobsPath(self) -> str:
        """Where this job's record is stored, empty when it has none."""
        return self.jobData.get("jobsPath", "")

    def setStatus(self, status: str, error: str | None = None) -> None:
        """Move to `status`, then mirror it onto the stored record.

        The record is written in the background: a job's own progress is
        nothing a caller is waiting on. The record itself is not optional
        though - submission writes it before the job ever exists - so a job
        that cannot find its own is failed and cleaned up.
        """
        stored = store.fetchRecord(self.jobsPath, self.jobId)
        if stored is None:
            self.failMissingRecord()
            return

        self.status = status
        self.error = error
        LOGGER.info("Job %s is %s.", self.jobId, status)

        updates = {"status": status, "updatedAt": utcNow().isoformat()}
        if error is not None:
            updates["error"] = error
        store.writeRecordSoon(
            self.jobsPath, self.jobId, {**stored, **updates}
        )

    def failMissingRecord(self) -> None:
        """Fail this job because the record it writes to is not there.

        There is nowhere left to write progress and nothing to show for the
        work, so the manager is asked to stop the job and drop it. Only the
        log says what happened, which is why it is logged at error.
        """
        self.status = JOB_STATUS_FAILED
        self.error = f"No job record at {self.jobsPath!r}."
        LOGGER.error(
            "Job %s has no record at %s; failing it and cleaning it up.",
            self.jobId,
            self.jobsPath,
        )
        self.manager.interruptAndDelete(self.jobId)

    def toStatusReport(self) -> dict:
        """What `getStatus` hands back for this job."""
        return {
            "jobId": self.jobId,
            "status": self.status,
            "createdAt": self.createdAt.isoformat(),
            "expiresAt": self.expiresAt.isoformat(),
            "error": self.error,
        }


class JobManager:
    """Runs jobs behind a shared `asyncio.Semaphore`.

    Build one instance per process and share it - the cap only means
    something if every job created anywhere goes through the same one.

    Example:
        manager = JobManager()
        manager.createJob(jobId, {"documentLink": link, "jobsPath": path})
    """

    def __init__(self, maxConcurrentJobs: int = MAX_CONCURRENT_JOBS) -> None:
        """Set up the semaphore that enforces `maxConcurrentJobs`."""
        self._semaphore = asyncio.Semaphore(maxConcurrentJobs)
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def createJob(
        self,
        jobId: str,
        jobData: dict[str, Any],
        ttl: timedelta = DEFAULT_JOB_TTL,
    ) -> Job:
        """Register `jobId` and start it, queueing behind the cap.

        Returns as soon as the job is on the list; it may still be waiting
        for a slot when this returns, which is what `queued` means.
        """
        job = Job(
            jobId=jobId,
            jobData=jobData,
            ttl=ttl,
            createdAt=utcNow(),
            manager=self,
        )
        self._jobs[jobId] = job
        LOGGER.info("Created job %s with %s.", jobId, jobData)

        task = asyncio.create_task(self._workOn(job))
        self._tasks[jobId] = task
        task.add_done_callback(lambda _: self._tasks.pop(jobId, None))

        # Last, so that a job with no record to write to is cleaned up with
        # its task already on the list to cancel - and never runs.
        job.setStatus(JOB_STATUS_QUEUED)
        return job

    def getStatus(self, jobId: str) -> dict | None:
        """Where `jobId` has got to, or None when it is not on the list."""
        job = self._jobs.get(jobId)
        return None if job is None else job.toStatusReport()

    def interruptAndDelete(self, jobId: str) -> bool:
        """Stop `jobId` if it is still going and forget it.

        Returns whether there was a job to interrupt. Cancelling a task
        that has already finished does nothing, so this is safe either way.
        """
        task = self._tasks.pop(jobId, None)
        if task is not None:
            task.cancel()

        job = self._jobs.pop(jobId, None)
        if job is None:
            LOGGER.info("Nothing to interrupt for job %s.", jobId)
            return False
        LOGGER.info(
            "Interrupted job %s at %s, and deleted it.", jobId, job.status
        )
        return True

    async def runJob(self, job: Job) -> None:
        """Work `job` through its steps.

        TODO: unpack the download and collect its metadata. This stops at
        the download for now.
        """
        loadedFile = await downloader.downloadDocument(
            job.jobData["documentLink"]
        )
        LOGGER.info(
            "Job %s downloaded %s, %s bytes.",
            job.jobId,
            loadedFile.fileName,
            loadedFile.sizeBytes,
        )

    async def _workOn(self, job: Job) -> None:
        """Wait for a free slot, then run `job` while holding it.

        A failure ends its own job and nothing else: it is recorded on the
        job, which writes it to the record, and logged with its traceback.
        """
        async with self._semaphore:
            job.setStatus(JOB_STATUS_RUNNING)
            try:
                await self.runJob(job)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                job.setStatus(JOB_STATUS_FAILED, str(error))
                LOGGER.exception("Job %s failed.", job.jobId)
                return
            job.setStatus(JOB_STATUS_COMPLETED)


_JOB_MANAGER: JobManager | None = None


def getJobManager() -> JobManager:
    """The process wide job manager, built once and shared.

    Every job must be created through this instance, or the concurrency cap
    is only enforced against part of the workload.
    """
    global _JOB_MANAGER
    if _JOB_MANAGER is None:
        _JOB_MANAGER = JobManager()
    return _JOB_MANAGER
