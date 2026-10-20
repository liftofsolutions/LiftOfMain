"""The jobs this process holds, and when each of them runs.

    from documentProcessing.jobManager import getJobManager

    created = getJobManager().createJob(job)
    busy = getJobManager().isHeld(jobId)
    deleted = getJobManager().interruptAndDelete(jobId)

Three entry points, and nothing else is meant to be called from outside:

* `createJob` puts a job on the pending queue, then starts waiting jobs
  while fewer than `maxRunning` are running. When none can run, it just
  returns; the job waits.
* `isHeld` says whether a job is waiting or running here.
* `interruptAndDelete` forgets a job: a waiting one never runs, a running
  one is stopped. Its slot is freed and the next waiting job started at
  once, not when the stopped job has finished winding down.

A job is started by running its `start` in a task. Scheduling is driven
by the jobs themselves: each is given `onJobFinished` as its `onFinished`,
and calls it once it ends, done or failed, with that status and, if it
failed, the error. That writes
both onto the job, frees its slot and starts the next waiting job. A job
that is interrupted is called back for by `interruptAndDelete` instead,
and its own call back is taken away, so `onJobFinished` still runs once
per job.

Every call is O(1). A job deleted while it waits is not searched for in
the queue; it is left there and skipped when it reaches the front, since
it is no longer among the jobs held.

One manager per process - `getJobManager` - so the limit on running jobs
holds for the whole server. Its jobs are in memory only: a restart loses
them.
"""

import asyncio
import logging
from collections import deque

from documentProcessing.ragJob import JOB_STATUS_INTERRUPTED, RagJob, utcNow

LOGGER = logging.getLogger(__name__)

# Jobs running at once; the rest wait in the pending queue.
MAX_RUNNING_JOBS = 5


class JobManager:
    """A pending queue of jobs, run at most `maxRunning` at a time."""

    def __init__(self, maxRunning: int = MAX_RUNNING_JOBS) -> None:
        """An empty manager that runs up to `maxRunning` jobs at once."""
        self.maxRunning = maxRunning
        # Waiting to run, oldest first; may hold jobs since deleted.
        self.pendingQueue: deque[RagJob] = deque()
        # Holding a slot, by job id, until they call back.
        self.runningJobs: dict[str, RagJob] = {}
        # Every job waiting or running and not deleted, by job id.
        self.heldJobs: dict[str, RagJob] = {}

    def createJob(self, job: RagJob) -> bool:
        """Queue `job`, then start what can run.

        False, and nothing queued, if a job with its id is already held or
        still holds a slot.
        """
        if job.jobId in self.heldJobs or job.jobId in self.runningJobs:
            LOGGER.error("Job %s is already held; not queued.", job.jobId)
            return False
        job.onFinished = self.onJobFinished
        self.heldJobs[job.jobId] = job
        self.pendingQueue.append(job)
        LOGGER.info("Queued job: %s", job)
        self.startPendingJobs()
        return True

    def isHeld(self, jobId: str) -> bool:
        """Whether the job with `jobId` is waiting or running here."""
        return jobId in self.heldJobs

    def interruptAndDelete(self, jobId: str) -> bool:
        """Stop the job, forget it, and start the next waiting job.

        A waiting job is never started. A running one is cancelled and its
        slot handed to the next job straight away; the cancelled task may
        take until its next pause to actually stop. False if no such job is
        held.
        """
        job = self.heldJobs.get(jobId)
        if job is None:
            LOGGER.error("No job %s to interrupt.", jobId)
            return False
        # Called back for here, so the stopped task must not call again.
        job.onFinished = None
        job.interrupt()
        self.onJobFinished(job, JOB_STATUS_INTERRUPTED)
        LOGGER.info("Interrupted and deleted job %s.", jobId)
        return True

    def onJobFinished(
        self,
        job: RagJob,
        status: str | None = None,
        error: str | None = None,
    ) -> None:
        """A job's call back, with how it ended and why, if it gives them.

        Writes the status and the error, each when given, and the finish
        time onto the job, frees its slot, forgets it, and starts the next
        waiting job. With no status, the job keeps the one it has.
        """
        if status is not None:
            job.status = status
        if error is not None:
            job.error = error
        job.finishedAt = utcNow()
        LOGGER.info("Job finished: %s", job)
        if self.runningJobs.get(job.jobId) is job:
            del self.runningJobs[job.jobId]
        if self.heldJobs.get(job.jobId) is job:
            del self.heldJobs[job.jobId]
        self.startPendingJobs()

    def startPendingJobs(self) -> None:
        """Start waiting jobs, oldest first, until no slot is free.

        Returns as soon as every slot is taken or nothing is waiting.
        """
        while len(self.runningJobs) < self.maxRunning and self.pendingQueue:
            job = self.pendingQueue.popleft()
            if self.heldJobs.get(job.jobId) is not job:
                # Deleted while it waited.
                continue
            self.runningJobs[job.jobId] = job
            job.task = asyncio.create_task(job.start())


_JOB_MANAGER: JobManager | None = None


def getJobManager() -> JobManager:
    """The process wide job manager, built once and shared."""
    global _JOB_MANAGER
    if _JOB_MANAGER is None:
        _JOB_MANAGER = JobManager()
    return _JOB_MANAGER
