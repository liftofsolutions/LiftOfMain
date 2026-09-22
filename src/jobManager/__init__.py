"""Runs submitted jobs concurrently, behind a shared concurrency cap.

    from jobManager import getJobManager
    getJobManager().createJob(jobId, jobData)

The manager exposes three things: `createJob`, `getStatus` and
`interruptAndDelete`.
"""

from jobManager.manager import (
    DEFAULT_JOB_TTL,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_QUEUED,
    JOB_STATUS_RUNNING,
    MAX_CONCURRENT_JOBS,
    Job,
    JobManager,
    getJobManager,
)

__all__ = [
    "DEFAULT_JOB_TTL",
    "JOB_STATUS_COMPLETED",
    "JOB_STATUS_FAILED",
    "JOB_STATUS_QUEUED",
    "JOB_STATUS_RUNNING",
    "MAX_CONCURRENT_JOBS",
    "Job",
    "JobManager",
    "getJobManager",
]
