"""One rag job: what it was submitted with, and where it has got to.

`submitRagJob` makes a `RagJob` and hands it to the `JobManager`, which
gives it `onFinished` and starts it when a slot is free. The job marks
itself running; once it is over it calls `onFinished` with how it ended -
done, failed or interrupted - and the manager writes that status onto the
job, frees its slot and starts the next job waiting.

The call back is made from the task's done callback, not from inside
`run`, so it happens however the task ends: a task cancelled before it
ever started runs none of its own code, so a `finally` in `run` would
never be reached and the slot would stay taken.

What a job does - downloading and indexing its document - belongs in
`process`, which does nothing until that work is written.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

LOGGER = logging.getLogger(__name__)

# Waiting in the manager's queue for a slot.
JOB_STATUS_SUBMITTED = "submitted"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_DONE = "done"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_INTERRUPTED = "interrupted"


def utcNow() -> datetime:
    """Current UTC time, kept behind a function so tests can freeze it."""
    return datetime.now(timezone.utc)


# Compared by identity: the manager tells a job from a later one that
# reuses its id.
@dataclass(eq=False)
class RagJob:
    """A submitted rag job, its progress, and its hooks to the manager."""

    # What the job was submitted with.
    jobId: str
    accountUuid: str
    projectId: str
    ragId: str
    documentId: str
    countryId: str | None = None
    metadata: dict = field(default_factory=dict)

    # Where it has got to.
    status: str = JOB_STATUS_SUBMITTED
    createdAt: datetime = field(default_factory=utcNow)
    startedAt: datetime | None = None
    finishedAt: datetime | None = None
    error: str | None = None

    # Given by the manager: called once, when the job is over, with the
    # status it ended in.
    onFinished: Callable[["RagJob", str], None] | None = field(
        default=None, repr=False
    )
    # The task running the job, once it has been started.
    task: asyncio.Task | None = field(default=None, repr=False)

    def start(self) -> None:
        """Run the job in a task, calling back when the task is over."""
        self.task = asyncio.create_task(self.run())
        self.task.add_done_callback(self.finish)

    async def run(self) -> None:
        """Do the job's work, keeping the error if it fails.

        How the job ended is worked out by `finish`. An interruption is not
        caught here: it cancels the task.
        """
        self.status = JOB_STATUS_RUNNING
        self.startedAt = utcNow()
        LOGGER.info("Job started: %s", self)
        try:
            await self.process()
        except Exception as error:
            LOGGER.exception("Job %s failed.", self.jobId)
            self.error = str(error)

    async def process(self) -> None:
        """The job's work: downloading and indexing its document.

        Not written yet, so a job finishes as soon as it starts.
        """

    def interrupt(self) -> None:
        """Cancel the job's task, if it has one still running."""
        if self.task is not None and not self.task.done():
            self.task.cancel()

    def finish(self, task: asyncio.Task) -> None:
        """Call the manager back with how the job ended."""
        if task.cancelled():
            status = JOB_STATUS_INTERRUPTED
        elif self.error is not None:
            status = JOB_STATUS_FAILED
        else:
            status = JOB_STATUS_DONE
        if self.onFinished is not None:
            self.onFinished(self, status)
