"""One rag job: what it was submitted with, and where it has got to.

`submitRagJob` makes a `RagJob` and hands it to the `JobManager`, which
gives it `onFinished` and, when a slot is free, runs `start` in a task with
`finish` as the task's done callback. `start` is the whole job: it marks
the job running, downloads the document from the link it was submitted
with, has `documentAnalyzer` work out what the document holds, file by
file, then takes the strategy `ragStrategy` picks by its token count.
Plain text is saved on the rag's record. A rag's text is cut into
`chunks`; sending them to Pinecone, which embeds them, is still to be
written.

`start` does not catch errors: one - a link to a format not taken, a
download that fails, a document nothing can be read from - ends the task.
`finish` then works out how the job ended and calls `onFinished` with the
status and, for a failure, the error; the manager writes both onto the job,
frees its slot and starts the next job waiting.

The call back is made from the task's done callback, not from inside
`start`, so it happens however the task ends: a task cancelled before it
ever ran executes none of its own code, so a `finally` in `start` would
never be reached and the slot would stay taken.

The download, and anything unpacked from it, stays in the job's folder
while the job runs. A job that ends done or failed removes them; an
interrupted one leaves them, as its analysis thread may still be reading.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from documentProcessing import documentAnalyzer, downloader, ragStrategy
from documentProcessing.documentAnalyzer import DocumentAnalysis
from documentProcessing.ragStrategy import Chunk

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
    documentLink: str
    countryId: str | None = None
    metadata: dict = field(default_factory=dict)

    # Where it has got to.
    status: str = JOB_STATUS_SUBMITTED
    createdAt: datetime = field(default_factory=utcNow)
    startedAt: datetime | None = None
    finishedAt: datetime | None = None
    error: str | None = None

    # What processing found: where the document was saved, what it holds,
    # the strategy it takes, and for a rag, its chunks.
    downloadPath: Path | None = None
    analysis: DocumentAnalysis | None = field(default=None, repr=False)
    strategy: str | None = None
    chunks: list[Chunk] = field(default_factory=list, repr=False)

    # Given by the manager: called once, when the job is over, with the
    # status it ended in and the error, if it failed.
    onFinished: Callable[["RagJob", str, str | None], None] | None = field(
        default=None, repr=False
    )
    # The task the manager runs `start` in.
    task: asyncio.Task | None = field(default=None, repr=False)

    async def start(self) -> None:
        """Do the job: download, analyze, then take the document's strategy.

        Parsing and chunking run in a thread, since they block.
        Errors are left to end the task; `finish` passes them on.
        """
        self.status = JOB_STATUS_RUNNING
        self.startedAt = utcNow()
        LOGGER.info("Job started: %s", self)

        self.downloadPath = await downloader.downloadDocument(
            self.documentLink, self.jobId
        )
        LOGGER.info("Job %s downloaded %s.", self.jobId, self.downloadPath)

        self.analysis = await asyncio.to_thread(
            documentAnalyzer.analyzeDocument,
            self.downloadPath,
            downloader.fileNameOf(self.documentLink),
        )
        LOGGER.info(
            "Job %s analyzed %s: %s; skipped %s",
            self.jobId,
            self.analysis.fileName,
            self.analysis.totals(),
            self.analysis.skippedFiles,
        )

        self.strategy = ragStrategy.selectStrategy(
            self.analysis.totals()["tokenCount"]
        )
        if self.strategy == ragStrategy.STRATEGY_PLAIN_TEXT:
            await ragStrategy.savePlainText(
                self.ragId, self.documentLink, self.analysis
            )
        else:
            self.chunks = await asyncio.to_thread(
                ragStrategy.chunkDocument, self.analysis
            )
        LOGGER.info(
            "Job %s took the %s strategy: %s chunks.",
            self.jobId,
            self.strategy,
            len(self.chunks),
        )

    def interrupt(self) -> None:
        """Cancel the job's task, if it has one still running."""
        if self.task is not None and not self.task.done():
            self.task.cancel()

    def finish(self, task: asyncio.Task) -> None:
        """Call the manager back with how the job ended, and any error.

        A done or failed job's folder is removed: whatever read it has
        finished by the time the task has, threads included. An
        interrupted job's is left, as its thread may still be reading it.
        """
        error = None
        if task.cancelled():
            status = JOB_STATUS_INTERRUPTED
        elif task.exception() is not None:
            status = JOB_STATUS_FAILED
            error = str(task.exception())
            LOGGER.error(
                "Job %s failed.", self.jobId, exc_info=task.exception()
            )
        else:
            status = JOB_STATUS_DONE
        if status != JOB_STATUS_INTERRUPTED:
            downloader.removeDownload(self.jobId)
        if self.onFinished is not None:
            self.onFinished(self, status, error)
