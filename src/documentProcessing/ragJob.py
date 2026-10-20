"""One rag job: what it was submitted with, and where it has got to.

`submitRagJob` makes a `RagJob` and hands it to the `JobManager`, which
gives it `onFinished` and, when a slot is free, runs `start` in a task.
`start` is the whole job: it marks the job running, downloads the document
from the link it was submitted with, has `documentAnalyzer` work out what
the document holds, file by file, then takes the strategy `ragStrategy`
picks by its token count. Plain text is saved on the rag's record. A rag's
text is cut into `chunks`, which are written to the rag's namespace in
Pinecone - its index embeds them - and the rag's record gets their count.

Every change of status goes through `updateRagJobStatus`, which writes it
to the job's record in Firestore; nothing is cached.

* running: `started_at`, one more `attempts`, and `expires_at`
  RUNNING_JOB_RETENTION away, so the TTL cannot delete it part way; the
  `error` and `finished_at` of an earlier run that failed are cleared.
* done or failed - the end: `finished_at`, `expires_at` moved to
  FINISHED_JOB_RETENTION after it and, for a failure, the error. The rag
  is made ready, or failed, in the same batch, so the two always agree.
  Then the download is removed and `onFinished` is called with the status
  and any error; the manager writes both onto the job, frees its slot and
  starts the next job waiting.

An error anywhere - a link to a format not taken, a download that fails, a
document nothing can be read from, Firestore refusing a write - fails the
job, so `start` never raises and every run ends in exactly one call back.
An interrupted job is called back for by the manager instead, and records
nothing more: it is being deleted. Its download is left, as its analysis
thread may still be reading it.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from google.cloud.firestore import DELETE_FIELD, Increment

from DbAdapters import Document, mergeOperation
from documentProcessing import (
    documentAnalyzer,
    downloader,
    ragStrategy,
    utilities,
)
from documentProcessing.documentAnalyzer import DocumentAnalysis
from documentProcessing.ragStrategy import Chunk

LOGGER = logging.getLogger(__name__)

# Waiting in the manager's queue for a slot.
JOB_STATUS_SUBMITTED = "submitted"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_DONE = "done"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_INTERRUPTED = "interrupted"

# How long a job's record is kept before the TTL deletes it: while it
# runs, far longer than a run takes; once it has finished, a few days.
RUNNING_JOB_RETENTION = timedelta(days=30)
FINISHED_JOB_RETENTION = timedelta(days=7)


def utcNow() -> datetime:
    """Current UTC time, kept behind a function so tests can freeze it."""
    return datetime.now(timezone.utc)


def errorTextOf(error: BaseException) -> str:
    """What an error says, or its type's name when it says nothing."""
    return str(error) or type(error).__name__


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
        """Do the job: download, analyze, take the strategy, then end it.

        Parsing and chunking run in a thread, since they block. Any error,
        recording `done` included, ends the job failed instead; nothing is
        raised.
        """
        try:
            await self.updateRagJobStatus(JOB_STATUS_RUNNING)
            self.downloadPath = await downloader.downloadDocument(
                self.documentLink, self.jobId
            )
            LOGGER.info(
                "Job %s downloaded %s.", self.jobId, self.downloadPath
            )

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
                saved = await ragStrategy.saveChunks(
                    self.ragId, self.projectId, self.chunks
                )
                await ragStrategy.saveIndexedRag(
                    self.ragId, self.documentLink, self.analysis, saved
                )
            LOGGER.info(
                "Job %s took the %s strategy: %s chunks.",
                self.jobId,
                self.strategy,
                len(self.chunks),
            )
            await self.updateRagJobStatus(JOB_STATUS_DONE)
        except Exception as error:
            await self.updateRagJobStatus(JOB_STATUS_FAILED, error)

    async def updateRagJobStatus(
        self, status: str, error: Exception | None = None
    ) -> None:
        """Move the job to `status`, on the job and in Firestore.

        Running and done raise what Firestore raises, for `start` to fail
        the job. At the end - done, or failed with `error` - the rag is
        made ready or failed with the job, the download is removed and the
        manager called back. A failure that cannot be recorded is logged
        rather than raised, so the manager is still called back, with the
        error that failed the job.
        """
        self.status = status
        if status == JOB_STATUS_RUNNING:
            self.startedAt = utcNow()
            LOGGER.info("Job started: %s", self)
            await self.saveRecord(
                {
                    "status": JOB_STATUS_RUNNING,
                    "started_at": self.startedAt,
                    "attempts": Increment(1),
                    "expires_at": self.startedAt + RUNNING_JOB_RETENTION,
                    "error": DELETE_FIELD,
                    "finished_at": DELETE_FIELD,
                }
            )
            return

        self.finishedAt = utcNow()
        if status == JOB_STATUS_DONE:
            await self.saveEnd(ragStrategy.RAG_STATUS_READY)
        else:
            self.error = errorTextOf(error)
            LOGGER.error("Job %s failed.", self.jobId, exc_info=error)
            try:
                await self.saveEnd(ragStrategy.RAG_STATUS_FAILED)
            except Exception:
                LOGGER.exception(
                    "Could not record job %s as failed.", self.jobId
                )
        # Every read of the folder is over: threads are awaited, not left.
        downloader.removeDownload(self.jobId)
        if self.onFinished is not None:
            self.onFinished(self, self.status, self.error)

    async def saveRecord(self, fields: dict) -> None:
        """Merge `fields` into the job's record in Firestore."""
        await utilities.getFirestoreAdapter().upsertDocument(
            utilities.RAG_JOBS,
            Document(self.jobId, {**fields, "updated_at": utcNow()}),
            merge=True,
        )

    async def saveEnd(self, ragStatus: str) -> None:
        """Record how the job ended, and its rag's `ragStatus`, in one batch.

        The error is written only for a failure: the schema types it as a
        string, never null.
        """
        jobFields = {
            "status": self.status,
            "finished_at": self.finishedAt,
            "expires_at": self.finishedAt + FINISHED_JOB_RETENTION,
            "updated_at": self.finishedAt,
        }
        if self.error is not None:
            jobFields["error"] = self.error
        await utilities.getFirestoreAdapter().applyAtomically(
            [
                mergeOperation(utilities.RAG_JOBS, self.jobId, jobFields),
                mergeOperation(
                    utilities.RAGS,
                    self.ragId,
                    {"status": ragStatus, "updated_at": self.finishedAt},
                ),
            ]
        )

    def interrupt(self) -> None:
        """Cancel the job's task, if it has one still running."""
        if self.task is not None and not self.task.done():
            self.task.cancel()
