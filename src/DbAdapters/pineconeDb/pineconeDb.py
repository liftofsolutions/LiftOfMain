"""Every Pinecone call in this process, queued and run over a fixed pool.

This folder is the one way the server reaches Pinecone. `pineconeAdapter`
beside it speaks to the index; nothing outside this folder builds one, and
`getPineconeDb` holds the single adapter the whole server shares.

    from DbAdapters.pineconeDb import getPineconeDb

    written = await getPineconeDb().add(namespace, records)
    records = await getPineconeDb().fetch(namespace, ids)
    await getPineconeDb().update(namespace, recordId, metadata)
    deleted = await getPineconeDb().delete(namespace, ids)
    ids = await getPineconeDb().listIds(namespace)

`getPineconeDb` hands out one shared `PineconeDb`. Each operation is cut
into jobs no bigger than one Pinecone request - `RECORDS_PER_ADD` records,
`IDS_PER_REQUEST` ids, one record per update - and every job goes on one
queue; a listing is one job, however many pages it walks.
`MAX_CONNECTIONS` workers take jobs off it, each over its own connection,
so however many callers there are, no more than that many requests are
ever in flight.

Jobs on the same record run in the order they were made. A job depends on
every earlier unfinished job that touches any of its records, in the same
namespace, so a delete made after an add of the same id never overtakes it,
and a fetch made after a write sees that write. Jobs on different records do
not wait for each other. A job waits for the earlier one to finish, not to
succeed: a delete after a failed add still runs.

A listing touches no record by id, so it waits for nothing, and like any
read it may lag writes made just before it.

A job with nothing to wait for goes straight on the queue; one that depends
on others is held off it, and goes on when the last of them finishes. So a
worker only ever takes a job it can run at once - no connection sits idle
waiting, and no set of jobs can leave every worker stuck.

A job that fails is logged and its error handed back to the caller that
queued it; it does not stop its worker or any other job. An operation cut
into several jobs waits for all of them, then raises the first failure, so
the jobs that could land do. Every write is keyed by record id, so sending
it again after a failure rewrites rather than duplicates.

The index embeds server side, so `add` takes text records (an `_id` plus
the field the index embeds) rather than vectors.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from config import ConfigurationError, getServerConfig
from config.serverConfig import (
    PINECONE_API_KEY_ENV_VAR,
    PINECONE_INDEX_HOST_ENV_VAR,
    PINECONE_INDEX_NAME_ENV_VAR,
    requireSetting,
)
from DbAdapters import VectorRecord
from DbAdapters.pineconeDb.pineconeAdapter import PineconeAdapter

LOGGER = logging.getLogger(__name__)

# Workers on the queue, and connections held open to Pinecone for them.
MAX_CONNECTIONS = 10

# Text records per add request; the records API embeds them, and caps a
# request at 96.
RECORDS_PER_ADD = 96
# Ids per fetch or delete request.
IDS_PER_REQUEST = 1000

OPERATION_ADD = "add"
OPERATION_FETCH = "fetch"
OPERATION_UPDATE = "update"
OPERATION_DELETE = "delete"
OPERATION_LIST = "list"


class PineconeDbError(RuntimeError):
    """A Pinecone job could not be done, or the queue was closed under it."""


# Compared by identity, so a job can sit in a set of the jobs to wait for.
@dataclass(eq=False)
class PineconeJob:
    """One Pinecone request waiting on the queue, and where its answer goes."""

    operation: str
    namespace: str
    payload: Any
    result: asyncio.Future = field(repr=False)
    # The `(namespace, recordId)` pairs this job touches.
    recordKeys: list[tuple[str, str]] = field(default_factory=list)
    # Earlier jobs on the same records still to finish before this one.
    waitingOn: int = 0
    # Later jobs on the same records, waiting for this one to finish.
    dependents: list["PineconeJob"] = field(default_factory=list, repr=False)


def recordIdsOf(operation: str, payload: Any) -> list[str]:
    """The ids of the records a job's payload touches."""
    if operation == OPERATION_LIST:
        return []
    if operation == OPERATION_ADD:
        recordIds = [
            record.get("_id") or record.get("id") for record in payload
        ]
    elif operation == OPERATION_UPDATE:
        recordIds = [recordId for recordId, _ in payload]
    else:
        recordIds = list(payload)
    return [recordId for recordId in recordIds if recordId]


class PineconeDb:
    """A queue of Pinecone jobs, worked by `maxConnections` workers.

    Build one per process and share it - `getPineconeDb` does - since the
    connection cap and the ordering per record only hold if every call goes
    through the same queue. The workers and the connection start on the
    first call.
    """

    def __init__(
        self,
        maxConnections: int = MAX_CONNECTIONS,
        adapter: PineconeAdapter | None = None,
    ) -> None:
        """Record the settings; nothing connects until the first call."""
        self.maxConnections = maxConnections
        self._adapter = adapter
        self._queue: asyncio.Queue[PineconeJob] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._startLock = asyncio.Lock()
        # The latest job made on each record, until it finishes.
        self._lastJobByRecord: dict[tuple[str, str], PineconeJob] = {}
        # Every job made and not yet finished, queued or held back.
        self._unfinishedJobs: set[PineconeJob] = set()

    async def add(self, namespace: str, records: list[dict]) -> int:
        """Write text records into `namespace`; how many were written."""
        results = await self._submitAll(
            OPERATION_ADD, namespace, records, RECORDS_PER_ADD
        )
        return sum(results)

    async def fetch(
        self, namespace: str, ids: list[str]
    ) -> dict[str, VectorRecord]:
        """The records under `ids`, keyed by id; ids not held are absent."""
        results = await self._submitAll(
            OPERATION_FETCH, namespace, ids, IDS_PER_REQUEST
        )
        return {
            recordId: record
            for found in results
            for recordId, record in found.items()
        }

    async def update(
        self, namespace: str, recordId: str, metadata: dict[str, Any]
    ) -> None:
        """Merge `metadata` into one record's metadata."""
        await self._submitAll(
            OPERATION_UPDATE, namespace, [(recordId, metadata)], 1
        )

    async def delete(self, namespace: str, ids: list[str]) -> int:
        """Remove the records under `ids`; how many ids were sent."""
        results = await self._submitAll(
            OPERATION_DELETE, namespace, ids, IDS_PER_REQUEST
        )
        return sum(results)

    async def listIds(self, namespace: str) -> list[str]:
        """Every record id in `namespace`; none for one never written to."""
        [recordIds] = await self._submitAll(
            OPERATION_LIST, namespace, [namespace], 1
        )
        return recordIds

    async def close(self) -> None:
        """Stop the workers, fail what is still queued, close the connection."""
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

        for job in self._unfinishedJobs:
            if not job.result.done():
                job.result.set_exception(
                    PineconeDbError("The Pinecone queue was closed.")
                )
        self._unfinishedJobs.clear()
        self._lastJobByRecord.clear()
        self._queue = asyncio.Queue()

        if self._adapter is not None:
            await self._adapter.close()

    async def _submitAll(
        self, operation: str, namespace: str, items: list, perJob: int
    ) -> list:
        """Queue `items` as jobs of at most `perJob`; every job's result.

        Waits for all of them, then raises the first failure.
        """
        if not items:
            return []
        await self._ensureStarted()
        jobs = [
            self._queueJob(operation, namespace, items[start : start + perJob])
            for start in range(0, len(items), perJob)
        ]
        results = await asyncio.gather(
            *(job.result for job in jobs), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return results

    def _queueJob(
        self, operation: str, namespace: str, payload: Any
    ) -> PineconeJob:
        """Make a job behind the earlier unfinished jobs on its records.

        It goes on the queue now if there are none, or when the last of them
        finishes. No await here, so no other job can be made in between and
        slip in out of order.
        """
        recordKeys = [
            (namespace, recordId)
            for recordId in recordIdsOf(operation, payload)
        ]
        earlierJobs = {
            self._lastJobByRecord[recordKey]
            for recordKey in recordKeys
            if recordKey in self._lastJobByRecord
        }
        job = PineconeJob(
            operation=operation,
            namespace=namespace,
            payload=payload,
            result=asyncio.get_running_loop().create_future(),
            recordKeys=recordKeys,
            waitingOn=len(earlierJobs),
        )
        for earlier in earlierJobs:
            earlier.dependents.append(job)
        for recordKey in recordKeys:
            self._lastJobByRecord[recordKey] = job
        self._unfinishedJobs.add(job)
        if job.waitingOn == 0:
            self._queue.put_nowait(job)
        return job

    async def _ensureStarted(self) -> None:
        """Connect and start the workers, once, on the first call."""
        if self._workers:
            return
        async with self._startLock:
            if self._workers:
                return
            if self._adapter is None:
                self._adapter = buildPineconeAdapter(
                    connectionPoolSize=self.maxConnections
                )
            await self._adapter.connect()
            self._workers = [
                asyncio.create_task(self._work())
                for _ in range(self.maxConnections)
            ]
            LOGGER.info(
                "Pinecone queue started with %s workers.", self.maxConnections
            )

    async def _work(self) -> None:
        """Take jobs off the queue, one at a time, until cancelled."""
        while True:
            job = await self._queue.get()
            try:
                result = await self._run(job)
            except asyncio.CancelledError:
                # Closed mid request: answer the caller rather than leave
                # it waiting on a job no worker will finish.
                if not job.result.done():
                    job.result.set_exception(
                        PineconeDbError("The Pinecone queue was closed.")
                    )
                raise
            except Exception as error:
                LOGGER.exception(
                    "Pinecone %s in %s failed.", job.operation, job.namespace
                )
                if not job.result.done():
                    job.result.set_exception(error)
            else:
                if not job.result.done():
                    job.result.set_result(result)
            finally:
                self._finish(job)
                self._queue.task_done()

    def _finish(self, job: PineconeJob) -> None:
        """Forget `job`, and queue the later jobs left waiting on nothing."""
        self._unfinishedJobs.discard(job)
        for recordKey in job.recordKeys:
            if self._lastJobByRecord.get(recordKey) is job:
                del self._lastJobByRecord[recordKey]
        for dependent in job.dependents:
            dependent.waitingOn -= 1
            if dependent.waitingOn == 0:
                self._queue.put_nowait(dependent)

    async def _run(self, job: PineconeJob) -> Any:
        """Send one job to Pinecone.

        Raises:
            PineconeDbError: If a delete was not applied in full, or the
                operation is not one this queue knows.
        """
        if job.operation == OPERATION_ADD:
            return await self._adapter.upsertTextRecords(
                job.payload, namespace=job.namespace
            )
        if job.operation == OPERATION_FETCH:
            return await self._adapter.fetchVectors(
                job.payload, namespace=job.namespace
            )
        if job.operation == OPERATION_UPDATE:
            [(recordId, metadata)] = job.payload
            await self._adapter.updateVector(
                recordId, metadata=metadata, namespace=job.namespace
            )
            return None
        if job.operation == OPERATION_DELETE:
            outcome = await self._adapter.deleteVectors(
                job.payload, namespace=job.namespace
            )
            if not outcome.succeeded:
                raise PineconeDbError(
                    f"Deleting {outcome.failedIds} from {job.namespace} "
                    f"failed: {outcome.failures[0].message}"
                )
            return outcome.appliedCount
        if job.operation == OPERATION_LIST:
            return await self._adapter.listVectorIds(namespace=job.namespace)
        raise PineconeDbError(f"No Pinecone operation {job.operation!r}.")


def buildPineconeAdapter(**overrides) -> PineconeAdapter:
    """A Pinecone adapter pointed at the configured index.

    Passing the host as well as the name saves a describe round trip on
    connect, so both are handed over when both are known.

    Raises:
        ConfigurationError: If the API key or the index is not configured.
    """
    serverConfig = getServerConfig()
    requireSetting(
        serverConfig.pineconeApiKey, PINECONE_API_KEY_ENV_VAR, "Pinecone"
    )
    if not (serverConfig.pineconeIndexName or serverConfig.pineconeIndexHost):
        raise ConfigurationError(
            f"Pinecone needs ${PINECONE_INDEX_NAME_ENV_VAR} or "
            f"${PINECONE_INDEX_HOST_ENV_VAR}, and neither is set."
        )
    return PineconeAdapter(
        **{
            "apiKey": serverConfig.pineconeApiKey,
            "indexName": serverConfig.pineconeIndexName,
            "indexHost": serverConfig.pineconeIndexHost,
            **overrides,
        }
    )


_PINECONE_DB: PineconeDb | None = None


def getPineconeDb() -> PineconeDb:
    """The process wide Pinecone queue, built once and shared."""
    global _PINECONE_DB
    if _PINECONE_DB is None:
        _PINECONE_DB = PineconeDb()
    return _PINECONE_DB
