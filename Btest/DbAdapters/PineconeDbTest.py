"""Btest for `DbAdapters.pineconeDb`, against the real Pinecone index.

No fakes: every request here really goes to the Pinecone test index from
`test.env`, and the whole module is skipped when Pinecone is not
configured. The tests run in file order and some build on each other:

1. a connection opens and the index answers; the server shares one queue;
2. fifty adds sent at once run ten at a time, and all land;
3. add, update, fetch and delete each work, alone and queued together;
4. big operations are split into requests Pinecone accepts;
5. a mixed batch - adds, deletes of the fifty adds from test 2, a delete of
   a record added in the same batch, and fetches - where the jobs on the
   same records queue behind each other and the rest do not;
6. ordering is per record and namespace, and a waiting job holds no
   connection;
7. a request Pinecone rejects fails alone, and the queue keeps going;
8. rapid commands never exceed the connections: ten by default, four when
   overloaded with twelve;
9. closing the queue answers every caller still waiting.

`WatchedAdapter` wraps the real adapter and only notes when each request
starts and ends, so order and the number in flight are checked on real
calls. Pinecone reads are eventually consistent, so `waitUntil` polls for a
write to show. Everything goes in namespaces drawn for this run, deleted at
the end.

    uv run pytest Btest/DbAdapters/PineconeDbTest.py -q
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

import pytest

from config import getServerConfig
from DbAdapters import VectorRecord
from DbAdapters.pineconeDb import (
    PineconeAdapter,
    PineconeDb,
    PineconeDbError,
    buildPineconeAdapter,
    getPineconeDb,
)

LOGGER = logging.getLogger(__name__)

pytestmark = pytest.mark.skipif(
    not getServerConfig().hasPinecone,
    reason="Pinecone is not configured for this environment.",
)

NAMESPACE = f"btest-{uuid.uuid4().hex}"
OTHER_NAMESPACE = f"{NAMESPACE}-other"

# Sent all at once, five times the default connections.
CONCURRENT_ADD_IDS = [f"concurrent-{index}" for index in range(50)]
SINGLE_ID = "single-1"
CHAIN_ID = "chain-1"
FIRST_IDS = ["batch-first-1", "batch-first-2"]
OTHER_IDS = ["batch-other-1", "batch-other-2", "batch-other-3"]
LOAD_ADD_IDS = [f"load-add-{index}" for index in range(8)]
LOAD_FETCH_IDS = [f"load-fetch-{index}" for index in range(4)]

POLL_SECONDS = 0.5
POLL_TIMEOUT_SECONDS = 60
# A job nobody answers would hang its caller; fail the test instead.
FLOW_TIMEOUT_SECONDS = 180


@pytest.fixture(scope="module", autouse=True)
def cleanNamespaces():
    """Delete everything this run wrote, whatever the tests did."""
    yield

    async def deleteNamespaces() -> None:
        adapter = buildPineconeAdapter()
        try:
            for namespace in (NAMESPACE, OTHER_NAMESPACE):
                try:
                    await adapter.deleteNamespace(namespace=namespace)
                except Exception as error:
                    LOGGER.warning("Deleting %s: %s", namespace, error)
        finally:
            await adapter.close()

    asyncio.run(deleteNamespaces())


@dataclass
class Span:
    """When one request to Pinecone started and ended."""

    operation: str
    namespace: str
    recordIds: list[str]
    start: float
    end: float = 0.0


@dataclass
class WatchedAdapter:
    """The real adapter, noting each connect and each request's span."""

    adapter: PineconeAdapter
    spans: list[Span] = field(default_factory=list)
    connects: int = 0
    inFlight: int = 0
    mostInFlight: int = 0

    async def connect(self) -> None:
        self.connects += 1
        await self.adapter.connect()

    async def close(self) -> None:
        await self.adapter.close()

    async def watch(self, operation, namespace, recordIds, call):
        span = Span(operation, namespace, list(recordIds), time.perf_counter())
        self.spans.append(span)
        self.inFlight += 1
        self.mostInFlight = max(self.mostInFlight, self.inFlight)
        try:
            return await call
        finally:
            self.inFlight -= 1
            span.end = time.perf_counter()

    async def upsertTextRecords(self, records, *, namespace):
        recordIds = [record["_id"] for record in records]
        return await self.watch(
            "add",
            namespace,
            recordIds,
            self.adapter.upsertTextRecords(records, namespace=namespace),
        )

    async def fetchVectors(self, ids, *, namespace):
        return await self.watch(
            "fetch",
            namespace,
            ids,
            self.adapter.fetchVectors(ids, namespace=namespace),
        )

    async def updateVector(self, recordId, *, metadata, namespace):
        return await self.watch(
            "update",
            namespace,
            [recordId],
            self.adapter.updateVector(
                recordId, metadata=metadata, namespace=namespace
            ),
        )

    async def deleteVectors(self, ids, *, namespace):
        return await self.watch(
            "delete",
            namespace,
            ids,
            self.adapter.deleteVectors(ids, namespace=namespace),
        )

    def spanOf(
        self, operation: str, recordId: str, namespace: str = NAMESPACE
    ) -> Span:
        """The first request of `operation` that touched `recordId`."""
        return next(
            span
            for span in self.spans
            if span.operation == operation
            and span.namespace == namespace
            and recordId in span.recordIds
        )

    def sizesOf(self, operation: str) -> list[int]:
        """How many records each request of `operation` carried, sorted."""
        return sorted(
            len(span.recordIds)
            for span in self.spans
            if span.operation == operation
        )


def buildRecord(recordId: str) -> dict:
    """A text record the index embeds, with its own id in the text."""
    return {
        "_id": recordId,
        "chunkText": f"Btest chunk {recordId}.",
        "fileName": "btest.md",
    }


def buildRecords(prefix: str, count: int) -> list[dict]:
    return [buildRecord(f"{prefix}-{index}") for index in range(count)]


def buildBadRecord(recordId: str) -> dict:
    """A record Pinecone rejects: it has no text for the index to embed."""
    return {"_id": recordId, "fileName": "btest.md"}


def runLive(operation, maxConnections: int = 10):
    """Run `operation(database)` on a real queue; its result and the watch."""

    async def flow():
        watched = WatchedAdapter(
            buildPineconeAdapter(connectionPoolSize=maxConnections)
        )
        database = PineconeDb(maxConnections=maxConnections, adapter=watched)
        try:
            result = await asyncio.wait_for(
                operation(database), FLOW_TIMEOUT_SECONDS
            )
            return result, watched
        finally:
            await database.close()

    return asyncio.run(flow())


async def waitUntil(
    database: PineconeDb,
    recordIds: list[str],
    isDone,
    namespace: str = NAMESPACE,
) -> dict[str, VectorRecord]:
    """Fetch `recordIds` until `isDone(found)`, or fail after the timeout."""
    deadline = time.perf_counter() + POLL_TIMEOUT_SECONDS
    while True:
        found = await database.fetch(namespace, recordIds)
        if isDone(found):
            return found
        if time.perf_counter() > deadline:
            raise AssertionError(
                f"{recordIds} never settled; last fetch held {sorted(found)}."
            )
        await asyncio.sleep(POLL_SECONDS)


def allOf(recordIds: list[str]):
    return lambda found: set(recordIds) <= set(found)


def noneOf(recordIds: list[str]):
    return lambda found: not set(recordIds) & set(found)


class TestConnection:
    """1. A real connection opens, once, and the server shares one queue."""

    def testConnectsAndDescribesIndex(self) -> None:
        async def connectAndDescribe() -> tuple[bool, dict]:
            adapter = buildPineconeAdapter(connectionPoolSize=4)
            try:
                await adapter.connect()
                return adapter.isConnected, await adapter.describeStats()
            finally:
                await adapter.close()

        connected, stats = asyncio.run(connectAndDescribe())

        assert connected is True
        assert stats["dimension"] == 1024
        assert stats["metric"] == "cosine"

    def testBuildsAdapterForConfiguredIndex(self) -> None:
        adapter = buildPineconeAdapter(connectionPoolSize=4)
        assert adapter._indexName == getServerConfig().pineconeIndexName
        assert adapter._indexHost == getServerConfig().pineconeIndexHost
        assert adapter._connectionPoolSize == 4

    def testConnectsOnceForManyCalls(self) -> None:
        async def manyCalls(database: PineconeDb) -> None:
            await database.fetch(NAMESPACE, ["missing-1"])
            await database.fetch(NAMESPACE, ["missing-2"])
            await database.delete(NAMESPACE, ["missing-3"])

        _, watched = runLive(manyCalls)
        assert watched.connects == 1

    def testNothingToDoConnectsNothing(self) -> None:
        written, watched = runLive(
            lambda database: database.add(NAMESPACE, [])
        )
        assert written == 0
        assert watched.connects == 0

    def testOneQueueForTheServer(self) -> None:
        assert getPineconeDb() is getPineconeDb()


class TestConcurrentAdds:
    """2. Fifty adds sent at once: ten in flight at a time, all land."""

    def testFiftyAddsShareTenConnections(self) -> None:
        async def manyAdds(database: PineconeDb):
            written = await asyncio.gather(
                *(
                    database.add(NAMESPACE, [buildRecord(recordId)])
                    for recordId in CONCURRENT_ADD_IDS
                )
            )
            found = await waitUntil(
                database, CONCURRENT_ADD_IDS, allOf(CONCURRENT_ADD_IDS)
            )
            return written, found

        (written, found), watched = runLive(manyAdds)

        assert written == [1] * 50
        assert watched.mostInFlight == 10
        assert {
            recordId: record.metadata["chunkText"]
            for recordId, record in found.items()
        } == {
            recordId: f"Btest chunk {recordId}."
            for recordId in CONCURRENT_ADD_IDS
        }


class TestEachOperation:
    """3. Add, update, fetch and delete, one by one and queued together."""

    def testOneByOne(self) -> None:
        async def eachOperation(database: PineconeDb):
            added = await database.add(NAMESPACE, [buildRecord(SINGLE_ID)])
            await waitUntil(database, [SINGLE_ID], allOf([SINGLE_ID]))

            await database.update(
                NAMESPACE, SINGLE_ID, {"fileName": "updated.md"}
            )
            updated = await waitUntil(
                database,
                [SINGLE_ID],
                lambda found: SINGLE_ID in found
                and found[SINGLE_ID].metadata["fileName"] == "updated.md",
            )

            deleted = await database.delete(NAMESPACE, [SINGLE_ID])
            gone = await waitUntil(database, [SINGLE_ID], noneOf([SINGLE_ID]))
            return added, updated, deleted, gone

        (added, updated, deleted, gone), _ = runLive(eachOperation)

        assert added == 1
        assert updated[SINGLE_ID].metadata == {
            "chunkText": f"Btest chunk {SINGLE_ID}.",
            "fileName": "updated.md",
        }
        assert len(updated[SINGLE_ID].values) == 1024
        assert deleted == 1
        assert gone == {}

    def testQueuedTogetherRunInOrder(self) -> None:
        """All four on one record at once: each waits for the one before."""

        async def chain(database: PineconeDb):
            return await asyncio.gather(
                database.add(NAMESPACE, [buildRecord(CHAIN_ID)]),
                database.update(NAMESPACE, CHAIN_ID, {"fileName": "b.md"}),
                database.fetch(NAMESPACE, [CHAIN_ID]),
                database.delete(NAMESPACE, [CHAIN_ID]),
            )

        results, watched = runLive(chain)

        add = watched.spanOf("add", CHAIN_ID)
        update = watched.spanOf("update", CHAIN_ID)
        fetch = watched.spanOf("fetch", CHAIN_ID)
        delete = watched.spanOf("delete", CHAIN_ID)
        assert add.end <= update.start
        assert update.end <= fetch.start
        assert fetch.end <= delete.start
        assert results[0] == 1
        assert results[3] == 1


class TestBatching:
    """4. Big operations go out as requests Pinecone accepts."""

    def testAddSplitsIntoRequestsOf96(self) -> None:
        records = buildRecords("bulk", 250)
        recordIds = [record["_id"] for record in records]

        async def bulkAdd(database: PineconeDb):
            written = await database.add(NAMESPACE, records)
            await waitUntil(database, recordIds, allOf(recordIds))
            return written

        written, watched = runLive(bulkAdd)

        assert written == 250
        assert watched.sizesOf("add") == [58, 96, 96]

    def testFetchSplitsIntoRequestsOf1000(self) -> None:
        recordIds = [f"bulk-{index}" for index in range(2500)]

        found, watched = runLive(
            lambda database: database.fetch(NAMESPACE, recordIds)
        )

        assert watched.sizesOf("fetch") == [500, 1000, 1000]
        assert set(found) == {f"bulk-{index}" for index in range(250)}

    def testDeleteSplitsIntoRequestsOf1000(self) -> None:
        recordIds = [f"bulk-{index}" for index in range(1500)]

        async def bulkDelete(database: PineconeDb):
            deleted = await database.delete(NAMESPACE, recordIds)
            await waitUntil(database, recordIds[:250], noneOf(recordIds))
            return deleted

        deleted, watched = runLive(bulkDelete)

        assert deleted == 1500
        assert watched.sizesOf("delete") == [500, 1000]


class TestDependentBatch:
    """5. One mixed batch: same-record jobs queue, the rest run at once."""

    def testSameRecordJobsQueueOthersDoNot(self) -> None:
        async def mixedBatch(database: PineconeDb):
            results = await asyncio.gather(
                database.add(
                    NAMESPACE, [buildRecord(recordId) for recordId in FIRST_IDS]
                ),
                database.delete(NAMESPACE, CONCURRENT_ADD_IDS),
                database.delete(NAMESPACE, FIRST_IDS),
                database.add(
                    NAMESPACE, [buildRecord(recordId) for recordId in OTHER_IDS]
                ),
                database.fetch(NAMESPACE, OTHER_IDS),
            )
            deletedIds = FIRST_IDS + CONCURRENT_ADD_IDS
            await waitUntil(database, deletedIds, noneOf(deletedIds))
            await waitUntil(database, OTHER_IDS, allOf(OTHER_IDS))
            return results

        results, watched = runLive(mixedBatch)

        firstAdd = watched.spanOf("add", FIRST_IDS[0])
        firstDelete = watched.spanOf("delete", FIRST_IDS[0])
        previousDelete = watched.spanOf("delete", CONCURRENT_ADD_IDS[0])
        otherAdd = watched.spanOf("add", OTHER_IDS[0])
        otherFetch = watched.spanOf("fetch", OTHER_IDS[0])

        # The delete of the first add waited for that add to finish.
        assert firstAdd.end <= firstDelete.start
        # The fetch of the other add waited for that add to finish.
        assert otherAdd.end <= otherFetch.start
        # Jobs on other records did not wait for the first add.
        assert previousDelete.start < firstAdd.end
        assert otherAdd.start < firstAdd.end
        assert results[:4] == [2, 50, 2, 3]


class TestOrdering:
    """6. Ordering is per record and namespace; waiting holds no connection."""

    def testSameIdInOtherNamespaceDoesNotWait(self) -> None:
        recordId = "namespaced-1"

        async def twoNamespaces(database: PineconeDb):
            return await asyncio.gather(
                database.add(NAMESPACE, [buildRecord(recordId)]),
                database.delete(OTHER_NAMESPACE, [recordId]),
            )

        _, watched = runLive(twoNamespaces)

        add = watched.spanOf("add", recordId)
        delete = watched.spanOf("delete", recordId, OTHER_NAMESPACE)
        assert delete.start < add.end

    def testWaitingJobsHoldNoConnection(self) -> None:
        """On one connection, independent work runs while a chain waits."""

        async def chainAndOther(database: PineconeDb):
            return await asyncio.gather(
                database.add(NAMESPACE, [buildRecord("waiting-1")]),
                database.delete(NAMESPACE, ["waiting-1"]),
                database.fetch(NAMESPACE, ["waiting-1"]),
                database.add(NAMESPACE, [buildRecord("independent-1")]),
            )

        _, watched = runLive(chainAndOther, maxConnections=1)

        assert [
            (span.operation, span.recordIds[0]) for span in watched.spans
        ] == [
            ("add", "waiting-1"),
            ("add", "independent-1"),
            ("delete", "waiting-1"),
            ("fetch", "waiting-1"),
        ]


class TestFailures:
    """7. A request Pinecone rejects fails alone; the queue keeps going."""

    def testRejectedRequestFailsAloneOthersLand(self) -> None:
        good = buildRecords("rejected-batch", 95 + 96)
        # The bad record rides in the first request of 96, so only that
        # request is rejected; the second, all good, lands.
        records = [buildBadRecord("rejected-bad")] + good
        landedIds = [record["_id"] for record in good[95:]]
        lostIds = [record["_id"] for record in good[:95]]

        async def addWithBadRecord(database: PineconeDb):
            with pytest.raises(Exception, match="chunkText"):
                await database.add(NAMESPACE, records)
            raisedAt = time.perf_counter()
            await waitUntil(database, landedIds, allOf(landedIds))
            return await database.fetch(NAMESPACE, lostIds), raisedAt

        (lost, raisedAt), watched = runLive(addWithBadRecord)

        addSpans = [span for span in watched.spans if span.operation == "add"]
        assert sorted(len(span.recordIds) for span in addSpans) == [96, 96]
        # The error came back only once both requests had finished.
        assert all(0 < span.end <= raisedAt for span in addSpans)
        assert lost == {}

    def testQueueKeepsGoingAfterAFailure(self) -> None:
        async def failThenAdd(database: PineconeDb) -> int:
            with pytest.raises(Exception, match="chunkText"):
                await database.add(NAMESPACE, [buildBadRecord("after-bad")])
            return await database.add(NAMESPACE, [buildRecord("after-good")])

        written, _ = runLive(failThenAdd, maxConnections=1)
        assert written == 1

    def testLaterJobRunsAfterAFailedOne(self) -> None:
        async def failedAddThenDelete(database: PineconeDb):
            return await asyncio.gather(
                database.add(NAMESPACE, [buildBadRecord("failed-then")]),
                database.delete(NAMESPACE, ["failed-then"]),
                return_exceptions=True,
            )

        results, watched = runLive(failedAddThenDelete)

        assert "chunkText" in str(results[0])
        assert results[1] == 1
        assert watched.spanOf("add", "failed-then").end <= (
            watched.spanOf("delete", "failed-then").start
        )


class TestConnectionCap:
    """8. Rapid commands never exceed the connections."""

    def testDefaultTenAtOnce(self) -> None:
        records = buildRecords("cap", 96 * 12)

        written, watched = runLive(
            lambda database: database.add(NAMESPACE, records)
        )

        assert written == 96 * 12
        assert len(watched.spans) == 12
        assert watched.mostInFlight == 10

    def testTwelveCommandsOnFourConnections(self) -> None:
        async def rapidCommands(database: PineconeDb):
            results = await asyncio.gather(
                *(
                    database.add(NAMESPACE, [buildRecord(recordId)])
                    for recordId in LOAD_ADD_IDS
                ),
                *(
                    database.fetch(NAMESPACE, [recordId])
                    for recordId in LOAD_FETCH_IDS
                ),
            )
            await waitUntil(database, LOAD_ADD_IDS, allOf(LOAD_ADD_IDS))
            return results

        results, watched = runLive(rapidCommands, maxConnections=4)

        commandSpans = watched.spans[:12]
        assert {span.recordIds[0] for span in commandSpans} == set(
            LOAD_ADD_IDS + LOAD_FETCH_IDS
        )
        assert watched.mostInFlight == 4
        # More commands than connections: later ones waited for a slot.
        firstEnd = min(span.end for span in commandSpans)
        assert sum(span.start >= firstEnd for span in commandSpans) >= 8
        assert results == [1] * 8 + [{}] * 4


class TestClose:
    """9. Closing answers every caller, running or still waiting."""

    def testCloseFailsRunningAndWaitingJobs(self) -> None:
        async def closeMidChain(database: PineconeDb):
            calls = asyncio.gather(
                database.add(NAMESPACE, [buildRecord("closing-1")]),
                database.delete(NAMESPACE, ["closing-1"]),
                return_exceptions=True,
            )
            watched = database._adapter
            while watched.inFlight == 0:
                await asyncio.sleep(0.01)
            await database.close()
            return await calls

        results, watched = runLive(closeMidChain)

        assert [type(result) for result in results] == [
            PineconeDbError,
            PineconeDbError,
        ]
        assert [span.operation for span in watched.spans] == ["add"]
