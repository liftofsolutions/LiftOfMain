"""Btest for `DbAdapters.memoryDb`, the in-memory store.

Three collections, one per eviction policy - `lru`, `lfu` and `fifo` -
each capped at `MAX_RECORDS`, held by one store. Every test drives the
store through its own calls and checks it twice: against exact expected
keys, and against `ListCache`, the same cache kept the slow obvious way in
a plain list. After every call, the order the store would evict in must be
identical to the list.

1. Add and fetch: each value comes back as it was stored, whatever it is,
   and the counts and bytes agree.
2. Exhausting a limit: past a collection's record cap, its byte cap, or
   the store's byte cap, the next add evicts exactly the record expected,
   and every other record and count is as it was.
3. Rapid calls: a million requests per policy, each timed - the total,
   and the average per request, per add, per fetch and per eviction,
   printed as a table - and the average does not grow with the records
   held.
4. Who is evicted: the same calls on all three collections evict three
   different, fixed sequences - printed, and named in the eviction log -
   and over a seeded run of a million calls, every eviction is the one
   the list names and the one the log names.

The two million-call runs are the long stress tests; `STRESS_REQUESTS`
sets their length. Deleting a record is not exercised: callers do not
delete.

    uv run pytest Btest/DbAdapters/MemoryDbTest.py -q
"""

import logging
import random
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator

import pytest

from DbAdapters.memoryDb import (
    DEFAULT_MAX_BYTES,
    EVICTION_FIFO,
    EVICTION_LFU,
    EVICTION_LRU,
    EVICTION_POLICIES,
    CollectionConfig,
    CollectionStats,
    MemoryCollection,
    MemoryDb,
    MemoryDbStats,
    measureBytes,
)

STORE_LOGGER = "DbAdapters.memoryDb"

MAX_RECORDS = 5
KEYS = [f"k{index}" for index in range(MAX_RECORDS)]
NEW_KEYS = ["k5", "k6", "k7"]

# Fetched in this order once KEYS are added, so the three policies part
# ways: k1 is the most used but the longest unused, k2 the first of the
# least used, and k0 the first stored.
USAGE = ["k1", "k1", "k1", "k2", "k3", "k4", "k0"]

# What each collection evicts as NEW_KEYS are added after USAGE.
EXPECTED_VICTIMS = {
    EVICTION_LRU: ["k1", "k2", "k3"],
    EVICTION_LFU: ["k2", "k5", "k6"],
    EVICTION_FIFO: ["k0", "k1", "k2"],
}
# What each holds after the first of NEW_KEYS, next to be evicted first.
EXPECTED_ORDER_AFTER_ONE = {
    EVICTION_LRU: ["k2", "k3", "k4", "k0", "k5"],
    EVICTION_LFU: ["k5", "k3", "k4", "k0", "k1"],
    EVICTION_FIFO: ["k1", "k2", "k3", "k4", "k5"],
}
# Which record the store's own cap takes from the largest collection, after
# k0-k3 are added there and k0 is fetched.
STORE_CAP_VICTIMS = {
    EVICTION_LRU: "k1",
    EVICTION_LFU: "k1",
    EVICTION_FIFO: "k0",
}

# Returned by `getRecord` for an absent key, so a stored None is told apart.
MISSING = object()

SEED = 20261012
# Calls in each stress run: the timed rapid calls, and the long run checked
# against the list after every call.
STRESS_REQUESTS = 1_000_000
LONG_RUN_KEYS = 20

RAPID_MAX_RECORDS = 1_000
# Far above what a call takes here, so only a real slowdown fails.
MAX_AVERAGE_MICROSECONDS = 500

SMALL_HELD = 1_000
LARGE_HELD = 100_000
SCALING_REQUESTS = 5_000
SCALING_REPEATS = 3
# A hundred times the records may make a call this many times slower at
# most; O(n) work per call would make it about a hundred times slower.
GROWTH_ALLOWED = 5


class StoredThing:
    """A plain object, to show the store keeps objects as they are."""

    def __init__(self, payload: str) -> None:
        self.payload = payload


class ListCache:
    """The same cache kept the slow, obvious way: in one plain list.

    `order` holds the keys in the order they would be evicted, next first -
    what `getEvictionOrder` must return. Every policy evicts from the front
    of the list, skipping the key just added; they differ only in where a
    key goes when it is added or used:

    * LRU: to the back, on every add and every use.
    * LFU: behind every key used as often or less, so the list runs from
      least to most used, and among equals from first to last to get there.
    * FIFO: to the back when first added, and never moved again.
    """

    def __init__(self, policy: str, maxRecords: int) -> None:
        self.policy = policy
        self.maxRecords = maxRecords
        self.order: list[str] = []
        self.values: dict[str, object] = {}
        self.useCounts: dict[str, int] = {}
        self.hitCount = 0
        self.missCount = 0

    def add(self, key: str, value: object) -> list[str]:
        """Store `value`; the keys evicted to make room, in eviction order."""
        if key in self.values:
            self.use(key)
        else:
            self.useCounts[key] = 1
            self.place(key)
        self.values[key] = value
        evicted = []
        while len(self.order) > self.maxRecords:
            victim = next(other for other in self.order if other != key)
            self.order.remove(victim)
            del self.values[victim]
            del self.useCounts[victim]
            evicted.append(victim)
        return evicted

    def get(self, key: str) -> object:
        """The value under `key`, counted as a use; None when absent."""
        if key not in self.values:
            self.missCount += 1
            return None
        self.hitCount += 1
        self.use(key)
        return self.values[key]

    def use(self, key: str) -> None:
        """Count a use of `key`, and move it where its policy says."""
        self.useCounts[key] += 1
        if self.policy != EVICTION_FIFO:
            self.order.remove(key)
            self.place(key)

    def place(self, key: str) -> None:
        """Put `key` where its policy puts a key just added or used."""
        if self.policy != EVICTION_LFU:
            self.order.append(key)
            return
        count = self.useCounts[key]
        position = next(
            (
                index
                for index, other in enumerate(self.order)
                if self.useCounts[other] > count
            ),
            len(self.order),
        )
        self.order.insert(position, key)


def buildConfig(policy: str, **overrides) -> CollectionConfig:
    """A collection evicting by `policy`, capped at MAX_RECORDS."""
    return CollectionConfig(
        **{"evictionPolicy": policy, "maxRecords": MAX_RECORDS, **overrides}
    )


def buildMemoryDb(
    storeMaxBytes: int = DEFAULT_MAX_BYTES, **configOverrides
) -> MemoryDb:
    """A store with one collection per policy, each at the path of its name.

    `configOverrides` apply to all three collections.
    """
    memoryDb = MemoryDb(maxBytes=storeMaxBytes)
    for policy in EVICTION_POLICIES:
        assert memoryDb.createCollection(
            policy, buildConfig(policy, **configOverrides)
        )
    return memoryDb


def buildValue(key: str) -> dict:
    """A record's value, its own object per call, shaped like a cached job."""
    return {"id": key, "status": "running", "chunks": [key, key, key]}


def recordBytes(key: str, value: object) -> int:
    """The bytes the store counts for `value` stored under `key`."""
    return sys.getsizeof(key) + measureBytes(value)


def addToBoth(
    memoryDb: MemoryDb, listCache: ListCache, key: str, value: object
) -> list[str]:
    """Add to the store and to the list alike; the keys the store evicted.

    Fails unless both evict the same keys and end in the same order.
    """
    path = listCache.policy
    heldBefore = memoryDb.getEvictionOrder(path)
    assert memoryDb.addRecord(path, key, value) is True
    expectedEvicted = listCache.add(key, value)
    heldAfter = memoryDb.getEvictionOrder(path)
    evicted = [held for held in heldBefore if held not in heldAfter]
    assert evicted == expectedEvicted
    assert heldAfter == listCache.order
    return evicted


def fetchFromBoth(memoryDb: MemoryDb, listCache: ListCache, key: str):
    """Fetch from the store and the list alike; the store's answer.

    Fails unless both answer with the same object and end in the same order.
    """
    path = listCache.policy
    value = memoryDb.getRecord(path, key)
    assert value is listCache.get(key)
    assert memoryDb.getEvictionOrder(path) == listCache.order
    return value


def fillAndUse(memoryDb: MemoryDb, policy: str) -> tuple[ListCache, dict]:
    """Add KEYS to `policy`'s collection, then fetch USAGE, both mirrored.

    The list that mirrors it, and the values stored by key.
    """
    listCache = ListCache(policy, MAX_RECORDS)
    values = {key: buildValue(key) for key in KEYS}
    for key in KEYS:
        assert addToBoth(memoryDb, listCache, key, values[key]) == []
    for key in USAGE:
        fetchFromBoth(memoryDb, listCache, key)
    return listCache, values


class EvictionRecorder(logging.Handler):
    """Keeps the eviction records the store logs, and nothing else."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Keep `record` if it logs an eviction."""
        if record.msg.startswith("Evicting"):
            self.records.append(record)

    def victims(self) -> list[tuple[str, str]]:
        """Each eviction kept, as (path, evicted key), in the order made."""
        return [(record.args[1], record.args[0]) for record in self.records]


@pytest.fixture
def evictionLog():
    """The store's evictions, recorded apart from pytest's log capture.

    A million-call run logs hundreds of thousands of evictions. Passed on
    to the root logger, pytest would keep every one in memory, so for the
    test the store's logger keeps them to itself.
    """
    storeLogger = logging.getLogger(STORE_LOGGER)
    recorder = EvictionRecorder()
    previousLevel = storeLogger.level
    previousPropagate = storeLogger.propagate
    storeLogger.addHandler(recorder)
    storeLogger.setLevel(logging.INFO)
    storeLogger.propagate = False
    yield recorder
    storeLogger.propagate = previousPropagate
    storeLogger.setLevel(previousLevel)
    storeLogger.removeHandler(recorder)


class EvictionTimer:
    """Times each eviction, by the path of the collection that made it."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def wrap(self, evictOne: Callable) -> Callable:
        """`evictOne`, timed whenever it evicts."""

        def timedEvictOne(collection, *args, **kwargs):
            started = time.perf_counter()
            evicted = evictOne(collection, *args, **kwargs)
            if evicted:
                self.seconds[collection.path] += time.perf_counter() - started
                self.counts[collection.path] += 1
            return evicted

        return timedEvictOne


@pytest.fixture
def evictionTimer(monkeypatch) -> EvictionTimer:
    """Times every eviction any collection makes during the test."""
    timer = EvictionTimer()
    monkeypatch.setattr(
        MemoryCollection, "evictOne", timer.wrap(MemoryCollection.evictOne)
    )
    return timer


def report(capsys, lines: list[str]) -> None:
    """Print `lines` to the terminal, even while pytest captures output."""
    with capsys.disabled():
        print()
        print("\n".join(lines))


def buildRequests(count: int, keySpace: int) -> Iterator[tuple[str, str]]:
    """`count` adds and fetches, about half each, over `keySpace` keys.

    Made one at a time as they are taken, so a million are never held at
    once. The seed makes every call make the same requests.
    """
    generator = random.Random(SEED)
    for _ in range(count):
        yield (
            generator.choice(("add", "fetch")),
            f"key{generator.randrange(keySpace)}",
        )


def runTimed(
    memoryDb: MemoryDb, path: str, requests: Iterable[tuple[str, str]]
) -> tuple[dict[str, float], dict[str, int]]:
    """Make each request, timing each call alone.

    The seconds spent and the requests made, both by kind.
    """
    seconds = {"add": 0.0, "fetch": 0.0}
    counts = {"add": 0, "fetch": 0}
    for kind, key in requests:
        if kind == "add":
            value = buildValue(key)
            started = time.perf_counter()
            memoryDb.addRecord(path, key, value)
        else:
            started = time.perf_counter()
            memoryDb.getRecord(path, key)
        seconds[kind] += time.perf_counter() - started
        counts[kind] += 1
    return seconds, counts


def bestAverageMicroseconds(policy: str, heldRecords: int) -> float:
    """The fastest average request, in microseconds, over a few runs.

    Made on a collection kept full at `heldRecords` records, so every add
    of a new key evicts one.
    """
    memoryDb = buildMemoryDb(maxRecords=heldRecords)
    for index in range(heldRecords):
        memoryDb.addRecord(policy, f"key{index}", index)
    averages = []
    for _ in range(SCALING_REPEATS):
        requests = buildRequests(SCALING_REQUESTS, keySpace=2 * heldRecords)
        seconds, _ = runTimed(memoryDb, policy, requests)
        averages.append(sum(seconds.values()) / SCALING_REQUESTS * 1e6)
    return min(averages)


@pytest.mark.parametrize("policy", EVICTION_POLICIES)
class TestAddAndFetch:
    """Case 1: what is added comes back, and the counts agree."""

    def testEveryValueFetchesBackAsStored(self, policy):
        memoryDb = buildMemoryDb()
        listCache = ListCache(policy, MAX_RECORDS)
        values = {key: buildValue(key) for key in KEYS}

        for key in KEYS:
            assert addToBoth(memoryDb, listCache, key, values[key]) == []
        for key in KEYS:
            assert fetchFromBoth(memoryDb, listCache, key) is values[key]

        assert memoryDb.getCollectionStats(policy) == CollectionStats(
            path=policy,
            evictionPolicy=policy,
            recordCount=MAX_RECORDS,
            usedBytes=sum(recordBytes(key, values[key]) for key in KEYS),
            maxRecords=MAX_RECORDS,
            maxBytes=None,
            evictedCount=0,
            hitCount=MAX_RECORDS,
            missCount=0,
        )

    def testAnyKindOfValueIsStoredAsItIs(self, policy):
        memoryDb = buildMemoryDb()
        values = {
            "dict": {"nested": {"list": [1, 2, 3]}},
            "list": [b"bytes", 1.5, ("tuple",)],
            "object": StoredThing("payload"),
            "function": buildValue,
            "none": None,
        }

        for key, value in values.items():
            assert memoryDb.addRecord(policy, key, value) is True

        for key, value in values.items():
            assert memoryDb.getRecord(policy, key, MISSING) is value

    def testFetchingAnAbsentKeyIsAMiss(self, policy):
        memoryDb = buildMemoryDb()
        listCache, _ = fillAndUse(memoryDb, policy)

        assert memoryDb.getRecord(policy, "absent", MISSING) is MISSING

        stats = memoryDb.getCollectionStats(policy)
        assert (stats.hitCount, stats.missCount) == (len(USAGE), 1)
        assert memoryDb.getEvictionOrder(policy) == listCache.order

    def testAddingAStoredKeyReplacesItsValue(self, policy):
        memoryDb = buildMemoryDb()
        listCache, values = fillAndUse(memoryDb, policy)
        values["k0"] = buildValue("k0") | {"status": "done"}

        assert addToBoth(memoryDb, listCache, "k0", values["k0"]) == []

        assert fetchFromBoth(memoryDb, listCache, "k0") is values["k0"]
        stats = memoryDb.getCollectionStats(policy)
        assert stats.recordCount == MAX_RECORDS
        assert stats.usedBytes == sum(
            recordBytes(key, values[key]) for key in KEYS
        )

    def testCreateRecordRefusesAStoredKey(self, policy):
        memoryDb = buildMemoryDb()
        first, second = buildValue("k0"), buildValue("k0")

        assert memoryDb.createRecord(policy, "k0", first) is True
        assert memoryDb.createRecord(policy, "k0", second) is False

        assert memoryDb.getRecord(policy, "k0") is first


@pytest.mark.parametrize("policy", EVICTION_POLICIES)
class TestExhaustingTheLimit:
    """Case 2: past a limit, exactly the expected record goes."""

    def testAddPastTheRecordCapEvictsOneAndKeepsTheRest(self, policy):
        memoryDb = buildMemoryDb()
        listCache, values = fillAndUse(memoryDb, policy)
        newKey = NEW_KEYS[0]
        values[newKey] = buildValue(newKey)

        evicted = addToBoth(memoryDb, listCache, newKey, values[newKey])

        victim = EXPECTED_VICTIMS[policy][0]
        held = EXPECTED_ORDER_AFTER_ONE[policy]
        assert evicted == [victim]
        assert memoryDb.getEvictionOrder(policy) == held
        assert memoryDb.getCollectionStats(policy) == CollectionStats(
            path=policy,
            evictionPolicy=policy,
            recordCount=MAX_RECORDS,
            usedBytes=sum(recordBytes(key, values[key]) for key in held),
            maxRecords=MAX_RECORDS,
            maxBytes=None,
            evictedCount=1,
            hitCount=len(USAGE),
            missCount=0,
        )
        assert memoryDb.getRecord(policy, victim, MISSING) is MISSING
        for key in held:
            assert memoryDb.getRecord(policy, key) is values[key]

    def testOtherCollectionsAndTheStoreTotalsAgree(self, policy):
        memoryDb = buildMemoryDb()
        listCaches = {
            each: fillAndUse(memoryDb, each)[0] for each in EVICTION_POLICIES
        }

        addToBoth(
            memoryDb, listCaches[policy], NEW_KEYS[0], buildValue(NEW_KEYS[0])
        )

        for each in EVICTION_POLICIES:
            assert memoryDb.getEvictionOrder(each) == listCaches[each].order
            stats = memoryDb.getCollectionStats(each)
            assert stats.recordCount == MAX_RECORDS
            assert stats.evictedCount == (1 if each == policy else 0)
        collections = [
            memoryDb.getCollectionStats(each) for each in EVICTION_POLICIES
        ]
        assert memoryDb.getStats() == MemoryDbStats(
            usedBytes=sum(stats.usedBytes for stats in collections),
            maxBytes=DEFAULT_MAX_BYTES,
            recordCount=MAX_RECORDS * len(EVICTION_POLICIES),
            collections=collections,
        )

    def testByteCapEvictsUntilTheRecordsFit(self, policy):
        keys = [f"k{index}" for index in range(8)]
        size = recordBytes("k0", buildValue("k0"))
        # Every record weighs the same, so a cap of three records' bytes
        # evicts exactly as a cap of three records would.
        assert {recordBytes(key, buildValue(key)) for key in keys} == {size}
        memoryDb = buildMemoryDb(maxRecords=None, maxBytes=3 * size)
        listCache = ListCache(policy, maxRecords=3)

        for key in keys[:5]:
            addToBoth(memoryDb, listCache, key, buildValue(key))
        fetchFromBoth(memoryDb, listCache, "k3")
        for key in keys[5:]:
            addToBoth(memoryDb, listCache, key, buildValue(key))

        stats = memoryDb.getCollectionStats(policy)
        assert stats.usedBytes == 3 * size
        assert stats.evictedCount == len(keys) - 3

    def testRecordOverTheByteCapIsStillAddedAndKeptAlone(self, policy):
        size = recordBytes("k0", buildValue("k0"))
        memoryDb = buildMemoryDb(maxRecords=None, maxBytes=3 * size)
        for key in KEYS[:3]:
            memoryDb.addRecord(policy, key, buildValue(key))
        oversized = "x" * (4 * size)

        assert memoryDb.addRecord(policy, "big", oversized) is True

        assert memoryDb.getEvictionOrder(policy) == ["big"]
        assert memoryDb.getRecord(policy, "big") is oversized
        stats = memoryDb.getCollectionStats(policy)
        assert stats.evictedCount == 3
        assert stats.usedBytes == recordBytes("big", oversized) > 3 * size

    def testStoreCapEvictsFromTheLargestCollection(self, policy):
        size = recordBytes("k0", buildValue("k0"))
        assert recordBytes("s0", buildValue("s0")) == size
        memoryDb = buildMemoryDb(storeMaxBytes=5 * size)
        listCache = ListCache(policy, MAX_RECORDS)
        for key in KEYS[:4]:
            addToBoth(memoryDb, listCache, key, buildValue(key))
        fetchFromBoth(memoryDb, listCache, "k0")

        # Five records: exactly at the cap, so nothing goes yet.
        memoryDb.addRecord("small", "s0", buildValue("s0"))
        assert memoryDb.getEvictionOrder(policy) == listCache.order
        memoryDb.addRecord("small", "s1", buildValue("s1"))

        assert listCache.order[0] == STORE_CAP_VICTIMS[policy]
        assert memoryDb.getEvictionOrder(policy) == listCache.order[1:]
        assert memoryDb.getEvictionOrder("small") == ["s0", "s1"]
        assert memoryDb.getCollectionStats(policy).evictedCount == 1
        assert memoryDb.getStats().usedBytes == 5 * size


class TestRapidCalls:
    """Case 3: how long a request takes, for each policy."""

    def testRequestTimesForEveryPolicy(self, capsys, evictionTimer):
        rows = []
        averages = {}
        evictionAverages = {}

        for policy in EVICTION_POLICIES:
            memoryDb = buildMemoryDb(maxRecords=RAPID_MAX_RECORDS)
            requests = buildRequests(
                STRESS_REQUESTS, keySpace=2 * RAPID_MAX_RECORDS
            )
            seconds, counts = runTimed(memoryDb, policy, requests)
            totalSeconds = sum(seconds.values())
            averages[policy] = totalSeconds / STRESS_REQUESTS * 1e6
            evictionCount = evictionTimer.counts[policy]
            assert evictionCount == (
                memoryDb.getCollectionStats(policy).evictedCount
            )
            evictionAverages[policy] = (
                evictionTimer.seconds[policy] / evictionCount * 1e6
            )
            rows.append(
                f"{policy:<8}{totalSeconds * 1e3:>10.1f}"
                f"{averages[policy]:>12.2f}"
                f"{seconds['add'] / counts['add'] * 1e6:>10.2f}"
                f"{seconds['fetch'] / counts['fetch'] * 1e6:>10.2f}"
                f"{evictionCount:>11,}"
                f"{evictionAverages[policy]:>13.2f}"
            )

        # The seed makes the same requests, so the same counts, per policy.
        report(
            capsys,
            [
                f"Rapid calls: {STRESS_REQUESTS:,} requests per policy "
                f"({counts['add']:,} adds, {counts['fetch']:,} fetches), "
                f"{RAPID_MAX_RECORDS:,} records kept; averages in us",
                f"{'policy':<8}{'total ms':>10}{'us/request':>12}"
                f"{'us/add':>10}{'us/fetch':>10}{'evictions':>11}"
                f"{'us/eviction':>13}",
                *rows,
            ],
        )
        for policy in EVICTION_POLICIES:
            assert averages[policy] < MAX_AVERAGE_MICROSECONDS, policy
            assert evictionAverages[policy] < MAX_AVERAGE_MICROSECONDS, policy

    @pytest.mark.parametrize("policy", EVICTION_POLICIES)
    def testRequestTimeDoesNotGrowWithTheRecordsHeld(self, policy, capsys):
        small = bestAverageMicroseconds(policy, SMALL_HELD)
        large = bestAverageMicroseconds(policy, LARGE_HELD)

        report(
            capsys,
            [
                f"{policy}: {small:.2f} us/request holding {SMALL_HELD:,}"
                f" records, {large:.2f} holding {LARGE_HELD:,}"
            ],
        )
        assert large < small * GROWTH_ALLOWED


class TestWhoIsEvicted:
    """Case 4: which record each policy gives up, and in what order."""

    def testSameCallsEvictAFixedSequencePerPolicy(self, capsys, evictionLog):
        memoryDb = buildMemoryDb()
        evictedBy = {}

        for policy in EVICTION_POLICIES:
            listCache, _ = fillAndUse(memoryDb, policy)
            evictedBy[policy] = [
                victim
                for key in NEW_KEYS
                for victim in addToBoth(
                    memoryDb, listCache, key, buildValue(key)
                )
            ]

        report(
            capsys,
            [f"Evicted as {', '.join(NEW_KEYS)} were added after the same use:"]
            + [
                f"  {policy:<4}: {' '.join(victims)}"
                for policy, victims in evictedBy.items()
            ],
        )
        assert evictedBy == EXPECTED_VICTIMS
        assert evictionLog.victims() == [
            (policy, victim)
            for policy in EVICTION_POLICIES
            for victim in EXPECTED_VICTIMS[policy]
        ]
        assert evictionLog.records[0].getMessage() == (
            "Evicting k1 from lru by lru: collection over its limit."
        )

    @pytest.mark.parametrize("policy", EVICTION_POLICIES)
    def testEveryEvictionInALongRunIsTheOneTheListNames(
        self, policy, evictionLog, capsys
    ):
        memoryDb = buildMemoryDb()
        listCache = ListCache(policy, MAX_RECORDS)
        generator = random.Random(SEED)
        evictedCount = 0
        started = time.perf_counter()

        for _ in range(STRESS_REQUESTS):
            key = f"k{generator.randrange(LONG_RUN_KEYS)}"
            if generator.random() < 0.5:
                evicted = addToBoth(memoryDb, listCache, key, buildValue(key))
                # Checked and let go call by call, so the log of a million
                # calls is never held at once.
                assert evictionLog.victims() == [
                    (policy, victim) for victim in evicted
                ]
                evictionLog.records.clear()
                evictedCount += len(evicted)
            else:
                fetchFromBoth(memoryDb, listCache, key)

        seconds = time.perf_counter() - started
        stats = memoryDb.getCollectionStats(policy)
        report(
            capsys,
            [
                f"{policy}: {STRESS_REQUESTS:,} calls, each checked against "
                f"the list, in {seconds:.1f} s - {evictedCount:,} evicted, "
                f"{stats.hitCount:,} hits, {stats.missCount:,} misses"
            ],
        )
        assert evictedCount > 0
        assert stats.evictedCount == evictedCount
        assert (stats.hitCount, stats.missCount) == (
            listCache.hitCount,
            listCache.missCount,
        )
