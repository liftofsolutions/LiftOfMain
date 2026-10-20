"""A Redis-like store in this process's own memory, shared by the server.

    from DbAdapters.memoryDb import CollectionConfig, EVICTION_LFU
    from DbAdapters.memoryDb import getMemoryDb

    memoryDb = getMemoryDb()
    memoryDb.createCollection(
        "ragJobs", CollectionConfig(evictionPolicy=EVICTION_LFU, maxRecords=500)
    )
    memoryDb.addRecord("ragJobs", jobId, job)
    job = memoryDb.getRecord("ragJobs", jobId)
    memoryDb.deleteRecord("ragJobs", jobId)

Three levels. `MemoryDb` is the store: one per process, built by
`getMemoryDb` when the app starts, before it takes a request. It holds
collections by path - a path is just the collection's name, such as
`"ragJobs"` or `"chunks/report"`. Each `MemoryCollection` holds its records
and its own `CollectionConfig`: which eviction policy picks its victims,
and how many records or bytes it may keep. A write to a path with no
collection makes one with the default config.

Values are kept as they are, not copied or serialised, so anything can be
stored - a dict, a model, a client. The flip side: a caller that changes a
stored value in place changes the stored record, and the bytes counted for
it stay what it held when written. Write it again to recount.

A write is never refused for space. The record goes in, then the store
trims back under its limits, never evicting the record just written:

1. The collection evicts by its own policy until it is within its own
   `maxRecords` and `maxBytes`.
2. If the whole store is then over its `maxBytes`, it evicts from the
   collection holding the most bytes, by that collection's policy, one
   record at a time, until it fits.

A record that alone is bigger than a limit is still kept; everything it
competes with is evicted, and a warning is logged.

Nothing here raises. The store is a cache, so a failure inside it - a
key that cannot be hashed, a value that will not be measured, a bug - is
logged and answered with a safe value (False, None, or the caller's
default), never raised into a request or the server.

Cost: reads, deletes and stats for one collection are O(1); a write is
O(log n) in the number of collections, for keeping the largest one to
hand, plus the walk that measures the value (`measureBytes`), which is
O(size of the value) since nothing shorter can say what a value holds.
Each eviction costs the same O(log n). Listing every collection's stats is
O(n), the size of the answer. The byte count is an estimate of what the
records hold, not of the process's memory.

Every call takes one lock, so the store is safe from the event loop and
from the threads FastAPI runs sync handlers on alike. Nothing waits while
holding it.
"""

import functools
import heapq
import itertools
import logging
import sys
import threading
from collections import deque
from collections.abc import Callable
from types import FunctionType, MethodType, ModuleType
from typing import Any

from DbAdapters.memoryDb.memoryCollection import MemoryCollection
from DbAdapters.memoryDb.model import (
    CollectionConfig,
    CollectionStats,
    MemoryDbStats,
    isLimit,
)

LOGGER = logging.getLogger(__name__)

# What the whole store may hold before it starts evicting.
DEFAULT_MAX_BYTES = 256 * 1024 * 1024

# Stale entries the largest-first heap may carry past two per collection
# before it is rebuilt from the live collections.
HEAP_SLACK = 64

# Counted as they are, without looking inside: they hold nothing else, or
# what they hold is code rather than data the store owns.
LEAF_TYPES = (
    str,
    bytes,
    bytearray,
    int,
    float,
    complex,
    type(None),
    type,
    ModuleType,
    FunctionType,
    MethodType,
)
CONTAINER_TYPES = (list, tuple, set, frozenset, deque)


def innerValuesOf(value: Any) -> list:
    """What `value` holds: a dict's keys and values, a container's items,
    or a plain object's `__dict__` and slot values."""
    if isinstance(value, dict):
        return [*value.keys(), *value.values()]
    if isinstance(value, CONTAINER_TYPES):
        return list(value)
    innerValues = []
    if hasattr(value, "__dict__"):
        innerValues.append(vars(value))
    for ownerClass in type(value).__mro__:
        slotNames = getattr(ownerClass, "__slots__", ())
        if isinstance(slotNames, str):
            slotNames = (slotNames,)
        for slotName in slotNames:
            if hasattr(value, slotName):
                innerValues.append(getattr(value, slotName))
    return innerValues


def measureBytes(value: Any) -> int:
    """Roughly how many bytes `value` holds, counting everything inside it.

    `sys.getsizeof` alone counts only the outer object - a dict of a
    thousand strings weighs as the dict - so containers and object
    attributes are walked. Each object counts once however often it is
    referenced, so shared and cyclic values do not inflate the total. An
    object that will not give its size or its contents counts for what it
    did give, and the rest of the value is still measured.
    """
    seenIds: set[int] = set()
    pending = [value]
    totalBytes = 0
    while pending:
        current = pending.pop()
        if id(current) in seenIds:
            continue
        seenIds.add(id(current))
        try:
            totalBytes += sys.getsizeof(current)
            if not isinstance(current, LEAF_TYPES):
                pending.extend(innerValuesOf(current))
        except Exception:
            continue
    return totalBytes


def failSafe(fallback: Any) -> Callable:
    """Make a store call log any error and return `fallback` instead."""

    def decorate(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            try:
                return method(self, *args, **kwargs)
            except Exception:
                LOGGER.exception(
                    "MemoryDb.%s failed; answering %r.",
                    method.__name__,
                    fallback,
                )
                return fallback

        return wrapper

    return decorate


class MemoryDb:
    """Collections of records by path, under one memory limit.

    Build one per process and share it - `getMemoryDb` does - since the
    memory limit only holds if every record goes through the same store.
    """

    def __init__(self, maxBytes: int = DEFAULT_MAX_BYTES) -> None:
        """An empty store that evicts once it holds over `maxBytes`."""
        if maxBytes is None or not isLimit(maxBytes):
            LOGGER.warning(
                "Memory limit %r is unusable; using %s bytes.",
                maxBytes,
                DEFAULT_MAX_BYTES,
            )
            maxBytes = DEFAULT_MAX_BYTES
        self.maxBytes = maxBytes
        self.usedBytes = 0
        self._collections: dict[str, MemoryCollection] = {}
        self._lock = threading.Lock()
        # Collections by bytes held, largest first, as
        # (-usedBytes, pushOrder, collection). An entry goes stale when its
        # collection's bytes change or the collection is deleted; stale
        # entries are dropped as they surface. `pushOrder` breaks ties, so
        # two collections are never compared.
        self._largestFirst: list[tuple[int, int, MemoryCollection]] = []
        self._pushOrder = itertools.count()

    @failSafe(False)
    def createCollection(
        self, path: str, config: CollectionConfig | None = None
    ) -> bool:
        """Make an empty collection at `path` with its own policy and limits.

        False, and nothing made, if `path` already has a collection or the
        config cannot work.
        """
        config = config or CollectionConfig()
        problem = config.describeProblem()
        if problem is not None:
            LOGGER.error("Collection %r not made: %s.", path, problem)
            return False
        with self._lock:
            if path in self._collections:
                return False
            self._collections[path] = MemoryCollection(path, config)
            return True

    @failSafe(False)
    def deleteCollection(self, path: str) -> bool:
        """Drop the collection at `path` and every record in it.

        Whether there was one to drop.
        """
        with self._lock:
            collection = self._collections.pop(path, None)
            if collection is None:
                return False
            self.usedBytes -= collection.usedBytes
            return True

    @failSafe(False)
    def createRecord(self, path: str, key: str, value: Any) -> bool:
        """Store a new record under `key`. Never refused for space.

        False, and nothing stored, if `key` is already stored at `path`.
        """
        return self._write(path, key, value, mustBeNew=True)

    @failSafe(False)
    def addRecord(self, path: str, key: str, value: Any) -> bool:
        """Store `value` under `key`, replacing any record already there.

        Never refused for space, so False only if the store failed.
        """
        return self._write(path, key, value, mustBeNew=False)

    def getRecord(self, path: str, key: str, default: Any = None) -> Any:
        """The value stored under `key` at `path`, or `default` if none.

        Counted as a hit or a miss on the collection, and as a use by its
        eviction policy. A path with no collection has nothing to count on.
        """
        try:
            with self._lock:
                collection = self._collections.get(path)
                if collection is None:
                    LOGGER.debug("Miss on %s: no collection %s.", key, path)
                    return default
                entry = collection.get(key)
        except Exception:
            LOGGER.exception("MemoryDb.getRecord failed; answering default.")
            return default
        return default if entry is None else entry.value

    @failSafe(False)
    def deleteRecord(self, path: str, key: str) -> bool:
        """Drop the record under `key` at `path`; whether there was one."""
        with self._lock:
            collection = self._collections.get(path)
            entry = None if collection is None else collection.remove(key)
            if entry is None:
                return False
            self.usedBytes -= entry.sizeBytes
            self._noteSize(collection)
            return True

    @failSafe(None)
    def getEvictionOrder(self, path: str) -> list[str] | None:
        """The keys at `path`, in the order they would be evicted.

        The next to go comes first. Not counted as a read, so it changes
        nothing. O(n), the size of the answer. None if `path` has no
        collection.
        """
        with self._lock:
            collection = self._collections.get(path)
            return None if collection is None else collection.evictionOrder()

    @failSafe(None)
    def getCollectionStats(self, path: str) -> CollectionStats | None:
        """One collection's size, limits, evictions, hits and misses.

        None if `path` has no collection.
        """
        with self._lock:
            collection = self._collections.get(path)
            return None if collection is None else collection.stats()

    @failSafe(None)
    def getStats(self) -> MemoryDbStats | None:
        """Memory used by the store, and every collection's stats."""
        with self._lock:
            collections = [
                collection.stats()
                for collection in self._collections.values()
            ]
            usedBytes = self.usedBytes
        return MemoryDbStats(
            usedBytes=usedBytes,
            maxBytes=self.maxBytes,
            recordCount=sum(
                collection.recordCount for collection in collections
            ),
            collections=collections,
        )

    def _write(self, path: str, key: str, value: Any, mustBeNew: bool) -> bool:
        """Store `value`, then trim the collection and the store to fit.

        Measured before the lock is taken, so a large value being walked
        does not hold up every other call.
        """
        # An unusable key fails here, before anything has changed.
        hash(key)
        sizeBytes = sys.getsizeof(key) + measureBytes(value)
        with self._lock:
            collection = self._collections.get(path)
            if collection is None:
                collection = MemoryCollection(path, CollectionConfig())
                self._collections[path] = collection
            if mustBeNew and collection.has(key):
                return False
            usedBefore = collection.usedBytes
            collection.put(key, value, sizeBytes)
            self.usedBytes += collection.usedBytes - usedBefore
            self._noteSize(collection)
            self._trimToLimit(collection, key)
            return True

    def _trimToLimit(
        self, writtenTo: MemoryCollection, writtenKey: str
    ) -> None:
        """Evict from the largest collections until the store fits.

        Never evicts `writtenKey` from `writtenTo`, the record just written.
        """
        while self.usedBytes > self.maxBytes:
            collection = self._largestEvictable(writtenTo)
            if collection is None:
                LOGGER.warning(
                    "Record %s alone is over the store's limit; kept anyway.",
                    writtenKey,
                )
                return
            protectedKey = writtenKey if collection is writtenTo else None
            usedBefore = collection.usedBytes
            if not collection.evictOne(protectedKey, "store over its limit"):
                # Cannot happen for an evictable collection; stop rather
                # than loop forever if it ever does.
                return
            self.usedBytes += collection.usedBytes - usedBefore
            self._noteSize(collection)

    def _noteSize(self, collection: MemoryCollection) -> None:
        """Put `collection`'s current bytes on the largest-first heap.

        O(log n). Older entries for it are left to go stale; once stale
        entries outnumber live ones the heap is rebuilt, which keeps it
        O(log n) per call amortised.
        """
        heapq.heappush(self._largestFirst, self._heapEntry(collection))
        liveCount = len(self._collections)
        if len(self._largestFirst) > 2 * liveCount + HEAP_SLACK:
            self._largestFirst = [
                self._heapEntry(liveCollection)
                for liveCollection in self._collections.values()
            ]
            heapq.heapify(self._largestFirst)

    def _heapEntry(
        self, collection: MemoryCollection
    ) -> tuple[int, int, MemoryCollection]:
        """`collection`'s entry for the largest-first heap, as it is now."""
        return (-collection.usedBytes, next(self._pushOrder), collection)

    def _largestEvictable(
        self, writtenTo: MemoryCollection
    ) -> MemoryCollection | None:
        """The collection holding the most bytes with a record to give up.

        `writtenTo` counts only if it holds more than the record just
        written. Stale and empty entries met on the way are dropped.
        """
        setAside = None
        largest = None
        while self._largestFirst:
            negativeBytes, _, collection = self._largestFirst[0]
            isCurrent = (
                self._collections.get(collection.path) is collection
                and collection.usedBytes == -negativeBytes
            )
            holdsOnlyWritten = (
                collection is writtenTo and collection.recordCount == 1
            )
            if isCurrent and holdsOnlyWritten:
                # Still a candidate for the next write: put back below.
                setAside = heapq.heappop(self._largestFirst)
                continue
            if isCurrent and collection.recordCount > 0:
                largest = collection
                break
            heapq.heappop(self._largestFirst)
        if setAside is not None:
            heapq.heappush(self._largestFirst, setAside)
        return largest


_MEMORY_DB: MemoryDb | None = None


def getMemoryDb() -> MemoryDb:
    """The process wide store, built once and shared.

    `main.py` calls this as the app starts, so the store exists before the
    first request; every later call returns that same one.
    """
    global _MEMORY_DB
    if _MEMORY_DB is None:
        _MEMORY_DB = MemoryDb()
    return _MEMORY_DB
