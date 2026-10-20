"""One collection of the in-memory store, and the policies that trim it.

A collection is one path's records: an `OrderedDict` from record key to
`CacheEntry`, the bytes those entries hold, how many reads hit and missed,
and the eviction policy that picks which record goes when the collection
is over its limits.

Every operation here is O(1). Record keys are unique, so the dict is the
only index a collection needs - every read, write and delete is one hash
lookup. Its order doubles as the policy's bookkeeping for LRU and FIFO: a
use moves the key to the end, so the least recently used sits first, and
FIFO never moves it. LFU keeps its keys in buckets by use count, the
buckets linked lowest count first, so a use moves a key one bucket up and
the victim is always in the first bucket. No policy scans to pick a victim:
the record just written is the only one ever skipped, so at most two keys
are looked at.
"""

import logging
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

from DbAdapters.memoryDb.model import (
    EVICTION_FIFO,
    EVICTION_LFU,
    EVICTION_LRU,
    CacheEntry,
    CollectionConfig,
    CollectionStats,
)

LOGGER = logging.getLogger(__name__)


def firstKeyExcept(
    keys: Iterable[str], protectedKey: str | None
) -> str | None:
    """The first of `keys` that is not `protectedKey`, or None.

    Looks at two keys at most, since only one key is ever protected.
    """
    for key in keys:
        if key != protectedKey:
            return key
    return None


class EvictionPolicy:
    """Which record a collection gives up next.

    Told of every record added, used and removed, so `chooseVictim` can
    answer without scanning. `protectedKey` is never chosen: it is the
    record just written, which a write always keeps.
    """

    def __init__(self, records: OrderedDict[str, CacheEntry]) -> None:
        """Share the collection's own records, whose order it may use."""
        self.records = records

    def onAdd(self, key: str) -> None:
        """A new record was stored under `key`."""

    def onUse(self, key: str) -> None:
        """The record under `key` was read or rewritten."""

    def onRemove(self, key: str) -> None:
        """The record under `key` is gone."""

    def chooseVictim(self, protectedKey: str | None) -> str | None:
        """The key to evict next, or None if only `protectedKey` is left."""
        raise NotImplementedError

    def evictionOrder(self) -> list[str]:
        """Every key, in the order they would be evicted, next first.

        O(n), the size of the answer. The records' own order, for the
        policies that keep their order in it.
        """
        return list(self.records)


class LruPolicy(EvictionPolicy):
    """Evict the least recently used record."""

    def onUse(self, key: str) -> None:
        """Move `key` to the most recent end."""
        self.records.move_to_end(key)

    def chooseVictim(self, protectedKey: str | None) -> str | None:
        """The key at the least recent end."""
        return firstKeyExcept(self.records, protectedKey)


class FifoPolicy(EvictionPolicy):
    """Evict the record stored first; using or rewriting it changes nothing."""

    def chooseVictim(self, protectedKey: str | None) -> str | None:
        """The key stored longest ago."""
        return firstKeyExcept(self.records, protectedKey)


class CountBucket:
    """The keys used exactly `count` times, in the order they got there.

    Linked to the buckets of the next lower and higher counts in use, so
    LFU never has to search for the lowest count or the next one up.
    """

    __slots__ = ("count", "keys", "lower", "higher")

    def __init__(
        self,
        count: int,
        lower: "CountBucket | None",
        higher: "CountBucket | None",
    ) -> None:
        """An empty bucket for `count`, between `lower` and `higher`."""
        self.count = count
        self.keys: OrderedDict[str, None] = OrderedDict()
        self.lower = lower
        self.higher = higher


class LfuPolicy(EvictionPolicy):
    """Evict the least often used record.

    Among records used equally often, the one that reached that count first
    goes first. A write counts as a use.
    """

    def __init__(self, records: OrderedDict[str, CacheEntry]) -> None:
        """Start with no buckets."""
        super().__init__(records)
        self.bucketOf: dict[str, CountBucket] = {}
        # The bucket of the lowest count in use; the rest link up from it.
        self.lowest: CountBucket | None = None

    def onAdd(self, key: str) -> None:
        """Put `key` in the bucket for one use."""
        if self.lowest is None or self.lowest.count != 1:
            self.insertBucket(1, lower=None, higher=self.lowest)
        self.lowest.keys[key] = None
        self.bucketOf[key] = self.lowest

    def onUse(self, key: str) -> None:
        """Move `key` one bucket up."""
        bucket = self.bucketOf[key]
        higher = bucket.higher
        if higher is None or higher.count != bucket.count + 1:
            higher = self.insertBucket(
                bucket.count + 1, lower=bucket, higher=higher
            )
        higher.keys[key] = None
        self.bucketOf[key] = higher
        self.leaveBucket(key, bucket)

    def onRemove(self, key: str) -> None:
        """Take `key` out of its bucket."""
        self.leaveBucket(key, self.bucketOf.pop(key))

    def evictionOrder(self) -> list[str]:
        """Every key, lowest bucket first, each bucket in its own order."""
        keys = []
        bucket = self.lowest
        while bucket is not None:
            keys.extend(bucket.keys)
            bucket = bucket.higher
        return keys

    def chooseVictim(self, protectedKey: str | None) -> str | None:
        """The first key in the lowest bucket.

        If that bucket holds only the protected key, the first key of the
        bucket above it - never empty, since empty buckets are unlinked.
        """
        bucket = self.lowest
        while bucket is not None:
            victim = firstKeyExcept(bucket.keys, protectedKey)
            if victim is not None:
                return victim
            bucket = bucket.higher
        return None

    def insertBucket(
        self,
        count: int,
        lower: CountBucket | None,
        higher: CountBucket | None,
    ) -> CountBucket:
        """Link a new empty bucket for `count` between `lower` and `higher`."""
        bucket = CountBucket(count, lower, higher)
        if lower is None:
            self.lowest = bucket
        else:
            lower.higher = bucket
        if higher is not None:
            higher.lower = bucket
        return bucket

    def leaveBucket(self, key: str, bucket: CountBucket) -> None:
        """Take `key` out of `bucket`, unlinking the bucket if left empty."""
        del bucket.keys[key]
        if bucket.keys:
            return
        if bucket.lower is None:
            self.lowest = bucket.higher
        else:
            bucket.lower.higher = bucket.higher
        if bucket.higher is not None:
            bucket.higher.lower = bucket.lower


POLICY_CLASSES: dict[str, type[EvictionPolicy]] = {
    EVICTION_LRU: LruPolicy,
    EVICTION_LFU: LfuPolicy,
    EVICTION_FIFO: FifoPolicy,
}


class MemoryCollection:
    """The records under one path, kept within the path's own limits.

    Holds no lock of its own: `MemoryDb` holds its lock around every call,
    since evicting for the store's memory limit reaches across collections.
    Built only from a config `describeProblem` has passed.
    """

    def __init__(self, path: str, config: CollectionConfig) -> None:
        """An empty collection evicting by `config`'s policy."""
        self.path = path
        self.config = config
        self.usedBytes = 0
        self.evictedCount = 0
        self.hitCount = 0
        self.missCount = 0
        self._records: OrderedDict[str, CacheEntry] = OrderedDict()
        self._policy = POLICY_CLASSES[config.evictionPolicy](self._records)

    @property
    def recordCount(self) -> int:
        """How many records the collection holds."""
        return len(self._records)

    def has(self, key: str) -> bool:
        """Whether a record is stored under `key`; not counted as a read."""
        return key in self._records

    def evictionOrder(self) -> list[str]:
        """Every key, in the order they would be evicted; not a read."""
        return self._policy.evictionOrder()

    def get(self, key: str) -> CacheEntry | None:
        """The entry under `key`, counted as a hit and a use; else a miss."""
        entry = self._records.get(key)
        if entry is None:
            self.missCount += 1
            LOGGER.debug("Miss on %s in %s.", key, self.path)
            return None
        self.hitCount += 1
        self._policy.onUse(key)
        LOGGER.debug("Hit on %s in %s.", key, self.path)
        return entry

    def put(self, key: str, value: Any, sizeBytes: int) -> None:
        """Store `value` under `key`, then evict what the limits cannot fit.

        Rewriting a record counts as a use of it. The record just put is
        never the one evicted.
        """
        entry = self._records.get(key)
        if entry is None:
            self._records[key] = CacheEntry(value, sizeBytes)
            self._policy.onAdd(key)
            LOGGER.debug(
                "Added %s to %s (%s bytes).", key, self.path, sizeBytes
            )
        else:
            self.usedBytes -= entry.sizeBytes
            entry.value = value
            entry.sizeBytes = sizeBytes
            self._policy.onUse(key)
            LOGGER.debug(
                "Replaced %s in %s (%s bytes).", key, self.path, sizeBytes
            )
        self.usedBytes += sizeBytes
        self.trim(protectedKey=key)

    def remove(self, key: str) -> CacheEntry | None:
        """Delete the record under `key`; the entry deleted, or None."""
        entry = self._drop(key)
        if entry is not None:
            LOGGER.debug("Deleted %s from %s.", key, self.path)
        return entry

    def evictOne(self, protectedKey: str | None, reason: str) -> bool:
        """Evict the policy's next victim; False when there is none to take.

        `reason` says which limit was crossed, for the log.
        """
        victim = self._policy.chooseVictim(protectedKey)
        if victim is None:
            return False
        LOGGER.info(
            "Evicting %s from %s by %s: %s.",
            victim,
            self.path,
            self.config.evictionPolicy,
            reason,
        )
        self._drop(victim)
        self.evictedCount += 1
        return True

    def _drop(self, key: str) -> CacheEntry | None:
        """Take the record under `key` out, with its bytes and policy state."""
        entry = self._records.pop(key, None)
        if entry is not None:
            self.usedBytes -= entry.sizeBytes
            self._policy.onRemove(key)
        return entry

    def isOverLimit(self) -> bool:
        """Whether the collection holds more than its config allows."""
        maxRecords = self.config.maxRecords
        maxBytes = self.config.maxBytes
        return (maxRecords is not None and self.recordCount > maxRecords) or (
            maxBytes is not None and self.usedBytes > maxBytes
        )

    def trim(self, protectedKey: str | None) -> None:
        """Evict by policy until within limits, sparing `protectedKey`."""
        while self.isOverLimit():
            if not self.evictOne(protectedKey, "collection over its limit"):
                LOGGER.warning(
                    "Record %s alone is over the limit of %s; kept anyway.",
                    protectedKey,
                    self.path,
                )
                return

    def stats(self) -> CollectionStats:
        """Where the collection stands against its limits, and its reads."""
        return CollectionStats(
            path=self.path,
            evictionPolicy=self.config.evictionPolicy,
            recordCount=self.recordCount,
            usedBytes=self.usedBytes,
            maxRecords=self.config.maxRecords,
            maxBytes=self.config.maxBytes,
            evictedCount=self.evictedCount,
            hitCount=self.hitCount,
            missCount=self.missCount,
        )
