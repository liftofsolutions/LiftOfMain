"""The classes the in-memory store is built from and reports with.

`CollectionConfig` is what a caller hands `createCollection`; `CacheEntry`
is one stored record; the `*Stats` classes are what the stats calls hand
back. Plain slotted dataclasses rather than pydantic models: a store can
hold a great many entries, and each one should cost no more than its two
fields.
"""

from dataclasses import dataclass
from typing import Any

# Which record a full collection gives up first.
EVICTION_LRU = "lru"  # the least recently used
EVICTION_LFU = "lfu"  # the least often used
EVICTION_FIFO = "fifo"  # the first stored

EVICTION_POLICIES = (EVICTION_LRU, EVICTION_LFU, EVICTION_FIFO)


def isLimit(limit: Any) -> bool:
    """Whether `limit` is usable as a limit: unset, or a whole number >= 1."""
    return limit is None or (
        isinstance(limit, int) and not isinstance(limit, bool) and limit >= 1
    )


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    """How one collection picks what to evict, and how much it may keep.

    A limit left at None bounds the collection only by the store's own
    memory limit. Building one never raises; `createCollection` checks it
    with `describeProblem` and refuses a config that cannot work.
    """

    evictionPolicy: str = EVICTION_LRU
    maxRecords: int | None = None
    maxBytes: int | None = None

    def describeProblem(self) -> str | None:
        """What makes this config unusable, or None when it is usable."""
        if self.evictionPolicy not in EVICTION_POLICIES:
            return (
                f"unknown eviction policy {self.evictionPolicy!r}; "
                f"expected one of {EVICTION_POLICIES}"
            )
        if not (isLimit(self.maxRecords) and isLimit(self.maxBytes)):
            return "maxRecords and maxBytes must be None or 1 or more"
        return None


@dataclass(slots=True)
class CacheEntry:
    """One stored record: the value itself, and the bytes counted for it."""

    value: Any
    sizeBytes: int


@dataclass(frozen=True, slots=True)
class CollectionStats:
    """Where one collection stands against its limits, and how reads went.

    `hitCount` and `missCount` count `getRecord` calls on this collection
    that found a record and that did not.
    """

    path: str
    evictionPolicy: str
    recordCount: int
    usedBytes: int
    maxRecords: int | None
    maxBytes: int | None
    evictedCount: int
    hitCount: int
    missCount: int


@dataclass(frozen=True, slots=True)
class MemoryDbStats:
    """Where the whole store stands against its memory limit."""

    usedBytes: int
    maxBytes: int
    recordCount: int
    collections: list[CollectionStats]
