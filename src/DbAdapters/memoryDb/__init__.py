"""A Redis-like store in this process's memory, for the whole server.

    from DbAdapters.memoryDb import getMemoryDb
    getMemoryDb().addRecord("ragJobs", jobId, job)

`memoryDb` is the store and its singleton, `memoryCollection` one path's
records and the eviction policies, `model` the classes both use. Kept
under `DbAdapters` with the other stores, in its own folder because it is
more than one module. No call into the store raises.
"""

from DbAdapters.memoryDb.memoryCollection import MemoryCollection
from DbAdapters.memoryDb.memoryDb import (
    DEFAULT_MAX_BYTES,
    MemoryDb,
    getMemoryDb,
    measureBytes,
)
from DbAdapters.memoryDb.model import (
    EVICTION_FIFO,
    EVICTION_LFU,
    EVICTION_LRU,
    EVICTION_POLICIES,
    CacheEntry,
    CollectionConfig,
    CollectionStats,
    MemoryDbStats,
)

__all__ = [
    "DEFAULT_MAX_BYTES",
    "EVICTION_FIFO",
    "EVICTION_LFU",
    "EVICTION_LRU",
    "EVICTION_POLICIES",
    "CacheEntry",
    "CollectionConfig",
    "CollectionStats",
    "MemoryCollection",
    "MemoryDb",
    "MemoryDbStats",
    "getMemoryDb",
    "measureBytes",
]
