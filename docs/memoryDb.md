# memoryDb

`src/DbAdapters/memoryDb/` is a Redis-like cache that lives in the server's
own memory. There is one store per process, shared by every component through
`getMemoryDb()` and built when the app starts. It holds collections of
records by path; each collection has its own eviction policy and its own
limits, and the store as a whole has a memory limit.

Use it as a **cache** - fast lookups in front of a real store - never as the
record itself:

- **It is per process.** Two uvicorn workers, or two containers on AWS, each
  have their own store; a record written in one is invisible to the other.
- **It is lost on restart.** Nothing is written to disk.
- **It evicts.** A record can be gone the next time you look, by design.

Keep the record of truth in Firestore, MongoDB or Redis, and use this to
avoid reading it twice.

## Quick start

Import from the package:

```python
from DbAdapters.memoryDb import CollectionConfig, EVICTION_LRU, getMemoryDb

memoryDb = getMemoryDb()

# Once, before anything writes to the path - see "Collections" below.
memoryDb.createCollection(
    "ragJobs", CollectionConfig(evictionPolicy=EVICTION_LRU, maxRecords=500)
)

memoryDb.addRecord("ragJobs", jobId, job)   # insert, or replace
job = memoryDb.getRecord("ragJobs", jobId)  # None when not cached
memoryDb.deleteRecord("ragJobs", jobId)
```

Every call is a plain function, not a coroutine: no `await`, from async and
sync code alike. Calls are short and never wait on anything (see
[Thread safety](#thread-safety)).

The usual shape - read the cache, fall back to the real store, fill the
cache:

```python
async def fetchJob(jobId: str) -> dict | None:
    """The job, from the cache if it is there, else from Firestore."""
    memoryDb = getMemoryDb()
    job = memoryDb.getRecord("ragJobs", jobId)
    if job is None:
        job = await readJobFromFirestore(jobId)  # your read of the record
        if job is not None:
            memoryDb.addRecord("ragJobs", jobId, job)
    return job
```

## The calls

| Call | Does | Returns |
| --- | --- | --- |
| `createCollection(path, config=None)` | makes an empty collection with its own policy and limits | `False` if the path already has one, or the config is unusable |
| `deleteCollection(path)` | drops the collection and every record in it | whether there was one |
| `addRecord(path, key, value)` | stores `value`, replacing any record under `key` | `True`; `False` only if the store failed |
| `createRecord(path, key, value)` | stores `value` only if `key` is new | `False` if `key` is already stored |
| `getRecord(path, key, default=None)` | the stored value; counts a hit or a miss | the value, or `default` |
| `deleteRecord(path, key)` | drops one record | whether there was one |
| `getCollectionStats(path)` | one collection's size, limits, evictions, hits and misses | `CollectionStats`, or `None` if no collection |
| `getStats()` | the whole store, and every collection's stats | `MemoryDbStats` |
| `getEvictionOrder(path)` | the keys, next to be evicted first; not counted as a read | `list[str]`, or `None` if no collection |

No call raises - see [Failures](#failures).

## Collections

A **path** is just a collection's name: `"ragJobs"`, `"chunks/report"`. There
is no nesting - `"chunks"` and `"chunks/report"` are unrelated collections.
Keys are strings, unique within their collection.

**Create a collection before anything writes to its path.** A write to a path
with no collection makes one with the default config - LRU, no limits of its
own - and from then on `createCollection` for that path returns `False` and
changes nothing. The simplest place is the lifespan in `src/main.py`, right
after the store is built. To change a collection's config, delete it and
create it again.

```python
CollectionConfig(
    evictionPolicy=EVICTION_LRU,  # or EVICTION_LFU, EVICTION_FIFO
    maxRecords=None,              # at most this many records
    maxBytes=None,                # at most this many bytes of records
)
```

A limit left at `None` bounds the collection only by the store's own memory
limit. Building a `CollectionConfig` never raises; an unusable one (an unknown
policy, a limit below 1) makes `createCollection` log an error and return
`False`.

### Choosing a policy

| Policy | Evicts | A read | Suits |
| --- | --- | --- | --- |
| `EVICTION_LRU` | the least recently used record | makes it the last to go | the default; anything read again soon after |
| `EVICTION_LFU` | the least often used; among equals, the first to reach that count | adds one to its count | a hot set read over and over |
| `EVICTION_FIFO` | the record stored first | changes nothing | records that simply age, read or not |

Writing a key that is already stored counts as a use for LRU and LFU; FIFO
keeps the record where it first went in.

The same calls on each policy, from the tests: a collection capped at five
records holds `k0` to `k4`; `k1` is read three times, then `k2`, `k3`, `k4`
and `k0` once each; then `k5`, `k6` and `k7` are added.

| Policy | Evicts, in order | Why |
| --- | --- | --- |
| LRU | `k1`, `k2`, `k3` | `k1` was read the most, but the longest ago |
| LFU | `k2`, `k5`, `k6` | `k2` is the first of the least used (`k5` is spared as the record just written); after that each newcomer has the fewest uses, so the one before it goes |
| FIFO | `k0`, `k1`, `k2` | reads do not matter, only the order stored |

That LFU row is worth knowing before choosing it: in a full LFU collection, a
record that is not read soon after it is written is the first to go.

## Limits: what happens on a write

A write is **never refused for space**. The record goes in, then:

1. its collection evicts, by its own policy, until it is back within its own
   `maxRecords` and `maxBytes`;
2. if the whole store is then over its memory limit, it evicts one record at
   a time from whichever collection holds the most bytes, by that
   collection's policy, until it fits.

The record just written is never the one evicted. A record that alone is
bigger than a limit is still kept: everything it competes with is evicted,
and a warning is logged.

The shared store's limit is `DEFAULT_MAX_BYTES`, 256 MB. There is no setting
for it yet; `MemoryDb(maxBytes=...)` builds a store with another limit, for
tests.

## Things to know

**Values are kept as they are, not copied.** Anything can be stored - a dict,
a pydantic model, a client - and `getRecord` hands back the very same object.
So changing a stored value in place changes the cached record, and the bytes
counted for it stay what they were when it was written. Write it again to
recount.

**Bytes are an estimate.** A record weighs `sys.getsizeof(key)` plus
`measureBytes(value)`, which walks the value - dict keys and values, list,
tuple and set items, object attributes - counting each object once.
Functions, classes and modules count only as themselves. It estimates what
the records hold, not the process's memory, and it takes time in proportion
to the size of the value.

**Missing and `None` look alike.** `getRecord` returns `default` (`None`) for
a key that is not stored. If `None` can be a stored value, pass a default of
your own:

```python
MISSING = object()

value = memoryDb.getRecord("flags", key, MISSING)
if value is MISSING:
    ...
```

### Failures

No call raises: the store is a cache, and a failure in it must not take a
request or the server down. Each failure is logged with its traceback and
answered with a safe value - `False` from the writes and deletes, `None` from
the stats calls, `default` from `getRecord`. For example:

- a key that cannot be hashed, such as a list: the write returns `False`;
- a value whose size cannot be measured: it is stored anyway, counted for
  what could be measured;
- an unusable config: `createCollection` returns `False`;
- an unusable `maxBytes` for `MemoryDb`: the default is used instead.

So check the `bool` a write returns wherever it matters.

### Thread safety

Each call takes the store's one lock, so the store is safe from the event loop
and from the thread pool FastAPI runs sync handlers on alike. Nothing waits
while holding it, and a value is measured before the lock is taken, so a big
value does not hold up other calls.

## Seeing what it does

`getCollectionStats(path)` returns a `CollectionStats`:

| Field | Meaning |
| --- | --- |
| `path`, `evictionPolicy`, `maxRecords`, `maxBytes` | the collection's config |
| `recordCount`, `usedBytes` | what it holds now |
| `evictedCount` | records evicted since it was made |
| `hitCount`, `missCount` | `getRecord` calls that found a record, and that did not |

`getStats()` returns a `MemoryDbStats`: the store's `usedBytes`, `maxBytes`
and `recordCount`, and every collection's `CollectionStats`. Hits and misses
are counted per collection only; a read of a path with no collection is not
counted anywhere.

The store logs under `DbAdapters.memoryDb`:

| Level | Logged |
| --- | --- |
| `DEBUG` | every hit, miss, add, replace and delete |
| `INFO` | every eviction: `Evicting k1 from ragJobs by lru: collection over its limit.` |
| `WARNING` | a record kept although it is alone over a limit |
| `ERROR` | a collection not made, and every failure, with its traceback |

Nothing in this service configures logging yet, so out of the box only
warnings and errors are printed. To see evictions, or every call:

```python
import logging

logging.basicConfig(level=logging.INFO)  # evictions
logging.getLogger("DbAdapters.memoryDb").setLevel(logging.DEBUG)  # every call
```

`DEBUG` writes a line for every call - leave it off in production.

## Cost

`n` is the number of collections. How many records a collection holds slows
none of the calls except `getEvictionOrder`, which lists them, and
`deleteCollection`, which frees them.

| Call | Cost |
| --- | --- |
| `getRecord`, `getCollectionStats`, `createCollection` | O(1) |
| `deleteRecord` | O(log n) |
| `addRecord`, `createRecord` | O(log n), plus measuring the value, plus any evictions |
| an eviction for a collection's own limit | O(1) |
| an eviction for the store's limit | O(log n) |
| `getEvictionOrder` | O(records in the collection) |
| `getStats` | O(n) |
| `deleteCollection` | O(1), plus freeing its records |

On a development machine, a million requests per policy - half adds, half
fetches - on a collection kept full at 1,000 records:

```
policy    total ms  us/request    us/add  us/fetch  evictions  us/eviction
lru         2688.5        2.69      4.78      0.59    248,863         0.88
lfu         2884.7        2.88      5.03      0.74    249,807         0.96
fifo        2554.1        2.55      4.56      0.54    248,754         0.80
```

An add includes measuring its value (a small dict here) and any eviction it
makes.

## How it works

```
MemoryDb                          one per process: getMemoryDb()
  "ragJobs" -> MemoryCollection   LRU, maxRecords=500
                 OrderedDict: key -> CacheEntry(value, sizeBytes)
  "chunks"  -> MemoryCollection   LFU, maxBytes=50 MB
                 OrderedDict, plus its keys in use-count buckets
```

| File | Holds |
| --- | --- |
| `memoryDb.py` | `MemoryDb`, `getMemoryDb`, `measureBytes` |
| `memoryCollection.py` | `MemoryCollection` and the eviction policies |
| `model.py` | `CollectionConfig`, `CacheEntry`, the stats classes, the policy names |

**A collection** is one `OrderedDict` from key to `CacheEntry` (the value and
its bytes). Keys are unique, so the dict is the only index needed: every
read, write and delete is one lookup. Its order is also the eviction order for
LRU and FIFO - LRU moves a key to the end on every use, FIFO never moves it,
and both evict the first key. LFU keeps its keys in buckets by use count,
linked lowest first: a use moves a key one bucket up, and the victim is the
first key of the lowest bucket. Only the record just written is ever skipped,
so picking a victim looks at two keys at most.

**The store** keeps the collections by path, a running total of their bytes,
and a heap of collections by bytes, so the largest is always to hand when the
store has to evict. When a collection's size changes, its old heap entry is
not hunted down; stale entries are dropped as they surface, and the heap is
rebuilt once they outnumber the live ones.

## Tests

`Btest/DbAdapters/MemoryDbTest.py` drives one collection per policy and
checks every call twice: against exact expected keys, and against `ListCache`,
the same cache kept the slow obvious way in a plain list. After every call the
store's eviction order must equal the list. It covers adding and fetching,
exhausting each limit, who is evicted (and that the log names them), and two
stress runs of `STRESS_REQUESTS` (a million) calls per policy: one timed, one
checked against the list on every call.

```bash
uv run pytest Btest/DbAdapters/MemoryDbTest.py -q    # about 75 s
uv run pytest Btest/DbAdapters/MemoryDbTest.py -q \
    -k "not testRequestTimesForEveryPolicy and not testEveryEvictionInALongRun"
```

The second leaves out the four stress tests and runs in a few seconds.

In a test of your own, build a `MemoryDb()` rather than using `getMemoryDb()`,
so tests do not share records.

## Changing it

- **Keep every call O(1), and O(log n) at worst.** O(n) only where nothing
  less can do the job - measuring a value, listing a collection.
- **Keep every public call from raising.** Wrap a new one in `@failSafe`, or
  in a `try` as `getRecord` does.
- **To add a policy:** name it in `model.py` and add it to
  `EVICTION_POLICIES`; subclass `EvictionPolicy` in `memoryCollection.py`
  (`onAdd`, `onUse`, `onRemove`, `chooseVictim`, `evictionOrder`) and register
  it in `POLICY_CLASSES`; then teach the test's `ListCache` where the policy
  puts a key, and give it rows in the test's expected-victim tables.
