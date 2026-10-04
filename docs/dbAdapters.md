# DbAdapters

One async CRUD surface over the databases the project supports, so swapping a
store is a one-line change at the import rather than a rewrite at every call
site.

```python
from DbAdapters import MongoAdapter, Document
```

Import from the package, not from the vendor modules.

> Each adapter module carries a long docstring on what its store actually
> guarantees. Those are the authoritative reference; this page is the map.

## Two contracts, because there are two shapes of store

| Contract | For | Implemented by |
| --- | --- | --- |
| `VectorDbAdapter` | embeddings, searched by similarity | Pinecone |
| `DocumentDbAdapter` | documents, keyed by id | MongoDB, Firestore, Redis |

Both extend `DbAdapter` in `base.py`, which holds what they share regardless of
store: the `connect()` / `close()` lifecycle, and how a partly-applied write is
reported.

**`VectorDbAdapter`** - `upsertVectors`, `fetchVectors`, `queryVectors`,
`deleteVectors`, over `VectorRecord` (id, values, metadata) and `QueryMatch` (id,
score, values, metadata).

**`DocumentDbAdapter`** - `fetchDocument(s)`, `upsertDocument(s)`,
`deleteDocument(s)`, and `applyAtomically`, over `Document` (id, data) and
`WriteOperation` (kinds `set`, `merge`, `delete`).

The document surface is narrow on purpose: it is the honest intersection of three
stores that diverge sharply past it. Querying is not in it - MongoDB and
Firestore have real query engines and each exposes `findDocuments` on its own
class, Redis has key patterns and exposes `scanDocumentIds` instead. Reaching a
store's own strengths through its own module beats pretending a lowest common
denominator covers them.

## Partial writes are reported, not hidden

Every store here caps how much one request may carry, so a bulk write is split
into batches - and a batch failing partway leaves the earlier ones applied. The
bulk methods return a `WriteOutcome` saying exactly what happened, rather than
raising and leaving the caller guessing how far it got:

- `succeeded` - did all of it land
- `failedIds` - what did not
- `retryableIds` - of those, what is worth sending again

`WriteFailure` carries the message and a `retryable` flag per failed slice.
`chunkItems` in `base.py` does the splitting; only the size caps differ between
stores.

## What ACID means, per store

The most important thing to know before choosing one. The differences are large
and none of them can be papered over by an adapter.

| | Atomicity | Isolation | Read after write | Durability |
| --- | --- | --- | --- | --- |
| **MongoDB** | real multi-document transactions, server-side rollback | snapshot | consistent | write-concern governed, `majority` by default |
| **Firestore** | batched writes and transactions commit whole, across collections | serializable | strongly consistent | replicated before commit returns |
| **Redis** | `MULTI`/`EXEC` runs uninterrupted but **cannot roll back** | total during a block | consistent | weak - `appendfsync everysec` can lose a second |
| **Pinecone** | per vector only | none | **eventually** consistent | acknowledged writes are durable |

`applyAtomically` is the one method all three document stores promise
all-or-nothing on, and even there the promise differs - a real transaction on
MongoDB, an atomic batch on Firestore, and on Redis a queued block nothing else
interleaves with but which has nothing to undo itself with. Read the module
docstring before leaning on it.

Some consequences worth stating plainly:

- **Pinecone reads are eventually consistent.** A fetch issued straight after an
  upsert may legitimately miss it. `upsertVectorsAtomically` adds a compensating
  rollback for callers that need all-or-nothing, but with no isolation that
  rollback can undo a write another caller made in between.
- **Redis is a cache that survives restarts if you are lucky**, not a record of
  truth. Right home for cached retrievals, dedup markers, rate limits, session
  state.
- **MongoDB transactions need a replica set** or sharded cluster; a standalone
  `mongod` rejects them, and the adapter turns that into a message saying so
  rather than a bare `OperationFailure`.
- **Firestore's ceiling is size, not semantics**: 500 operations per batch or
  transaction, and no document written twice within one. Both are checked before
  the request goes out.

## Async that means it

Marking a method `async def` does not stop it blocking - an `async def` wrapping a
*synchronous* client still owns the event loop for the whole round trip, and
every other request on that worker waits behind it.

So each adapter drives a genuinely async transport: `PineconeAsyncio`,
`AsyncMongoClient` (PyMongo's own, which replaced Motor),
`google-cloud-firestore`'s `AsyncClient` (rather than `firebase-admin`, whose
Firestore surface is synchronous), and async `redis`.

Pure helpers stay ordinary functions. There is nothing to await while rebuilding
a dict, and marking one `async` adds call overhead and forces callers to await
something that never yields.

## Configuration

Constructor arguments first, falling back to environment variables named in each
module's `*_ENV_VAR` constants:

| Store | Variables |
| --- | --- |
| Pinecone | `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_INDEX_HOST`, `PINECONE_NAMESPACE` |
| MongoDB | `MONGODB_URI`, `MONGODB_DATABASE` |
| Firestore | `GOOGLE_CLOUD_PROJECT`, `FIRESTORE_DATABASE`, `GOOGLE_APPLICATION_CREDENTIALS` |
| Redis | `REDIS_URL`, `REDIS_KEY_PREFIX` |

Firestore also accepts the ambient credentials available on Cloud Run and GCE.

**Do not read those variables yourself.** Nothing loads `server.env` /
`test.env` into the environment, and this project does not spell every
variable the way the adapters do - it uses
`GCP_PROJECT_ID` and `FIRESTORE_DATABASE_ID` where the adapters default to
`GOOGLE_CLOUD_PROJECT` and `FIRESTORE_DATABASE`. The `config` component reconciles
the two and is the intended entry point:

```python
from config import buildFirestoreAdapter, buildPineconeAdapter

store = buildFirestoreAdapter()          # project, database and credentials wired in
index = buildPineconeAdapter()           # api key, index name and host wired in
```

It also anchors a relative `GOOGLE_APPLICATION_CREDENTIALS` to the repository
root, so a worker started from another directory finds the key. A store that is
not configured raises `ConfigurationError` naming the variable it wants, at the
moment it is asked for rather than at import.

One gotcha worth keeping in mind: this project's Firestore database is *named*
`default`, which is not Firestore's reserved `(default)` for an unnamed database.
They are different databases and `FIRESTORE_DATABASE_ID` has to say which.

## Integrated embedding on Pinecone

The configured index, `rag-chunks`, declares `chunkText` as a semantic text field
and embeds server-side with `llama-text-embed-v2`. That changes which methods
apply:

| Want | Integrated index | Raw vector index |
| --- | --- | --- |
| Write | `upsertTextRecords` | `upsertVectors` |
| Search | `searchByText` | `queryVectors` |

`upsertVectors` against an integrated index bypasses the semantic field and
leaves records no text search will ever match, so the write path is not
interchangeable. Reads are: the vectors underneath are ordinary dense 1024-dim
ones, so `queryVectors` works when you hold an embedding of your own, and
`fetchVectors`, `listVectorIds` and `deleteVectors` behave normally either way.
`upsertTextRecords` and `searchByText` both require a namespace - the records API
has no default one.

## Escape hatches

Where a store has something the shared contract cannot express, the adapter hands
over the client rather than growing a leaky abstraction: `getClient()` on Redis
and Firestore, `getIndex()` / `getDatabase()` / `getCollection()` on the others.
Redis expiry is the deliberate exception, wired into the write methods, because a
cache entry with no TTL is usually a bug and a second call is easy to forget.

## Tests

`Btest/config/ConfigTest.py` covers the configuration layer: variable
translation, credential path resolution, the `has*` flags and what each factory
hands its adapter.

The adapters themselves have none yet. `Btest/DbAdapters/` is where they go -
logic and payload shaping tested directly against fakes, per
[testing.md](testing.md).
