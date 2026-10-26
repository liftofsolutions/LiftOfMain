# Knowledge (RAG) schema

Knowledge bases for a project: which documents they hold, where their vectors live in Pinecone,
and the jobs that process documents. MongoDB-style documents; works on Firestore today.

## Files

| File | What it is |
|---|---|
| `rag-erd.svg`, `rag-erd.png` | Diagram with relationships and cardinality |
| `rag.types.ts` | TypeScript types, job status rules, vector id and country helpers |
| `rag.mongo.js` | MongoDB validators and indexes, including the TTL that removes old jobs |
| `rag.example.json` | A Pinecone index, two rags (India, all countries), two documents, two jobs, one vector |

## Collections

| Collection | One document per | Holds |
|---|---|---|
| `rags` | Knowledge base (many per project) | Name, optional `country`, Pinecone index and namespace, status |
| `rag_documents` | Document in a rag | Title, link or file path, unstructured `metadata`, chunk count, status |
| `pinecone_indexes` | Pinecone index | Name, host, dimension, metric, embedding model, cloud and region |
| `rag_jobs` | Processing task (temporary) | Type, status, attempts, progress; deleted automatically after it finishes |

## Relationships

```
projects          1 ── 0..N  rags
pinecone_indexes  1 ── 0..N  rags          (rag.namespace = rag._id)
rags              1 ── 0..N  rag_documents
rags              1 ── 0..N  rag_jobs
rag_documents     1 ── 0..N  rag_jobs      (index, then later reindex or delete)
```

## Country

`rags.country` is an ISO code such as `IN`. Leave it empty for knowledge that applies everywhere.
`ragsForCountry()` returns the ready rags for a customer's country plus the ones with no country.

## Pinecone

- Many rags share one index; each rag is its own namespace (`namespace = rag._id`). Deleting a rag deletes its namespace.
- An index's `dimension` must match its `embedding_model`.
- Vector id: `{document_id}#{chunk_index}`, with `document_id` and `rag_id` in the vector metadata,
  so one document can be re-indexed or deleted without touching the rest.
- `cloud` and `region` allow a country's data to stay in a matching region later.

## Processing a document

```
1. create   rag_documents (pending)  +  rag_jobs (created)
2. submit   send the job to a worker queue (e.g. Cloud Tasks / Pub/Sub)        → submitted
3. run      fetch/parse → chunk → embed → upsert into the rag's namespace     → running, progress updated
4. finish   document ready + chunk_count, rag ready                           → done
            on error: attempts + 1 and retry; after MAX_JOB_ATTEMPTS (3)       → failed, with error
5. expire   expires_at = finished_at + 7 days; the TTL index deletes the job
```

Allowed job status changes (`canMoveJob()`):
`created → submitted → running → done | failed`, and `failed → submitted` for a retry while attempts < 3.

On Firestore, set a TTL policy on `rag_jobs.expires_at` for the same automatic cleanup.

## Indexes

| Collection | Index | Use |
|---|---|---|
| `pinecone_indexes` | `name` (unique) | One row per Pinecone index |
| `rags` | `project_id, country` | A project's rags, and its rags for one country |
| `rag_documents` | `rag_id, created_at` | A rag's documents, newest first |
| `rag_documents` | `project_id` | A project's documents |
| `rag_jobs` | `document_id` | A document's jobs |
| `rag_jobs` | `rag_id, status` | Jobs still running for a rag |
| `rag_jobs` | `expires_at` (TTL) | Automatic cleanup |

## Open item

An agent searches the one rag its own record names: `rag_enabled`, with `rag_id` (see `db/agent-schema`).
Whether an agent should search several - automatic by the customer's country (`ragsForCountry()`), or an
explicit `rag_ids` list on the agent - is still to decide.
