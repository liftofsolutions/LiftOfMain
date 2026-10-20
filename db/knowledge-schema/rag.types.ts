// Knowledge (RAG) schema: rags, rag_documents, pinecone_indexes, rag_jobs.
// Connected to the Project schema (project_id) and the User schema (created_by).
// Keys: PK = primary key, FK = foreign key, UNIQUE = no two documents share the value.

// ---------- pinecone_indexes ----------
// One row per Pinecone index. Many rags share an index, each in its own namespace.

export interface PineconeIndex {
  _id: string;             // PK. UUID v4
  name: string;            // UNIQUE. Pinecone index name
  host: string;            // index host used by the client
  dimension: number;       // must match the embedding model's output size
  metric: "cosine" | "dotproduct" | "euclidean";
  embedding_model: string; // model that produces the vectors stored here
  cloud: string;           // e.g. "gcp", "aws"
  region: string;          // e.g. "us-east1"; lets a country's data stay in a matching region later
  created_at: Date;
  updated_at: Date;
}

// ---------- rags ----------
// A knowledge base. A project can have many.

export type RagStatus = "indexing" | "ready" | "failed";

export interface Rag {
  _id: string;               // PK. UUID v4
  project_id: string;        // FK → projects._id
  pinecone_index_id: string; // FK → pinecone_indexes._id
  namespace: string;         // Pinecone namespace; equals _id
  name: string;              // e.g. "India pricing & delivery"
  description?: string;
  country?: string;          // ISO 3166-1 alpha-2, e.g. "IN". Missing = applies to every country
  status: RagStatus;
  created_by: string;        // FK → users._id
  created_at: Date;
  updated_at: Date;
}

// ---------- rag_documents ----------
// Metadata for one document in a rag. The text itself lives in Pinecone as chunks.

export type DocumentSource = "upload" | "url" | "text";
export type DocumentStatus = "pending" | "ready" | "failed";

export interface RagDocument {
  _id: string;               // PK. UUID v4
  rag_id: string;            // FK → rags._id
  project_id: string;        // FK → projects._id (copied from the rag for direct queries)
  title: string;
  source: DocumentSource;
  url?: string;              // the link, when source is "url"
  file_path?: string;        // storage path, when source is "upload"
  metadata: Record<string, unknown>; // unstructured: mime type, size, checksum, language, author, tags, …
  chunk_count: number;       // vectors stored in Pinecone for this document
  status: DocumentStatus;
  error?: string;
  indexed_at?: Date;
  created_by: string;        // FK → users._id
  created_at: Date;
  updated_at: Date;
}

// ---------- rag_jobs ----------
// Temporary. Processing a document is long, so it runs as a job; the job is deleted
// automatically (TTL on expires_at) some days after it finishes.

export type JobType = "index" | "reindex" | "delete";
export type JobStatus = "created" | "submitted" | "running" | "done" | "failed";

export interface RagJob {
  _id: string;               // PK. UUID v4
  project_id: string;        // FK → projects._id
  rag_id: string;            // FK → rags._id
  document_id: string;       // FK → rag_documents._id
  type: JobType;
  status: JobStatus;
  attempts: number;
  progress?: Record<string, unknown>; // e.g. { chunks_done: 40, chunks_total: 120 }
  error?: string;
  started_at?: Date;
  finished_at?: Date;
  expires_at: Date;          // TTL: the database deletes the job after this time
  created_at: Date;
  updated_at: Date;
}

// ---------- Helpers ----------

export const MAX_JOB_ATTEMPTS = 3;
export const JOB_RETENTION_DAYS = 7;

// Allowed job status changes. "failed" can go back to "submitted" for a retry.
const JOB_TRANSITIONS: Record<JobStatus, JobStatus[]> = {
  created: ["submitted"],
  submitted: ["running", "failed"],
  running: ["done", "failed"],
  done: [],
  failed: ["submitted"],
};

export function canMoveJob(from: JobStatus, to: JobStatus, attempts: number): boolean {
  if (from === "failed" && to === "submitted" && attempts >= MAX_JOB_ATTEMPTS) return false;
  return JOB_TRANSITIONS[from].includes(to);
}

// expires_at for a job: far in the future while it runs, a few days after it finishes.
export function jobExpiry(now: Date, finished: boolean): Date {
  const days = finished ? JOB_RETENTION_DAYS : 30;
  return new Date(now.getTime() + days * 24 * 60 * 60 * 1000);
}

// Pinecone vector id for one chunk; lets one document be re-indexed or deleted on its own.
export function vectorId(documentId: string, chunkIndex: number): string {
  return `${documentId}#${chunkIndex}`;
}

// The rags that apply to a customer's country: that country's rags plus those with no country.
export function ragsForCountry(rags: Rag[], country?: string): Rag[] {
  return rags.filter((r) => r.status === "ready" && (!r.country || r.country === country));
}
