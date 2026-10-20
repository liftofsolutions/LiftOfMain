// rags, rag_documents, pinecone_indexes and rag_jobs collections for MongoDB: validators + indexes.
// Run with mongosh:  mongosh "<connection string>/<db>" rag.mongo.js
// Safe to re-run. Extra fields are allowed, so the schema stays flexible.

const uuid = {
  bsonType: "string",
  pattern: "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
};
const timestamps = {
  created_at: { bsonType: "date" },
  updated_at: { bsonType: "date" },
};

const pineconeIndexesValidator = {
  $jsonSchema: {
    bsonType: "object",
    required: ["_id", "name", "host", "dimension", "metric", "embedding_model", "cloud", "region", "created_at", "updated_at"],
    properties: {
      _id: uuid,
      name: { bsonType: "string", minLength: 1 },
      host: { bsonType: "string", minLength: 1 },
      dimension: { bsonType: "int", minimum: 1 },
      metric: { enum: ["cosine", "dotproduct", "euclidean"] },
      embedding_model: { bsonType: "string", minLength: 1 },
      cloud: { bsonType: "string" },
      region: { bsonType: "string" },
      ...timestamps,
    },
  },
};

const ragsValidator = {
  $jsonSchema: {
    bsonType: "object",
    required: ["_id", "project_id", "pinecone_index_id", "namespace", "name", "status", "created_by", "created_at", "updated_at"],
    properties: {
      _id: uuid,
      project_id: uuid,        // FK → projects._id
      pinecone_index_id: uuid, // FK → pinecone_indexes._id
      namespace: { bsonType: "string", minLength: 1 }, // equals _id
      name: { bsonType: "string", minLength: 1, maxLength: 120 },
      description: { bsonType: "string", maxLength: 2000 },
      country: { bsonType: "string", pattern: "^[A-Z]{2}$" }, // missing = every country
      status: { enum: ["indexing", "ready", "failed", "deleting"] },
      created_by: uuid,        // FK → users._id
      ...timestamps,
    },
  },
};

const ragDocumentsValidator = {
  $jsonSchema: {
    bsonType: "object",
    required: ["_id", "rag_id", "project_id", "title", "source", "metadata", "chunk_count", "status", "created_by", "created_at", "updated_at"],
    properties: {
      _id: uuid,
      rag_id: uuid,     // FK → rags._id
      project_id: uuid, // FK → projects._id
      title: { bsonType: "string", minLength: 1, maxLength: 300 },
      source: { enum: ["upload", "url", "text"] },
      url: { bsonType: "string" },
      file_path: { bsonType: "string" },
      metadata: { bsonType: "object" }, // unstructured
      chunk_count: { bsonType: "int", minimum: 0 },
      status: { enum: ["pending", "ready", "failed"] },
      error: { bsonType: "string" },
      indexed_at: { bsonType: "date" },
      created_by: uuid, // FK → users._id
      ...timestamps,
    },
  },
};

const ragJobsValidator = {
  $jsonSchema: {
    bsonType: "object",
    required: ["_id", "project_id", "rag_id", "document_id", "type", "status", "attempts", "expires_at", "created_at", "updated_at"],
    properties: {
      _id: uuid,
      project_id: uuid,  // FK → projects._id
      rag_id: uuid,      // FK → rags._id
      document_id: uuid, // FK → rag_documents._id
      type: { enum: ["index", "reindex", "delete"] },
      status: { enum: ["created", "submitted", "running", "done", "failed"] },
      attempts: { bsonType: "int", minimum: 0 },
      progress: { bsonType: "object" },
      error: { bsonType: "string" },
      started_at: { bsonType: "date" },
      finished_at: { bsonType: "date" },
      expires_at: { bsonType: "date" },
      ...timestamps,
    },
  },
};

function upsertCollection(name, validator) {
  const options = { validator, validationLevel: "strict", validationAction: "error" };
  if (db.getCollectionNames().includes(name)) {
    db.runCommand({ collMod: name, ...options });
  } else {
    db.createCollection(name, options);
  }
}

upsertCollection("pinecone_indexes", pineconeIndexesValidator);
upsertCollection("rags", ragsValidator);
upsertCollection("rag_documents", ragDocumentsValidator);
upsertCollection("rag_jobs", ragJobsValidator);

// UNIQUE: pinecone_indexes.name.
db.pinecone_indexes.createIndex({ name: 1 }, { name: "name_unique", unique: true });

// rags: a project's rags, and its rags for one country.
db.rags.createIndex({ project_id: 1, country: 1 }, { name: "project_country" });

// rag_documents: a rag's documents, newest first; a project's documents.
db.rag_documents.createIndex({ rag_id: 1, created_at: -1 }, { name: "rag_created" });
db.rag_documents.createIndex({ project_id: 1 }, { name: "project_id" });

// rag_jobs: jobs for a document or a rag; workers picking up work.
db.rag_jobs.createIndex({ document_id: 1 }, { name: "document_id" });
db.rag_jobs.createIndex({ rag_id: 1, status: 1 }, { name: "rag_status" });

// rag_jobs is temporary: each job is deleted once expires_at has passed.
db.rag_jobs.createIndex({ expires_at: 1 }, { name: "expires_ttl", expireAfterSeconds: 0 });

print("pinecone_indexes, rags, rag_documents and rag_jobs ready");
