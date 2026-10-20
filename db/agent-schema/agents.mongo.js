// node_types, agent_templates, agents and credentials collections for MongoDB: validators + indexes.
// Run with mongosh:  mongosh "<connection string>/<db>" agents.mongo.js
// Safe to re-run. Extra fields are allowed, so the schema stays flexible.
// Cross-document rules (same project, trigger → io node, unique names within a template)
// are checked by the backend before saving; see validateTemplate() in agents.types.ts.

const uuid = {
  bsonType: "string",
  pattern: "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
};
const timestamps = {
  created_at: { bsonType: "date" },
  updated_at: { bsonType: "date" },
};

const nodeTypesValidator = {
  $jsonSchema: {
    bsonType: "object",
    required: ["_id", "category", "handler", "name", "config", "is_active", "created_at", "updated_at"],
    properties: {
      _id: uuid,
      project_id: uuid, // FK → projects._id. Missing = platform node type
      category: { enum: ["chat_model", "tool", "io"] },
      handler: { bsonType: "string", minLength: 1 }, // backend integration key, e.g. "whatsapp"
      name: { bsonType: "string", minLength: 1, maxLength: 80 },
      config: { bsonType: "object" }, // default config
      is_active: { bsonType: "bool" },
      ...timestamps,
    },
  },
};

const node = {
  bsonType: "object",
  required: ["id", "node_type_id", "name", "config", "created_at", "updated_at"],
  properties: {
    id: uuid,
    node_type_id: uuid,  // FK → node_types._id
    credential_id: uuid, // FK → credentials._id (same project)
    name: { bsonType: "string", minLength: 1, maxLength: 60 },
    config: { bsonType: "object" }, // overrides node_types.config
    ...timestamps,
  },
};

const trigger = {
  bsonType: "object",
  required: ["id", "node_id", "event", "enabled", "created_at", "updated_at"],
  properties: {
    id: uuid,      // used in the webhook URL
    node_id: uuid, // FK → nodes[].id in the same template
    event: { bsonType: "string", minLength: 1 }, // e.g. "message.received", "schedule"
    enabled: { bsonType: "bool" },
    ...timestamps,
  },
};

const agentTemplatesValidator = {
  $jsonSchema: {
    bsonType: "object",
    required: ["_id", "name", "rag_enabled", "nodes", "triggers", "is_active", "created_at", "updated_at"],
    properties: {
      _id: uuid,
      project_id: uuid, // FK → projects._id. Missing = premade platform template
      name: { bsonType: "string", minLength: 1, maxLength: 120 },
      description: { bsonType: "string", maxLength: 2000 },
      rag_enabled: { bsonType: "bool" },
      nodes: { bsonType: "array", items: node },
      triggers: { bsonType: "array", items: trigger },
      is_active: { bsonType: "bool" },
      ...timestamps,
    },
  },
};

const agentsValidator = {
  $jsonSchema: {
    bsonType: "object",
    required: ["_id", "project_id", "template_id", "name", "status", "created_by", "created_at", "updated_at"],
    properties: {
      _id: uuid,
      project_id: uuid,  // FK → projects._id
      template_id: uuid, // FK → agent_templates._id
      name: { bsonType: "string", minLength: 1, maxLength: 120 },
      status: { enum: ["active", "paused"] },
      created_by: uuid,  // FK → users._id
      ...timestamps,
      deleted_at: { bsonType: "date" },
    },
  },
};

const credentialsValidator = {
  $jsonSchema: {
    bsonType: "object",
    required: ["_id", "project_id", "name", "secret_values", "created_by", "created_at", "updated_at"],
    properties: {
      _id: uuid,
      project_id: uuid, // FK → projects._id
      name: { bsonType: "string", minLength: 1, maxLength: 120 },
      values: { bsonType: "object" }, // not secret
      secret_values: {
        // AES-256-GCM, data key wrapped by KMS. Never returned to clients.
        bsonType: "object",
        required: ["ciphertext", "iv", "tag", "wrapped_dek", "kms_key_id"],
        properties: {
          ciphertext: { bsonType: "string" },
          iv: { bsonType: "string" },
          tag: { bsonType: "string" },
          wrapped_dek: { bsonType: "string" },
          kms_key_id: { bsonType: "string" },
        },
      },
      expires_at: { bsonType: "date" },
      created_by: uuid, // FK → users._id
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

upsertCollection("node_types", nodeTypesValidator);
upsertCollection("agent_templates", agentTemplatesValidator);
upsertCollection("agents", agentsValidator);
upsertCollection("credentials", credentialsValidator);

// node_types: platform types (project_id missing) and a project's own presets.
db.node_types.createIndex({ project_id: 1 }, { name: "project_id" });

// agent_templates: premade templates (project_id missing) and a project's own templates.
db.agent_templates.createIndex({ project_id: 1 }, { name: "project_id" });

// agents: a project's agents; agents running a template.
db.agents.createIndex({ project_id: 1 }, { name: "project_id" });
db.agents.createIndex({ template_id: 1 }, { name: "template_id" });

// credentials: a project's credentials.
db.credentials.createIndex({ project_id: 1 }, { name: "project_id" });

print("node_types, agent_templates, agents and credentials ready");
