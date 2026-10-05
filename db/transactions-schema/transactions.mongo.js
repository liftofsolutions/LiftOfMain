// transactions, prices and agent_storage collections for MongoDB: validators + indexes.
// Run with mongosh:  mongosh "<connection string>/<db>" transactions.mongo.js
// Safe to re-run. Extra fields are allowed, so the schema stays flexible.
// Money is integer micro-USD (1 USD = 1,000,000).

const uuid = {
  bsonType: "string",
  pattern: "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
};
const micros = { bsonType: ["int", "long"], minimum: 0 };
const timestamps = {
  created_at: { bsonType: "date" },
  updated_at: { bsonType: "date" },
};

const nodeCost = {
  bsonType: "object",
  required: ["node_id", "node_name", "price_id", "usage", "cost_micros"],
  properties: {
    node_id: uuid,   // → agent_templates.nodes[].id
    node_name: { bsonType: "string" },
    price_id: uuid,  // FK → prices._id
    usage: { bsonType: "object" }, // raw
    cost_micros: micros,
  },
};

const stepCost = {
  bsonType: "object",
  required: ["step", "total_micros", "nodes"],
  properties: {
    step: { bsonType: "int", minimum: 1 },
    total_micros: micros,
    nodes: { bsonType: "array", items: nodeCost },
  },
};

const transactionsValidator = {
  $jsonSchema: {
    bsonType: "object",
    required: ["_id", "project_id", "agent_id", "trigger_id", "input", "status", "total_micros", "steps", "created_at", "updated_at"],
    properties: {
      _id: uuid,
      project_id: uuid, // FK → projects._id
      agent_id: uuid,   // FK → agents._id
      trigger_id: uuid, // → agent_templates.triggers[].id
      input: { bsonType: "object" },  // raw JSON
      output: { bsonType: "object" }, // raw JSON
      status: { enum: ["running", "succeeded", "failed"] },
      error: { bsonType: "string" },
      total_micros: micros,
      steps: { bsonType: "array", items: stepCost },
      finished_at: { bsonType: "date" },
      ...timestamps,
    },
  },
};

const pricesValidator = {
  $jsonSchema: {
    bsonType: "object",
    required: ["_id", "provider", "handler", "sku", "unit", "price_micros", "per_units", "effective_from", "created_at", "updated_at"],
    properties: {
      _id: uuid,
      provider: { bsonType: "string", minLength: 1 },
      handler: { bsonType: "string", minLength: 1 }, // matches node_types.handler
      sku: { bsonType: "string", minLength: 1 },
      unit: { bsonType: "string", minLength: 1 },
      country: { bsonType: "string", pattern: "^[A-Z]{2}$" }, // missing = every country
      price_micros: micros,
      per_units: { bsonType: ["int", "long"], minimum: 1 },
      effective_from: { bsonType: "date" },
      effective_to: { bsonType: "date" },
      ...timestamps,
    },
  },
};

const agentStorageValidator = {
  $jsonSchema: {
    bsonType: "object",
    required: ["_id", "project_id", "agent_id", "kind", "key", "data", "created_at", "updated_at"],
    properties: {
      _id: uuid,
      project_id: uuid, // FK → projects._id
      agent_id: uuid,   // FK → agents._id
      kind: { bsonType: "string", minLength: 1 },
      key: { bsonType: "string", minLength: 1 },
      data: { bsonType: "object" }, // unstructured
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

upsertCollection("transactions", transactionsValidator);
upsertCollection("prices", pricesValidator);
upsertCollection("agent_storage", agentStorageValidator);

// transactions: cost and history per agent and per project over time.
db.transactions.createIndex({ agent_id: 1, created_at: -1 }, { name: "agent_created" });
db.transactions.createIndex({ project_id: 1, created_at: -1 }, { name: "project_created" });

// UNIQUE together: prices.handler + sku + unit + country + effective_from. Also serves the price lookup.
db.prices.createIndex(
  { handler: 1, sku: 1, unit: 1, country: 1, effective_from: -1 },
  { name: "price_unique", unique: true },
);

// UNIQUE together: agent_storage.agent_id + kind + key.
db.agent_storage.createIndex({ agent_id: 1, kind: 1, key: 1 }, { name: "agent_kind_key_unique", unique: true });
db.agent_storage.createIndex({ project_id: 1 }, { name: "project_id" });

print("transactions, prices and agent_storage ready");
