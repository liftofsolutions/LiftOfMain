// users collection for MongoDB: validator + indexes.
// Run with mongosh:  mongosh "<connection string>/<db>" users.mongo.js
// Safe to re-run. Extra fields are allowed, so the schema stays flexible.

const timestamps = {
  created_at: { bsonType: "date" },
  updated_at: { bsonType: "date" },
};

const validator = {
  $jsonSchema: {
    bsonType: "object",
    required: ["_id", "display_name", "identities", "created_at", "updated_at"],
    properties: {
      _id: {
        bsonType: "string",
        pattern: "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
      },
      display_name: { bsonType: "string", minLength: 1, maxLength: 100 },
      email: { bsonType: "string", pattern: "^[^@\\sA-Z]+@[^@\\sA-Z]+$" },
      avatar_url: { bsonType: "string" },

      identities: {
        bsonType: "array",
        minItems: 1,
        items: {
          bsonType: "object",
          required: ["provider", "provider_uid", "created_at", "updated_at"],
          properties: {
            provider: { enum: ["google", "email", "github"] },
            provider_uid: { bsonType: "string", minLength: 1 },
            ...timestamps,
          },
        },
      },

      profile: {
        bsonType: "object",
        required: ["created_at", "updated_at"],
        properties: {
          dob: { bsonType: "string", pattern: "^\\d{4}-\\d{2}-\\d{2}$" },
          gender: { bsonType: "string" },
          ...timestamps,
        },
      },

      settings: {
        bsonType: "object",
        required: ["schema_version", "created_at", "updated_at"],
        properties: {
          schema_version: { bsonType: "int", minimum: 1 },
          theme: { enum: ["light", "dark", "system"] },
          language: { bsonType: "string" },
          timezone: { bsonType: "string" },
          notifications: {
            bsonType: "object",
            properties: {
              email: { bsonType: "bool" },
              push: { bsonType: "bool" },
            },
          },
          ...timestamps,
        },
      },

      ...timestamps,
    },
  },
};

const options = { validator, validationLevel: "strict", validationAction: "error" };

if (db.getCollectionNames().includes("users")) {
  db.runCommand({ collMod: "users", ...options });
} else {
  db.createCollection("users", options);
}

// One account per email (only where an email exists).
db.users.createIndex(
  { email: 1 },
  { name: "email_unique", unique: true, partialFilterExpression: { email: { $type: "string" } } },
);

// One user per auth uid. Used to find the user on sign-in.
db.users.createIndex(
  { "identities.provider_uid": 1 },
  { name: "identity_uid_unique", unique: true },
);

print("users collection ready");
