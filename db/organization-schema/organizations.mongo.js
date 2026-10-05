// organizations and invitations collections for MongoDB: validators + indexes.
// plan_id points to plans in the Billing schema; that collection is set up there.
// Run with mongosh:  mongosh "<connection string>/<db>" organizations.mongo.js
// Safe to re-run. Extra fields are allowed, so the schema stays flexible.

const uuid = {
  bsonType: "string",
  pattern: "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
};
const timestamps = {
  created_at: { bsonType: "date" },
  updated_at: { bsonType: "date" },
};
const role = { enum: ["owner", "admin", "member"] };

const organizationsValidator = {
  $jsonSchema: {
    bsonType: "object",
    required: ["_id", "name", "slug", "plan_id", "status", "created_by", "created_at", "updated_at"],
    properties: {
      _id: uuid,
      name: { bsonType: "string", minLength: 1, maxLength: 120 },
      slug: { bsonType: "string", pattern: "^[a-z0-9]+(-[a-z0-9]+)*$", minLength: 2, maxLength: 48 },
      logo_url: { bsonType: "string" },
      plan_id: uuid, // FK → plans._id (Billing schema)
      status: { enum: ["active", "suspended", "deleted"] },
      created_by: uuid,
      ...timestamps,
      deleted_at: { bsonType: "date" },
      settings: {
        bsonType: "object",
        required: ["default_role", "invite_only", "allowed_providers", "created_at", "updated_at"],
        properties: {
          default_role: { enum: ["admin", "member"] },
          invite_only: { bsonType: "bool" },
          allowed_providers: {
            bsonType: "array",
            minItems: 1,
            items: { enum: ["google", "email", "github"] },
          },
          ...timestamps,
        },
      },
    },
  },
};

const invitationsValidator = {
  $jsonSchema: {
    bsonType: "object",
    required: [
      "_id", "org_id", "email", "role", "token_hash", "invited_by",
      "status", "expires_at", "created_at", "updated_at",
    ],
    properties: {
      _id: uuid,
      org_id: uuid,
      email: { bsonType: "string", pattern: "^[^@\\sA-Z]+@[^@\\sA-Z]+$" },
      role,
      token_hash: { bsonType: "string", pattern: "^[0-9a-f]{64}$" },
      invited_by: uuid,
      status: { enum: ["pending", "accepted", "revoked", "expired"] },
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

upsertCollection("organizations", organizationsValidator);
upsertCollection("invitations", invitationsValidator);

// UNIQUE: organizations.slug (used in URLs).
db.organizations.createIndex({ slug: 1 }, { name: "slug_unique", unique: true });

// organizations: find every org on a plan (e.g. before retiring it).
db.organizations.createIndex({ plan_id: 1 }, { name: "plan_id" });

// UNIQUE together: invitations.org_id + email, only while status is "pending".
db.invitations.createIndex(
  { org_id: 1, email: 1 },
  { name: "pending_invite_unique", unique: true, partialFilterExpression: { status: "pending" } },
);

// UNIQUE: invitations.token_hash. Finds the invite when someone opens the link.
db.invitations.createIndex({ token_hash: 1 }, { name: "token_hash_unique", unique: true });

// invitations: delete automatically 30 days after they expire.
db.invitations.createIndex(
  { expires_at: 1 },
  { name: "expires_ttl", expireAfterSeconds: 60 * 60 * 24 * 30 },
);

// users: list every member of an org (membership lives inside users).
db.users.createIndex({ "membership.org_id": 1 }, { name: "membership_org" });

print("organizations and invitations ready");
