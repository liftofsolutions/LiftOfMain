// projects, project_members and project_roles collections for MongoDB: validators + indexes.
// Run with mongosh:  mongosh "<connection string>/<db>" projects.mongo.js
// Safe to re-run. Extra fields are allowed, so the schema stays flexible.

const UUID = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}";
const uuid = { bsonType: "string", pattern: `^${UUID}$` };
const timestamps = {
  created_at: { bsonType: "date" },
  updated_at: { bsonType: "date" },
};

// Not limited to these: add more as features arrive (e.g. for agents).
const PERMISSIONS = ["view", "edit", "manage"];

const projectsValidator = {
  $jsonSchema: {
    bsonType: "object",
    required: ["_id", "name", "scope", "status", "member_count", "created_by", "created_at", "updated_at"],
    properties: {
      _id: uuid,
      name: { bsonType: "string", minLength: 1, maxLength: 120 },
      description: { bsonType: "string", maxLength: 2000 },
      scope: { enum: ["personal", "org"] },
      org_id: uuid, // FK → organizations._id
      status: { enum: ["active", "archived"] },
      member_count: { bsonType: "int", minimum: 1 },
      created_by: uuid, // FK → users._id
      ...timestamps,
      deleted_at: { bsonType: "date" },
    },
    // org_id is required for org projects and not allowed on personal ones.
    oneOf: [
      { properties: { scope: { enum: ["personal"] } }, not: { required: ["org_id"] } },
      { properties: { scope: { enum: ["org"] } }, required: ["org_id"] },
    ],
  },
};

const rolesValidator = {
  $jsonSchema: {
    bsonType: "object",
    required: ["_id", "project_id", "name", "permissions", "created_at", "updated_at"],
    properties: {
      _id: uuid,
      project_id: uuid, // FK → projects._id
      name: { bsonType: "string", minLength: 1, maxLength: 60 },
      permissions: {
        bsonType: "array",
        minItems: 1,
        uniqueItems: true,
        items: { enum: PERMISSIONS },
      },
      ...timestamps,
    },
  },
};

const membersValidator = {
  $jsonSchema: {
    bsonType: "object",
    required: ["_id", "project_id", "user_id", "role_id", "added_by", "created_at", "updated_at"],
    properties: {
      // PK, composite: "{project_id}_{user_id}"
      _id: { bsonType: "string", pattern: `^${UUID}_${UUID}$` },
      project_id: uuid, // FK → projects._id
      user_id: uuid,    // FK → users._id
      role_id: uuid,    // FK → project_roles._id
      added_by: uuid,   // FK → users._id
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

upsertCollection("projects", projectsValidator);
upsertCollection("project_roles", rolesValidator);
upsertCollection("project_members", membersValidator);

// projects: list an organization's projects.
db.projects.createIndex({ org_id: 1 }, { name: "org_id" });

// UNIQUE together: project_roles.project_id + name.
db.project_roles.createIndex({ project_id: 1, name: 1 }, { name: "project_role_name_unique", unique: true });

// project_members: _id is the composite key, so a user can't be added to a project twice.
// "My projects" for a user.
db.project_members.createIndex({ user_id: 1 }, { name: "user_id" });
// Everyone in a project.
db.project_members.createIndex({ project_id: 1 }, { name: "project_id" });
// Check whether a role is still in use before deleting it.
db.project_members.createIndex({ role_id: 1 }, { name: "role_id" });

print("projects, project_roles and project_members ready");
