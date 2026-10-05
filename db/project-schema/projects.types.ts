// projects, project_members and project_roles collections.
// Separate from the User schema (users._id) and the Organization schema (organizations._id).
// Keys: PK = primary key, FK = foreign key, UNIQUE = no two documents share the value.
//
// Rules enforced in code:
// - Personal project: the owner plus at most PERSONAL_SHARE_LIMIT other people.
// - Org project: members must belong to that org (users.membership.org_id === project.org_id).
// - Only existing accounts can be added; there are no project invites.
// - A member's role_id must be a role of the same project.
// - A change is rejected if it would leave no member whose role has "manage".

// ---------- projects ----------

export type ProjectScope = "personal" | "org";
export type ProjectStatus = "active" | "archived";

export const PERSONAL_SHARE_LIMIT = 2;

export interface Project {
  _id: string;            // PK. UUID v4
  name: string;
  description?: string;
  scope: ProjectScope;    // "personal" = owned by one user; "org" = owned by an organization
  org_id?: string;        // FK → organizations._id. Present only when scope is "org"
  status: ProjectStatus;
  member_count: number;   // updated in the same write as project_members; enforces the personal limit
  created_by: string;     // FK → users._id
  created_at: Date;
  updated_at: Date;
  deleted_at?: Date;
}

// ---------- project_roles ----------

// Permissions a role can grant.
// Not limited to these: add more as features arrive (e.g. for agents).
export type Permission =
  | "view"    // open the project
  | "edit"    // change the project
  | "manage"; // add and remove members, create and edit roles

export interface ProjectRole {
  _id: string;            // PK. UUID v4
  project_id: string;     // FK → projects._id. Each project owns and manages its own roles
  name: string;           // UNIQUE together with project_id
  permissions: Permission[];
  created_at: Date;
  updated_at: Date;
}

// Created with every new project; members with "manage" can change them afterwards.
export const DEFAULT_ROLES: ReadonlyArray<{ name: string; permissions: Permission[] }> = [
  { name: "Owner", permissions: ["view", "edit", "manage"] },
  { name: "Editor", permissions: ["view", "edit"] },
  { name: "Viewer", permissions: ["view"] },
];

// ---------- project_members ----------

export interface ProjectMember {
  _id: string;            // PK, composite: "{project_id}_{user_id}" (one entry per user per project)
  project_id: string;     // FK → projects._id
  user_id: string;        // FK → users._id
  role_id: string;        // FK → project_roles._id
  added_by: string;       // FK → users._id
  created_at: Date;
  updated_at: Date;
}

export function memberId(project_id: string, user_id: string): string {
  return `${project_id}_${user_id}`;
}
