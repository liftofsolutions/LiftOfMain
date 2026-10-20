// organizations and invitations collections, and the org field they add to users.
// Separate from the User schema (connected through users.membership)
// and from the Billing schema (connected through organizations.plan_id).
// Keys: PK = primary key, FK = foreign key, UNIQUE = no two documents share the value.

import type { AuthProvider, User } from "./users.types";

// ---------- organizations ----------

export type OrgStatus = "active" | "suspended" | "deleted";
export type OrgRole = "owner" | "admin" | "member";

export interface OrgSettings {
  default_role: Exclude<OrgRole, "owner">; // role given to new members
  invite_only: boolean;                    // true = people join only by invitation
  allowed_providers: AuthProvider[];       // sign-in methods members may use
  created_at: Date;
  updated_at: Date;
}

export interface Organization {
  _id: string;          // PK. UUID v4
  name: string;         // company name
  slug: string;         // UNIQUE. lowercase, used in URLs, e.g. "acme-robotics"
  logo_url?: string;
  plan_id: string;      // FK → plans._id (Billing schema)
  status: OrgStatus;
  created_by: string;   // FK → users._id
  created_at: Date;
  updated_at: Date;
  deleted_at?: Date;    // set with status "deleted"; purge after a grace period
  settings?: OrgSettings;
}

// ---------- invitations ----------
// UNIQUE together: org_id + email, while status is "pending"
// (one open invite per person per organization).

export type InvitationStatus = "pending" | "accepted" | "revoked" | "expired";

export interface Invitation {
  _id: string;          // PK. UUID v4
  org_id: string;       // FK → organizations._id
  email: string;        // lowercase; must match the signed-in user's email on accept
  role: OrgRole;        // copied into users.membership.role on accept
  token_hash: string;   // UNIQUE. SHA-256 hex of the invite token; the token itself is never stored
  invited_by: string;   // FK → users._id
  status: InvitationStatus;
  expires_at: Date;
  created_at: Date;
  updated_at: Date;
}

// ---------- Field added to the users document ----------

// A user belongs to one organization or none.
export interface Membership {
  org_id: string;       // FK → organizations._id
  role: OrgRole;
  created_at: Date;
  updated_at: Date;
}

export interface UserOrgFields {
  membership?: Membership; // missing = personal user
}

export type UserWithOrg = User & UserOrgFields;
