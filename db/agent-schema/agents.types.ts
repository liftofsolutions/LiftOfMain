// Agent schema: agent_templates (with embedded nodes and triggers), node_types, agents, credentials.
// Connected to the Project schema through project_id, and to the User schema through created_by.
// Keys: PK = primary key, FK = foreign key, UNIQUE = no two values share it in the stated scope.
//
// Rules (enforced by the backend on every write; clients never write these collections directly):
// - agent_templates.project_id missing = premade platform template; set = the project's own template.
// - node_types.project_id missing = platform node type; set = the project's own preset on an existing handler.
// - node.node_type_id → a platform node type, or one from the template's project.
// - node.credential_id → a credential from the template's project. Platform templates hold no credentials;
//   using a premade template copies it into the project first.
// - trigger.node_id → a node in the same template whose node type has category "io".
// - agents.template_id → a template from the agent's project.
// - node.id, node.name and trigger.id are UNIQUE within a template.
//
// Runtime:
// - A node's effective config = mergeConfig(node_types.config, node.config).
// - Webhook URL: /hooks/{projectId}/{agentId}/{triggerId}. Cron triggers: one scheduler job per trigger calls the same URL.
// - Inbound pipeline: read agent → read template → find the enabled trigger and its node (in memory)
//   → load the node's credential (same project) → verify the provider's signature → process.

// ---------- node_types ----------

export type NodeCategory = "chat_model" | "tool" | "io";

// The backend integration that runs a node. Handlers are code, so this list grows with the codebase.
// Not limited to these.
export type Handler =
  | "openai_chat"
  | "anthropic_chat"
  | "gemini_chat"
  | "tavily_search"
  | "http_api"
  | "whatsapp"
  | "email"
  | "line"
  | "cron";

export type Config = Record<string, unknown>;

export interface NodeType {
  _id: string;            // PK. UUID v4
  project_id?: string;    // FK → projects._id. Missing = platform node type
  category: NodeCategory;
  handler: Handler;       // which code runs it, e.g. "whatsapp", "anthropic_chat", "http_api"
  name: string;           // e.g. "WhatsApp", "Anthropic chat", "Acme CRM lookup"
  config: Config;         // default config; a node's own config overrides it
  is_active: boolean;     // false = no new nodes of this type; existing nodes stop running
  created_at: Date;
  updated_at: Date;
}

// ---------- agent_templates ----------

// A node is an instance of a node type inside one template.
export interface AgentNode {
  id: string;             // UUID v4. UNIQUE within the template
  node_type_id: string;   // FK → node_types._id
  credential_id?: string; // FK → credentials._id, same project as the template
  name: string;           // UNIQUE within the template, e.g. "reply", "web_search", "whatsapp"
  config: Config;         // overrides node_types.config, e.g. { model, system_prompt } or { cron, timezone }
  created_at: Date;
  updated_at: Date;
}

// What starts the agent. Points to an "io" node in the same template.
export interface Trigger {
  id: string;             // UUID v4. UNIQUE within the template; used in the webhook URL
  node_id: string;        // FK → nodes[].id in the same template
  event: string;          // which event of that node starts the agent. Not limited to:
                          // "message.received" | "message.status" | "email.received" | "postback" | "schedule"
  enabled: boolean;
  created_at: Date;
  updated_at: Date;
}

export interface AgentTemplate {
  _id: string;            // PK. UUID v4
  project_id?: string;    // FK → projects._id. Missing = premade platform template
  name: string;
  description?: string;
  rag_enabled: boolean;   // use the project's knowledge base (Knowledge module)
  nodes: AgentNode[];
  triggers: Trigger[];
  is_active: boolean;
  created_at: Date;
  updated_at: Date;
}

// ---------- agents ----------

export type AgentStatus = "active" | "paused";

export interface Agent {
  _id: string;            // PK. UUID v4
  project_id: string;     // FK → projects._id
  template_id: string;    // FK → agent_templates._id, same project
  name: string;
  status: AgentStatus;
  created_by: string;     // FK → users._id
  created_at: Date;
  updated_at: Date;
  deleted_at?: Date;
}

// ---------- credentials ----------

// AES-256-GCM ciphertext; the data key is wrapped by a KMS key. Only the backend decrypts it.
export interface EncryptedBlob {
  ciphertext: string;     // base64
  iv: string;             // base64
  tag: string;            // base64
  wrapped_dek: string;    // base64, data key encrypted by KMS
  kms_key_id: string;     // which KMS key wrapped it (for rotation)
}

export interface Credential {
  _id: string;                    // PK. UUID v4
  project_id: string;             // FK → projects._id
  name: string;                   // e.g. "Acme WhatsApp"
  values?: Record<string, string>; // not secret: phone_number_id, username, callback id, …
  secret_values: EncryptedBlob;   // secret: tokens, passwords, API keys
  expires_at?: Date;              // for tokens that expire (OAuth)
  created_by: string;             // FK → users._id
  created_at: Date;
  updated_at: Date;
}

// ---------- Runtime helpers ----------

function isPlainObject(v: unknown): v is Config {
  return typeof v === "object" && v !== null && !Array.isArray(v) && !(v instanceof Date);
}

// Objects merge deeply, everything else (including arrays) is replaced, undefined is skipped.
export function mergeConfig(base: Config, override: Config): Config {
  const out: Config = { ...base };
  for (const [key, value] of Object.entries(override)) {
    if (value === undefined) continue;
    const current = out[key];
    out[key] = isPlainObject(value) && isPlainObject(current) ? mergeConfig(current, value) : value;
  }
  return out;
}

export function resolveNodeConfig(type: NodeType, node: AgentNode): Config {
  return mergeConfig(type.config, node.config);
}

export function webhookPath(projectId: string, agentId: string, triggerId: string): string {
  return `/hooks/${projectId}/${agentId}/${triggerId}`;
}

// Checks the rules above before a template is saved. Returns a list of problems (empty = valid).
export function validateTemplate(
  template: AgentTemplate,
  nodeTypes: Map<string, NodeType>,
  credentials: Map<string, Credential>,
): string[] {
  const errors: string[] = [];
  const nodeIds = new Set<string>();
  const nodeNames = new Set<string>();

  for (const node of template.nodes) {
    if (nodeIds.has(node.id)) errors.push(`node ${node.id}: duplicate id`);
    if (nodeNames.has(node.name)) errors.push(`node ${node.id}: duplicate name "${node.name}"`);
    nodeIds.add(node.id);
    nodeNames.add(node.name);

    const type = nodeTypes.get(node.node_type_id);
    if (!type) {
      errors.push(`node ${node.name}: unknown node type`);
    } else {
      if (!type.is_active) errors.push(`node ${node.name}: node type is inactive`);
      if (type.project_id && type.project_id !== template.project_id) {
        errors.push(`node ${node.name}: node type belongs to another project`);
      }
    }

    if (node.credential_id) {
      const credential = credentials.get(node.credential_id);
      if (!template.project_id) errors.push(`node ${node.name}: platform templates cannot hold credentials`);
      else if (!credential) errors.push(`node ${node.name}: unknown credential`);
      else if (credential.project_id !== template.project_id) {
        errors.push(`node ${node.name}: credential belongs to another project`);
      }
    }
  }

  const triggerIds = new Set<string>();
  for (const trigger of template.triggers) {
    if (triggerIds.has(trigger.id)) errors.push(`trigger ${trigger.id}: duplicate id`);
    triggerIds.add(trigger.id);

    const node = template.nodes.find((n) => n.id === trigger.node_id);
    if (!node) {
      errors.push(`trigger ${trigger.id}: node not found in this template`);
      continue;
    }
    if (nodeTypes.get(node.node_type_id)?.category !== "io") {
      errors.push(`trigger ${trigger.id}: node "${node.name}" is not an io node`);
    }
  }

  return errors;
}
