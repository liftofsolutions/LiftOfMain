// Agent schema: agent_templates (with embedded nodes, edges and triggers), node_types, agents, credentials,
// agent_memory.
// Connected to the Project schema through project_id, and to the User schema through created_by.
// Keys: PK = primary key, FK = foreign key, UNIQUE = no two values share it in the stated scope.
//
// Rules (enforced by the backend on every write; clients never write these collections directly):
// - agent_templates.project_id missing = premade platform template; set = the project's own template.
// - agents.rag_id → a rag of the agent's project (Knowledge schema),
//   ready; required when rag_enabled is true. Each agent has its own, so
//   two agents of one template may search two rags.
// - node_types.project_id missing = platform node type; set = the project's own preset on an existing handler.
// - node.node_type_id → a platform node type, or one from the template's project.
// - node.credential_id → a credential from the template's project. Platform templates hold no credentials;
//   using a premade template copies it into the project first.
// - trigger.node_id → a node in the same template whose node type has category "io".
// - agents.template_id → a template from the agent's project.
// - node.id and trigger.id are UNIQUE within a template; node.name is a
//   label, and nodes may share one.
// - edges[] each lead from one node to one node of the same template, by
//   id, or from INPUT / to OUTPUT; no node has an id in KEPT_NODE_IDS. A
//   graph has one edge from INPUT and one or more to OUTPUT; nothing
//   leads to INPUT or leaves OUTPUT, and every other node has an edge out.
// - A node with config.rag_search runs only in an agent with a rag_id,
//   which the backend checks as it builds the agent.
// - A node with several edges out is a chat_model node without
//   rag_search. Its pick_limits are counts of 1 or more, by the id of a
//   node it has an edge to, and leave one of those unlimited.
//
// Runtime:
// - A node's effective config = mergeConfig(node_types.config, node.config).
// - The agent's graph is the nodes its edges name, wired as they say: a run
//   comes in at INPUT with its message and goes out at OUTPUT, one node at
//   a time, and each node reads the conversation so far and adds its
//   answer. A node with several edges out picks where the run goes on:
//   its model reads each node they lead to by name and config.description,
//   each as often as its config.pick_limits let it. The agent's answer is
//   the last reply of a chat_model node with one edge out, or the last
//   message if none answered, taken at OUTPUT; a run of more than 25 steps
//   is refused.
// - A chat_model node with config.rag_search searches its agent's rag.
//   A chat_history node keeps the conversation's chat history, in the
//   shape ChatHistory gives, summarised past summarise_after_tokens.
// - Webhook URL: /hooks/{projectId}/{agentId}/{triggerId}. Cron triggers: one scheduler job per trigger calls the same URL.
// - Inbound pipeline: read agent → read template → find the enabled trigger and its node (in memory)
//   → load the node's credential (same project) → verify the provider's signature → process.

// ---------- node_types ----------

export type NodeCategory = "chat_model" | "tool" | "io" | "memory";

// The backend integration that runs a node. Handlers are code, so this list grows with the codebase.
// Not limited to these.
export type Handler =
  | "openai_chat"
  | "anthropic_chat"
  | "gemini_chat"
  | "deepseek_chat"
  | "xai_chat"
  | "chat_history"
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
  id: string;             // UUID v4. UNIQUE within the template; edges name a node by it
  node_type_id: string;   // FK → node_types._id
  credential_id?: string; // FK → credentials._id, same project as the template
  name: string;           // a label, e.g. "reply", "web_search"; nodes may share one
  // Overrides node_types.config, e.g. { model, system_prompt } or
  // { cron, timezone }. Its description is what a node picking where a
  // run goes next reads of this one. A chat_model node may also have
  // rag_search: true, and pick_limits: { "<node id>": count }.
  config: Config;
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

// The two nodes every graph has, which edges name by these ids: where a
// run comes in, starting the conversation with the message a trigger or
// a query brought, and where it goes out, the last message its answer.
export const INPUT = "input";
export const OUTPUT = "output";
// Ids no node of a template may have: those, and LangGraph's own.
export const KEPT_NODE_IDS = [INPUT, OUTPUT, "__start__", "__end__"];

// Wires one node of a template to another, by id: which node runs after
// which, so where each node's output goes. { from: "<draft>", to:
// "<review>" } - <draft> standing for the id of a node named draft -
// runs review after draft. A node with several edges out picks one of
// the nodes they lead to as a run goes on, so an edge back makes a loop.
export interface Edge {
  from: string;
  to: string;
}

export interface AgentTemplate {
  _id: string;            // PK. UUID v4
  project_id?: string;    // FK → projects._id. Missing = premade platform template
  name: string;
  description?: string;
  nodes: AgentNode[];
  edges: Edge[];          // one edge from INPUT; one or more to OUTPUT
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
  rag_enabled?: boolean;  // its nodes with rag_search search rag_id
  rag_id?: string;        // FK → rags._id, same project; required when rag_enabled
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

// ---------- agent_memory ----------

// A conversation's chat history: whatever the agent's nodes keep of it.
// The document id is `${agent_id}_${convo_id}`, so a conversation is one
// agent's alone.
export interface AgentMemory {
  agent_id: string;       // FK → agents._id
  project_id: string;     // FK → projects._id
  convo_id: string;       // UUID: the conversation's id, the caller's pick
  chat_history: Record<string, unknown>; // the nodes' own, in any shape
  updated_at: Date;
}

// The chat_history a chat_history node keeps: the summary of the
// conversation before its messages, and those messages.
export interface ChatHistory {
  summary: string | null;
  messages: { role: "human" | "ai"; text: string }[];
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

  for (const node of template.nodes) {
    if (nodeIds.has(node.id)) errors.push(`node ${node.id}: duplicate id`);
    if (KEPT_NODE_IDS.includes(node.id)) {
      errors.push(`node ${node.id}: its id is one the graph keeps`);
    }
    nodeIds.add(node.id);

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

  if (template.edges.length === 0) errors.push("edges: none to build a graph of");
  // Where each node goes on to: the nodes its edges lead to.
  const nextNodes = new Map<string, Set<string>>();
  const reached = new Set<string>();
  template.edges.forEach((edge, index) => {
    const where = `edge ${index}`;
    if (edge.from === OUTPUT) errors.push(`${where}: leaves ${OUTPUT}`);
    if (edge.to === INPUT) errors.push(`${where}: leads to ${INPUT}`);
    for (const nodeId of [edge.from, edge.to]) {
      if (nodeId !== INPUT && nodeId !== OUTPUT && !nodeIds.has(nodeId)) {
        errors.push(`${where}: no node of id "${nodeId}" in this template`);
      }
    }
    if (!nextNodes.has(edge.from)) nextNodes.set(edge.from, new Set());
    nextNodes.get(edge.from)!.add(edge.to);
    reached.add(edge.to);
  });
  const firstNodes = nextNodes.get(INPUT)?.size ?? 0;
  if (firstNodes === 0) errors.push(`edges: none leaves ${INPUT}`);
  if (firstNodes > 1) {
    errors.push(`edges: ${firstNodes} leave ${INPUT}; a run comes in at one`);
  }
  if (!reached.has(OUTPUT)) errors.push(`edges: none reaches ${OUTPUT}`);
  for (const nodeId of reached) {
    if (nodeIds.has(nodeId) && !nextNodes.has(nodeId)) {
      errors.push(`node ${nodeId}: no edge out; every node leads on`);
    }
  }

  for (const node of template.nodes) {
    const config = node.config;
    const category = nodeTypes.get(node.node_type_id)?.category;
    const targets = nextNodes.get(node.id) ?? new Set<string>();
    if (targets.size > 1 && (category !== "chat_model" || config.rag_search)) {
      errors.push(`node ${node.name}: several edges out, but it cannot pick`);
    }
    const limits = config.pick_limits ?? {};
    if (!isPlainObject(limits)) {
      errors.push(`node ${node.name}: pick_limits are not by node id`);
      continue;
    }
    const limited = Object.keys(limits);
    for (const [nodeId, limit] of Object.entries(limits)) {
      if (!Number.isInteger(limit) || (limit as number) < 1) {
        errors.push(`node ${node.name}: pick limit of ${nodeId} is no count`);
      }
      if (!targets.has(nodeId)) {
        errors.push(`node ${node.name}: limits ${nodeId}, with no edge to it`);
      }
    }
    if (limited.length > 0 && [...targets].every((t) => limited.includes(t))) {
      errors.push(`node ${node.name}: pick_limits leave no way on`);
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
