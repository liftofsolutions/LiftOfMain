// transactions, prices and agent_storage collections.
// Connected to the Project schema (project_id) and the Agent schema (agent_id, template trigger and node ids).
// Keys: PK = primary key, FK = foreign key, UNIQUE = no two documents share the value in the stated scope.
// Money: integer micro-USD everywhere. 1 USD = 1,000,000 micros.

export type Json = Record<string, unknown>;

// ---------- transactions ----------
// One document per agent execution: a trigger fired, the agent ran, it replied.

export type TransactionStatus = "running" | "succeeded" | "failed";

// One priced line for a node. An LLM call usually has two lines (input tokens, output tokens),
// a WhatsApp send one line (message). A node's cost in a step = sum of its lines.
export interface NodeCost {
  node_id: string;       // → agent_templates.nodes[].id (Agent schema)
  node_name: string;     // copied from the template, so history survives template edits
  price_id: string;      // FK → prices._id
  usage: Json;           // raw usage, e.g. { input_tokens: 1200 } or { messages: 1, category: "service", country: "IN" }
  cost_micros: number;
}

export interface StepCost {
  step: number;          // 1, 2, 3 … in execution order
  total_micros: number;  // sum of nodes[].cost_micros
  nodes: NodeCost[];
}

export interface Transaction {
  _id: string;           // PK. UUID v4
  project_id: string;    // FK → projects._id
  agent_id: string;      // FK → agents._id
  trigger_id: string;    // → agent_templates.triggers[].id (Agent schema)
  input: Json;           // raw incoming JSON, stored as-is (e.g. the webhook body)
  output?: Json;         // raw outgoing JSON, stored as-is (e.g. the reply and the provider's send response)
  status: TransactionStatus;
  error?: string;
  total_micros: number;  // sum of steps[].total_micros
  steps: StepCost[];
  finished_at?: Date;
  created_at: Date;
  updated_at: Date;
}

// ---------- prices ----------
// What one unit costs us. UNIQUE together: handler + sku + unit + country + effective_from.

export interface Price {
  _id: string;           // PK. UUID v4
  provider: string;      // e.g. "anthropic", "openai", "meta", "tavily"
  handler: string;       // matches node_types.handler, e.g. "anthropic_chat", "whatsapp"
  sku: string;           // model id, message category, search tier, …
  unit: string;          // e.g. "input_token", "output_token", "message", "search"
  country?: string;      // ISO 3166-1 alpha-2, e.g. "IN". Missing = applies to every country
  price_micros: number;  // cost of per_units units
  per_units: number;     // e.g. 1_000_000 for token prices, 1 for messages
  effective_from: Date;
  effective_to?: Date;   // missing = still current
  created_at: Date;
  updated_at: Date;
}

// ---------- agent_storage ----------
// Unstructured data an agent keeps, e.g. conversation history. Not linked to transactions.
// UNIQUE together: agent_id + kind + key.

export interface AgentStorage {
  _id: string;           // PK. UUID v4
  project_id: string;    // FK → projects._id
  agent_id: string;      // FK → agents._id
  kind: string;          // what this is, e.g. "conversation", "lead"
  key: string;           // which one, e.g. "whatsapp:+919812345678"
  data: Json;            // anything, e.g. { messages: [{ type: "human", … }, { type: "ai", … }] }
  created_at: Date;
  updated_at: Date;
}

// ---------- Helpers ----------

// Active price for a unit: exact country first, then the row without a country.
export function findPrice(
  prices: Price[],
  q: { handler: string; sku: string; unit: string; country?: string; at: Date },
): Price | undefined {
  const active = prices.filter(
    (p) =>
      p.handler === q.handler &&
      p.sku === q.sku &&
      p.unit === q.unit &&
      p.effective_from <= q.at &&
      (!p.effective_to || q.at < p.effective_to),
  );
  return (q.country && active.find((p) => p.country === q.country)) || active.find((p) => !p.country);
}

// Cost of a quantity at a price, rounded to the nearest micro.
export function lineCost(quantity: number, price: Price): number {
  return Math.round((quantity * price.price_micros) / price.per_units);
}

export function stepTotal(nodes: NodeCost[]): number {
  return nodes.reduce((sum, n) => sum + n.cost_micros, 0);
}

export function transactionTotal(steps: StepCost[]): number {
  return steps.reduce((sum, s) => sum + s.total_micros, 0);
}

export const microsToUsd = (micros: number): number => micros / 1_000_000;
