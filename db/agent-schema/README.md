# Agent schema

Agents for a project: what they are made of (nodes), how those are wired (edges), what starts them (triggers), and
the keys they use (credentials).
MongoDB-style documents; works on Firestore today.

## Files

| File | What it is |
|---|---|
| `agents-erd.svg`, `agents-erd.png` | Diagram with relationships and cardinality |
| `agents.types.ts` | TypeScript types, config merge, webhook path, `validateTemplate()` |
| `agents.mongo.js` | MongoDB validators and indexes (run once with mongosh) |
| `agents.example.json` | A complete example: node types, a credential, a template whose graph picks and loops, an agent, its webhook URLs |

## Collections

| Collection | Owner | Holds |
|---|---|---|
| `node_types` | Platform, or one project | Reusable node templates: category, handler, default config |
| `agent_templates` | Platform, or one project | An agent's structure: embedded `nodes[]`, `edges[]` and `triggers[]`; `rag_enabled`, with the `rag_id` of its knowledge base |
| `agents` | One project | The running agent: which template, name, status |
| `credentials` | One project | Keys and account details a node uses; secrets encrypted |

Embedded inside `agent_templates`:

- **node**: an instance of a node type, with its own `config` and an optional `credential_id`.
- **edge**: wires one node to another by id - which node runs after which, so where each node's output goes.
  See **Graph** under Runtime.
- **trigger**: points to an `io` node and names the `event` that starts the agent.

## Relationships

```
projects        1 ── 0..N  agents
projects     0..1 ── 0..N  agent_templates     (missing project_id = premade platform template)
projects     0..1 ── 0..N  node_types          (missing project_id = platform node type)
projects        1 ── 0..N  credentials
rags         0..1 ── 0..N  agent_templates     (rag_id, set when rag_enabled; Knowledge schema)
agent_templates 1 ── 0..N  agents
agent_templates 1 ── N     nodes      (embedded)
agent_templates 1 ── 1..N  edges      (embedded)
agent_templates 1 ── 0..N  triggers   (embedded)
node_types      1 ── 0..N  nodes
credentials  0..1 ── 0..N  nodes
nodes           1 ── 0..N  edges      (by id)
nodes           1 ── 0..N  triggers
```

## Node categories

| category | Examples (`handler`) | Role |
|---|---|---|
| `chat_model` | `anthropic_chat`, `openai_chat`, `gemini_chat` | The model; `config` holds model, temperature, `system_prompt` |
| `tool` | `tavily_search`, `http_api` | Called by the model while processing |
| `io` | `whatsapp`, `email`, `line`, `cron` | Receives events (via triggers) and sends messages |

`handler` is the key into the backend's integration code. A project's own node type is a preset on an existing
handler, e.g. "Acme CRM lookup" = `handler: "http_api"` with the URL in its default `config`.

Only the nodes a template's edges name run in the agent's graph, and today those are chat models: `openai_chat`,
`gemini_chat`, `deepseek_chat` and `xai_chat` have node classes. `io` nodes stay out of the edges, as they receive
events and send messages on their own channels. Tool nodes join a graph once they can run.

## Rules (checked by the backend before saving)

Clients never write these collections directly; all writes go through the backend API.

1. A node's type is a platform node type or one from the template's project.
2. A node's credential belongs to the template's project. Platform templates hold no credentials; using a
   premade template copies it into the project first.
3. A trigger points to a node in the same template, and that node's type has category `io`.
4. An agent's template belongs to the agent's project.
5. Node ids and trigger ids are unique within a template. Node names are labels, and nodes may share one.
6. Each edge leads from one node to one node of the same template, by id, or from or to the two nodes every
   graph has, `input` and `output`; no node has either id, nor LangGraph's `__start__` / `__end__`. A graph has
   one edge from `input` and one or more to `output`; nothing leads to `input` or leaves `output`, and every
   other node has an edge out.
7. A template with `rag_enabled` names its knowledge base in `rag_id`: a rag of the template's project, from
   the Knowledge schema.

`validateTemplate()` in `agents.types.ts` implements rules 1–3, 5 and 6. The backend checks rules 5 and 6 again
as it builds an agent, and refuses one that breaks them. `agents.mongo.js` holds rule 7's part a validator can:
`rag_id` is required when `rag_enabled` is true.

## Runtime

**Config.** A node's effective config = `mergeConfig(node_types.config, node.config)`.
Objects merge deeply, the node wins, arrays are replaced.

**Graph.** `edges` wire a template's nodes by id, as ids are unique within a template and names are labels that
nodes may share. LangGraph builds graphs in code, not from JSON, so this is the template's own format. Each edge
leads from one node to one node: `{"from": "<draft>", "to": "<review>"}` runs `review` after `draft`, where
`<draft>` stands for the id of a node named draft, and so on.

Two nodes are in every graph, built by the backend rather than stored, and edges name them by their ids: a run
comes in at `input`, which starts the conversation with the message a trigger or a query brought, and goes out
at `output`, which takes the last message as the agent's answer. Each node between reads the whole conversation
so far and adds its answer.

A run goes on one node at a time. A node with one edge out goes on to the node it leads to; a node with several
picks one of the nodes they lead to. Its model answers as structured output: its answer, and the id of the node
it picks, which can be none but theirs. It reads each of them by its name and its config's `description`, and
`output` as the end of the run, its own answer going out as the agent's. So an if-else is a node with an edge
each way, and a loop an edge back; here `review` picks `draft`, to write it again, or `output`:

```json
"edges": [
  {"from": "input", "to": "<draft>"},
  {"from": "<draft>", "to": "<review>"},
  {"from": "<review>", "to": "<draft>"},
  {"from": "<review>", "to": "output"}
]
```

A run of more than 25 steps, a node run each, is refused: the bound on a loop whose node never picks the way out.

`agents.example.json` has a fuller graph: `classify` picks `facts` for a sales question, `apology` for a complaint
or `handoff` for anything else; a sale goes on through `draft` and `review`, which picks `draft` again or `reply`.

**Triggers.** A trigger can start an agent when its template has it, `enabled`, on one of its nodes; the backend
asks this of a built agent with `isTriggerAvailable(triggerId)`.

The backend builds an agent's graph once, on its first query: it reserves the nodes the edges name, configured
with their merged config, compiles them into a LangGraph graph, and keeps it for the next queries.

**Query.** Anyone may ask an agent: `POST /agents/{projectId}/{agentId}/query` with `{"message": "..."}`. The
agent must belong to `projectId`, be `active` and not deleted, and its template must build; nothing else is
checked.

**Inbound webhook.** Each trigger has its own URL:

```
POST /hooks/{projectId}/{agentId}/{triggerId}
```

1. Read the agent by id; it must belong to `projectId`, be `active` and not deleted.
2. Read its template (cache by id).
3. Find the trigger by id (enabled) and its node, in memory.
4. Load the node's credential (same project) and verify the provider's signature
   (`X-Hub-Signature-256` for WhatsApp, `X-Line-Signature` for LINE, …).
5. Acknowledge with 200 and process.

The URL is not a secret; the signature check is the authentication. WhatsApp supports a callback URL per phone
number through webhook overrides; LINE and email routes are configured per channel or address.

**Cron.** One scheduler job per `schedule` trigger calls the same URL on the node's `config.cron`.

**Credentials.** `values` hold non-secret account details (phone number id, username, callback id).
`secret_values` hold tokens, passwords and API keys, encrypted with AES-256-GCM; the data key is wrapped by a
KMS key. Only the backend decrypts them, at the moment a node needs them. They are never sent to clients.

## Left to later modules

- **Runs and conversations**: run history, messages, which node used how many credits.
  A template `version` field can be added then if runs need to record which version produced a reply.
- **Loop counters**: how many times a node may run in one run, kept in the node's config.
- **Knowledge (RAG)**: how an agent searches the rag its template's `rag_id` names once `rag_enabled` is on.
  For now `rag_id` is data only; nothing reads it yet.
- **Billing**: credit prices per model.
