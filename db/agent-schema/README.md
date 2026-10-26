# Agent schema

Agents for a project: what they are made of (nodes), how those are wired (edges), what starts them (triggers),
the keys they use (credentials), and what they remember of each conversation (chat history).
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
| `agent_templates` | Platform, or one project | An agent's structure: embedded `nodes[]`, `edges[]` and `triggers[]` |
| `agents` | One project | The running agent: which template, name, status; `rag_enabled`, with the `rag_id` of its knowledge base |
| `credentials` | One project | Keys and account details a node uses; secrets encrypted |
| `agent_memory` | One agent | A conversation's chat history: whatever the agent's nodes keep of it, as a plain object |

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
rags         0..1 ── 0..N  agents              (rag_id, set when rag_enabled; Knowledge schema)
agent_templates 1 ── 0..N  agents
agents          1 ── 0..N  agent_memory        (one per conversation)
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
| `memory` | `chat_history` | Keeps the conversation's chat history, summarised once it grows long |

`handler` is the key into the backend's integration code. A project's own node type is a preset on an existing
handler, e.g. "Acme CRM lookup" = `handler: "http_api"` with the URL in its default `config`.

Only the nodes a template's edges name run in the agent's graph, and today those are chat models -
`openai_chat`, `gemini_chat`, `deepseek_chat` and `xai_chat` have node classes - and `chat_history`. `io` nodes
stay out of the edges, as they receive events and send messages on their own channels. Tool nodes join a graph
once they can run; a chat model searches the knowledge base itself, with `rag_search`.

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
7. An agent with `rag_enabled` names its knowledge base in `rag_id`: a ready rag of the agent's project, from
   the Knowledge schema. Each agent has its own, so two agents of one template may search two rags. A node with
   `rag_search` runs only in an agent with one.
8. A node with several edges out is a chat model, which picks, and does not search: no `rag_search`. A node's
   `pick_limits` are counts of 1 or more, by the id of a node it has an edge to, and leave one of those
   unlimited, so a run always has a way on.

`validateTemplate()` in `agents.types.ts` implements rules 1–3, 5, 6 and 8. The backend checks rules 5–8 again
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
at `output` with the agent's answer: the last reply of a node that answers - a chat model with one edge out -
not of one that picks, so a reviewer passing a reply is not the one heard. A run in which none answered, a guard
turning a message away, goes out with its last message. Each node between reads the whole conversation so far
and adds its answer.

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

A node's `pick_limits` cap how often it picks each node in a run: `"pick_limits": {"<draft>": 3}` lets `review`
send a run back to `draft` three times. It counts its picks in its `nodeDetails`, `{"picks": {"<draft>": 2}}`,
offers its model only the nodes it may still pick, and with one left, sends the run there without asking.

A run of more than 25 steps, a node run each, is refused: the bound on a loop whose node never picks the way out.

A customer support agent, its message from a web page coming in as it is:

```json
"edges": [
  {"from": "input", "to": "<guard>"},
  {"from": "<guard>", "to": "<support>"},
  {"from": "<guard>", "to": "output"},
  {"from": "<support>", "to": "<review>"},
  {"from": "<review>", "to": "<support>"},
  {"from": "<review>", "to": "<history>"},
  {"from": "<history>", "to": "output"}
]
```

`guard` turns a message away, going out with its reply, or passes it to `support`, which searches the knowledge
base (`rag_search`) and answers. `review`, with `"pick_limits": {"<support>": 3}`, passes the reply or sends it
back with what to change, three times at most; `history` keeps the turn, and the answer going out is `support`'s.
`Btest/agents/CustomerSupportAgentTest.py` runs it.

`agents.example.json` has a fuller graph: `classify` picks `facts` for a sales question, `apology` for a complaint
or `handoff` for anything else; a sale goes on through `draft` and `review`, which picks `draft` again or `reply`.

**Triggers.** A trigger can start an agent when its template has it, `enabled`, on one of its nodes; the backend
asks this of a built agent with `isTriggerAvailable(triggerId)`.

The backend builds an agent's graph once, on its first query: it reserves the nodes the edges name, configured
with their merged config, compiles them into a LangGraph graph, and keeps it for the next queries.

**Configure.** A member of a project makes an agent of it - `POST /agents/{accountUuid}/{projectId}` with
`{"name", "templateId"}` - running one of the project's templates, or a premade platform template, which is
copied into the project first; the agent is `active`, `created_by` the account. `POST
/agents/{accountUuid}/{projectId}/{agentId}/rag` with `{"ragId"}` sets the agent's own `rag_enabled` and
`rag_id` to a ready rag of the project (rule 7), and drops the agent if it is kept, so it is built afresh on its
next query; other agents of its template keep their own rags. Both write Firestore, then the cache; another
process sees the change once its own cache drops the records.

**Query.** Anyone may ask an agent: `POST /agents/{projectId}/{agentId}/query` with `{"message": "..."}`. The
agent must belong to `projectId`, be `active` and not deleted, and its template must build; nothing else is
checked.

**Conversations.** A query may name its conversation by a `convoId`, a UUID the caller picks; the first query
with one starts the conversation. What is remembered of it is up to the agent's nodes: the run's state holds its
`chatHistory`, a plain object - empty, the first time - which nodes read and change as they will. The backend loads
it before the run and saves what the run ends with to `agent_memory` and to memoryDb, so the next query reads it
from memoryDb rather than Firestore. The answer goes back first: a run replies at its first node that runs after
the reply - a `chat_history` node - or at `output`, and keeping, summarising and saving the history happen after
that. A run that fails saves nothing; one that fails after its reply is logged. A query with no
`convoId` - a cron agent's report, say - is of no conversation: its history starts empty and is not saved. Each
node also has its own details for the run, `nodeDetails[nodeId]`, which start empty each run and are not saved.

A `chat_history` node keeps the conversation in this shape, which every chat model reads - the summary after its
system prompt, the messages before the run's own:

```json
"chat_history": {
  "summary": "The customer asked whether Acme delivers to Pune.",
  "messages": [
    {"role": "human", "text": "And to Nashik?"},
    {"role": "ai", "text": "Yes, in 2-3 business days."}
  ]
}
```

Each run it is in adds the message and the answer. Past its config's `summarise_after_tokens`, 10000 by default,
its model (`provider`, `model`) summarises all but the last message, the old summary with it, and the history
keeps that summary and the last message alone. It runs once the agent has replied, so none of this delays the
answer.

A chat model node's config may also set how much its model reasons: `reasoning_effort` on OpenAI and Gemini
(`minimal`, `low`, `medium`, `high`), `thinking: false` on DeepSeek, which a node that picks where a run goes
needs - DeepSeek's thinking mode refuses the forced tool call a pick is made with.

```json
{
  "agent_id": "88f95300-7453-4609-b685-2c424cc181d5",
  "project_id": "a7c3e2f1-9b4d-4e6a-8c2f-5d1b0e9a7f34",
  "convo_id": "6b0e3c1a-2d4f-4b8e-9a7c-5e1f3d2b4a69",
  "chat_history": {"...": "whatever the nodes keep"},
  "updated_at": "2026-10-25T09:00:04Z"
}
```

An `agent_memory` document's id is `{agent_id}_{convo_id}`, so a conversation is one agent's alone.

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

**Knowledge.** A chat model with `rag_search: true` may search its agent's rag as it answers, with the
`search_knowledge_base` tool: up to 3 rounds of searches, each search and what it found added to the conversation
before the answer, so the nodes after it read what the answer was made of. A rag not ready, or not of the
project, is not searched; the model is told so.

**Credentials.** `values` hold non-secret account details (phone number id, username, callback id).
`secret_values` hold tokens, passwords and API keys, encrypted with AES-256-GCM; the data key is wrapped by a
KMS key. Only the backend decrypts them, at the moment a node needs them. They are never sent to clients.

## Left to later modules

- **Runs**: run history, and which node used how many credits.
  A template `version` field can be added then if runs need to record which version produced a reply.
- **Billing**: credit prices per model.
