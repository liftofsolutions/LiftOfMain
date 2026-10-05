# Agent schema

Agents for a project: what they are made of (nodes), what starts them (triggers), and the keys they use (credentials).
MongoDB-style documents; works on Firestore today.

## Files

| File | What it is |
|---|---|
| `agents-erd.svg`, `agents-erd.png` | Diagram with relationships and cardinality |
| `agents.types.ts` | TypeScript types, config merge, webhook path, `validateTemplate()` |
| `agents.mongo.js` | MongoDB validators and indexes (run once with mongosh) |
| `agents.example.json` | A complete example: node types, a credential, a template, an agent, its webhook URLs |

## Collections

| Collection | Owner | Holds |
|---|---|---|
| `node_types` | Platform, or one project | Reusable node templates: category, handler, default config |
| `agent_templates` | Platform, or one project | An agent's structure: embedded `nodes[]` and `triggers[]`, `rag_enabled` |
| `agents` | One project | The running agent: which template, name, status |
| `credentials` | One project | Keys and account details a node uses; secrets encrypted |

Embedded inside `agent_templates`:

- **node**: an instance of a node type, with its own `config` and an optional `credential_id`.
- **trigger**: points to an `io` node and names the `event` that starts the agent.

## Relationships

```
projects        1 ── 0..N  agents
projects     0..1 ── 0..N  agent_templates     (missing project_id = premade platform template)
projects     0..1 ── 0..N  node_types          (missing project_id = platform node type)
projects        1 ── 0..N  credentials
agent_templates 1 ── 0..N  agents
agent_templates 1 ── N     nodes      (embedded)
agent_templates 1 ── 0..N  triggers   (embedded)
node_types      1 ── 0..N  nodes
credentials  0..1 ── 0..N  nodes
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

## Rules (checked by the backend before saving)

Clients never write these collections directly; all writes go through the backend API.

1. A node's type is a platform node type or one from the template's project.
2. A node's credential belongs to the template's project. Platform templates hold no credentials; using a
   premade template copies it into the project first.
3. A trigger points to a node in the same template, and that node's type has category `io`.
4. An agent's template belongs to the agent's project.
5. Node ids, node names and trigger ids are unique within a template.

`validateTemplate()` in `agents.types.ts` implements rules 1–3 and 5.

## Runtime

**Config.** A node's effective config = `mergeConfig(node_types.config, node.config)`.
Objects merge deeply, the node wins, arrays are replaced.

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
- **Knowledge (RAG)**: the knowledge base `rag_enabled` switches on, with its Pinecone index id.
- **Billing**: credit prices per model.
