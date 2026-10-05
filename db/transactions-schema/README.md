# Transactions schema

What each agent execution received, what it sent back, what it cost, and the data agents keep.
MongoDB-style documents; works on Firestore today.

## Files

| File | What it is |
|---|---|
| `transactions-erd.svg`, `transactions-erd.png` | Diagram with relationships and cardinality |
| `transactions.types.ts` | TypeScript types, price lookup, cost helpers |
| `transactions.mongo.js` | MongoDB validators and indexes (run once with mongosh) |
| `transactions.example.json` | Example prices, one WhatsApp transaction with its cost breakdown, one conversation in agent storage |

## Collections

| Collection | One document per | Holds |
|---|---|---|
| `transactions` | Agent execution (a trigger fired, the agent ran, it replied) | Raw `input` and `output` JSON, status, and the cost breakdown |
| `prices` | Priced unit per provider, sku and country | What one unit costs us |
| `agent_storage` | Item an agent keeps, by `kind` + `key` | Unstructured `data`, e.g. conversation history |

## Cost breakdown

```
transactions.total_micros            total for the execution
  └─ steps[]        step, total_micros
       └─ nodes[]   node_id, node_name, usage (raw), cost_micros, price_id → prices
```

- One `nodes[]` entry per priced line. An LLM call usually has two (input tokens, output tokens);
  a WhatsApp send has one (message). A node's cost in a step is the sum of its lines.
- `node_name` is copied from the template, so history reads correctly after a template changes.
- The cost is computed and stored when the transaction runs. Later price changes don't alter past transactions.

**Totals**

| Question | Query |
|---|---|
| Cost per agent | sum `transactions.total_micros` where `agent_id` = X (and a date range) |
| Cost per project | sum `transactions.total_micros` where `project_id` = X |
| Cost per node | MongoDB: `$unwind` `steps` and `steps.nodes`, group by `node_id`. Firestore can't sum inside arrays; add a daily rollup if needed |

## Money

Integer micro-USD everywhere: 1 USD = 1,000,000 micros. No floating-point rounding errors.

```
line cost = round(quantity × price_micros / per_units)
1,200 input tokens at 3,000,000 per 1,000,000 → 3,600 micros = $0.0036
```

## Prices and country

Lookup (`findPrice()`): the row with the same `handler`, `sku` and `unit` that is active at that time
(`effective_from ≤ t < effective_to`), using the recipient's `country` if a row exists, otherwise the row with no country.
To change a price, set `effective_to` on the old row and add a new one.

- **WhatsApp prices by country.** Meta charges per message by template category and the recipient's country
  (the customer's phone number, not the business number). Service replies inside the 24-hour customer-service
  window are free, as are utility and authentication templates inside that window. The backend reads the country
  from the customer's number when sending and records it in `usage`. Utility and authentication templates get
  monthly volume discounts, so compare stored cost against Meta's invoice.
- **AI models, search and email don't.** Their price rows have no country.
- **LINE** is a monthly plan per country, not a per-message price, so it isn't a line here.
- Country on the node (Agent schema) is a to-do for later.

`prices` is our cost. What customers are charged in credits belongs to the Billing module.

## Agent storage

Each agent keeps what it needs, in its own shape. It isn't linked to transactions.

| kind | key | data |
|---|---|---|
| `conversation` | `whatsapp:+919812345678` | `{ messages: [ {type: "human"}, {type: "ai"}, {type: "ai"}, {type: "human"} ] }` |
| `lead` | `+919812345678` | anything the agent extracts |

- `agent_id` + `kind` + `key` are unique together: one read loads a conversation.
- A document has a size cap (1 MiB on Firestore, 16 MB on MongoDB). A very long conversation can split itself by key,
  e.g. `whatsapp:+919812345678:2026-10`.
- On Firestore, uniqueness needs a deterministic document id such as `{agent_id}_{kind}_{key}` (with the key made URL-safe).

## Indexes

| Collection | Index | Use |
|---|---|---|
| `transactions` | `agent_id, created_at` | Cost and history per agent |
| `transactions` | `project_id, created_at` | Cost per project |
| `prices` | `handler, sku, unit, country, effective_from` (unique) | Price lookup; one price per unit, country and start date |
| `agent_storage` | `agent_id, kind, key` (unique) | Load an item |
| `agent_storage` | `project_id` | List a project's data |
