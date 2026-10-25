# LiftOfMain: rag service

A FastAPI service that turns a project's documents into a rag. A job
downloads one document, reads it, and stores it:

- up to 10,000 tokens: as plain text on the rag's record in Firestore;
- more: as chunks in Pinecone, in the rag's own namespace, embedded by the
  index.

Python 3.12, dependencies managed with [uv](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync
```

Put the variables in `server.env` at the repository root. The tests read
`test.env` instead (`RAG_ENV` picks the file: `server`, the default, or
`test`). Both files and `keys/` are git-ignored. A variable set in the
environment wins over the file.

| Variable                         | What it is                                |
| -------------------------------- | ----------------------------------------- |
| `PINECONE_API_KEY`               | Pinecone key                              |
| `PINECONE_INDEX_NAME`            | The index, with integrated embedding      |
| `PINECONE_INDEX_HOST`            | Optional; saves a lookup on connect       |
| `GCP_PROJECT_ID`                 | Google Cloud project holding Firestore    |
| `FIRESTORE_DATABASE_ID`          | Firestore database name                   |
| `GOOGLE_APPLICATION_CREDENTIALS` | Service account key, from the repo root   |
| `RAG_CHUNK_MAX_TOKENS`           | Chunk size in tokens: 500, at most 1500   |
| `RAG_CHUNK_OVERLAP_TOKENS`       | Overlap between chunks: 100 unless set    |
| `RAG_QUERY_TOP_K`                | Chunks a query returns: 10, at most 20    |
| `RAG_UNRAR_TOOL`                 | Optional unrar to read `.rar` with        |

`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY` and `XAI_API_KEY`
are for the chat models in `src/LLM/`; no endpoint uses them yet.

Rar files are read by an external tool. Windows has one (`tar.exe`); on
Linux install `unrar` or `libarchive-tools`.

## Run

```bash
uv run fastapi dev src/main.py               # reloads on save, port 8000
uv run fastapi run src/main.py --port 8000   # as deployed
```

Then open <http://127.0.0.1:8000/docs>. `/status` is a liveness check and
`/health` reports the machine's vitals.

## API

Every rag endpoint sits under an account and a project,
`/rag/{accountUuid}/{projectId}`. The account must exist (else 404), the
project must be live (404), and the account must be a member of it (403).

| Method   | Path                   | Does                                     |
| -------- | ---------------------- | ---------------------------------------- |
| `POST`   | `/jobs`                | Create a job and its rag                 |
| `POST`   | `/jobs/{jobId}/submit` | Give the job its document and queue it   |
| `GET`    | `/jobs/{jobId}/status` | Where the job has got to                 |
| `DELETE` | `/jobs/{jobId}`        | Delete a job that has not started        |
| `GET`    | `/rags`                | The project's rags and their state       |
| `POST`   | `/rags/{ragId}/query`  | The parts of a ready rag that match      |
| `DELETE` | `/rags/{ragId}`        | Delete a rag: its chunks, then records   |

Submit takes the document's link. `countryId` and `metadata` are optional:

```json
{"documentLink": "https://example.com/handbook.pdf"}
```

Query takes the question. A simple rag gives its closest chunks with their
scores, best first, as many as `RAG_QUERY_TOP_K` says; a plain-text rag
gives its whole text as the one match.

```json
{"query": "How long does delivery take?"}
```

A document is a `.pdf`, `.docx`, `.txt` or `.md`, or a `.zip`, `.rar` or
`.tar` of them, up to 100 MB, behind an http(s) link.

An error is `{"error": "..."}`:

| Status | When                                                          |
| ------ | ------------------------------------------------------------- |
| 404    | The account, project, job or rag does not exist               |
| 403    | The account is not a member of the project                    |
| 409    | Not allowed in its state, such as a done job submitted again  |
| 502    | Pinecone did not finish a delete; deleting again finishes it  |
| 500    | Anything else; the details go to the log                      |

### A job's life

```
created -> submitted -> running -> done | failed
```

- At most 5 jobs run at once; the rest wait their turn.
- Done marks the rag `ready`. Failed marks it `failed` and keeps the
  reason as the job's `error`.
- A failed job may be submitted again, 3 runs in all. A done job may not.
- Only a job that has not started may be deleted. A rag cannot be deleted
  while its job runs.
- Jobs are held in memory, so a restart drops the waiting ones; submit
  them again.

### Agents

An agent is for public use: it needs no account, only the project it
belongs to.

| Method | Path                                  | Does                            |
| ------ | ------------------------------------- | ------------------------------- |
| `POST` | `/agents/{projectId}/{agentId}/query` | The agent's answer to a message |

```json
{"message": "Do you deliver to Pune?"}
```

It answers `{"agentId": "...", "convoId": null, "answer": "..."}`. A
message may name the conversation it belongs to by a `convoId`, a UUID
the caller picks: the first message with one starts the conversation,
and the next go on with it. What the agent remembers of a conversation
is up to its nodes: they keep its chat history, a plain JSON object,
saved after every answer in `agent_memory` and cached in memory, so the
next message reads it without going to Firestore. A message with no
`convoId` - a cron agent's, say - is of no conversation, and nothing of
it is kept.

```json
{"message": "How long does it take?", "convoId": "6b0e3c1a-..."}
```

The agent is built from its template on its first query - the nodes its
`edges` wire, compiled into a LangGraph graph; see
[db/agent-schema/](db/agent-schema/README.md) - and kept for the next
ones: ten agents at most, the least recently used let go first.

| Status | When                                                       |
| ------ | ---------------------------------------------------------- |
| 404    | The project has no such agent                              |
| 409    | The agent is paused, cannot be built, or ran past 25 steps |
| 422    | The message is empty, or the `convoId` is not a UUID       |
| 500    | Anything else; the details go to the log                   |

## Tests

```bash
uv run pytest Btest -q   # logic tests: call the service directly
uv run pytest Ptest -q   # product tests: through the routes
uv run pytest -q         # both
```

The Ptest starts the server itself, indexes `Ptest/docs/docs.rar` and asks
every question in `Ptest/docs/question.json`; `gpt-6-luna` scores each
question's chunks from 0 to 10. It uses the real Firestore, Pinecone and
OpenAI in `test.env`, and costs a few cents a run. Add `-s` to see the
scores; the full report and the server's log land in `Ptest/results/`.
`Ptest/documentProcessing/dummyData.py` makes and removes its dummy account
and project, and can be run on its own.

## More

- [docs/](docs/README.md): architecture, database adapters, LLM, testing
- [db/agent-schema/](db/agent-schema/README.md): agents, their templates
  and the graphs those wire
- [db/knowledge-schema/](db/knowledge-schema/README.md): the rag and job
  records
- [CLAUDE.md](CLAUDE.md): the rules for code in this repository
