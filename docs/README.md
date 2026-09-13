# Documentation

What is in this repository, how it is put together, and why.

| Document | Covers |
| --- | --- |
| [architecture.md](architecture.md) | Layout, import root, component pattern, naming and platform rules |
| [serverDetails.md](serverDetails.md) | The `/health` and `/status` endpoints - full field reference and design decisions |
| [dbAdapters.md](dbAdapters.md) | The database adapter package: two contracts, four stores, what ACID means in each |
| [llm.md](llm.md) | Chat model per role: LLM config, system prompts |
| [testing.md](testing.md) | The Btest / Ptest split, how to write each, how to run them |

`CLAUDE.md` at the repo root is the short, enforceable version of the same rules,
kept there so it is loaded automatically when working in the project. These docs
are the longer explanation behind it.

## What this project is

`rag` is a FastAPI service, Python 3.12, dependencies managed with **uv**.

Three things exist so far:

1. **`serverDetails`** - operational endpoints. `/health` returns the machine's
   CPU, memory, disk, network and process vitals plus a verdict on whether the
   box is overloading; `/status` is a cheap liveness check.
2. **`DbAdapters`** - one async CRUD surface over Pinecone, MongoDB, Firestore and
   Redis, so the store behind a caller can be swapped without touching the
   caller.
3. **`LLM`** - one chat model per role (reviewer, chunker, summarizer,
   ragResponseEvaluator, chatAgent), with its provider, model, settings and
   system prompt.

The retrieval-augmented-generation logic the project is named for is not written
yet. What is here is the groundwork: the storage layer it will read and write
through, and the health surface it will be operated by.

## Running it

```bash
uv sync                             # create/refresh the venv from pyproject.toml
uv run fastapi dev src/main.py      # dev server with reload, on :8000
uv run pytest -q                    # the whole test suite
```

Then `http://127.0.0.1:8000/docs` for the interactive API docs, or:

```bash
curl http://127.0.0.1:8000/status
curl http://127.0.0.1:8000/health
```

## Adding dependencies

Always through uv, so `pyproject.toml` and `uv.lock` stay in step:

```bash
uv add <package>            # runtime
uv add --dev <package>      # test/tooling only
```
