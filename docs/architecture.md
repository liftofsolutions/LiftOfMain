# Architecture

## Layout

```
Rag/
  src/                         # the import root
    main.py                    # FastAPI() + include_router() only
    serverDetails/             # a component
      __init__.py
      routes.py                # APIRouter, thin handlers
      service.py               # the logic, no FastAPI imports
      schemas.py               # pydantic response models
    DbAdapters/                # a component (library shaped)
      __init__.py              # the package's public surface
      base.py                  # what every adapter shares
      documentBase.py          # contract for keyed-document stores
      vectorBase.py            # contract for embedding stores
      mongoAdapter.py
      firestoreAdapter.py
      redisAdapter.py
      pineconeAdapter.py
    LLM/                       # chat model per role (no routes)
      __init__.py
      agentChatModel.py        # ChatModel + buildChatModel
      configLLM.json           # provider -> model -> settings
      systemPrompt.json        # role -> provider -> model -> prompt
  Btest/                       # component / logic tests
    serverDetails/ServerDetailsTest.py
  Ptest/                       # product tests (routes end to end) - not written yet
  docs/
  CLAUDE.md                    # the rules, in short form
  pyproject.toml
```

## `src` is the import root, with no wrapper package inside it

A component is imported as `from serverDetails import service`, **not**
`from rag.serverDetails import service`.

This changed deliberately. The project was scaffolded in the usual "src layout"
for a distributable library - `src/rag/`, with everything one level deeper than it
needed to be. Since this is an application rather than a package anyone will
`pip install`, that wrapper directory bought nothing and cost a level of nesting
in every path and every import. It was removed and the contents flattened up into
`src/`.

The consequences, all in `pyproject.toml`:

- **No `[build-system]`.** The `uv_build` backend expects a package directory
  named after the project; without `src/rag/` there is nothing for it to build.
- **`package = false` under `[tool.uv]`.** `uv sync` installs the dependencies but
  no longer installs the project itself into the venv.
- **`pythonpath = ["src"]` under `[tool.pytest.ini_options]`,** which is how the
  tests find the components now that nothing is installed.

Running the app needs no equivalent setting: the FastAPI CLI walks up from the
file it is given until it stops finding `__init__.py`, so `fastapi dev src/main.py`
treats `src` as the root and imports `main` from it.

The one thing this trades away is being able to build and publish `rag` as a
distribution. If that is ever wanted, the wrapper package comes back.

## The component pattern

Routes and logic never live in the same file. `main.py` builds the app and wires
routers in; it holds no logic of its own.

```python
# src/main.py
from serverDetails.routes import router as serverDetailsRouter

app = FastAPI(title="rag", version="0.1.0")
app.include_router(serverDetailsRouter)
```

Inside a component:

| File | Holds | Must not |
| --- | --- | --- |
| `routes.py` | `APIRouter`, path declarations, `response_model` | contain logic - handlers delegate immediately |
| `service.py` | all the logic | import FastAPI, or touch `Request` / `HTTPException` |
| `schemas.py` | pydantic request/response models | contain behaviour |

The rule that carries the most weight is the second: **`service.py` must be
importable and fully exercisable without FastAPI.** That is what makes a Btest
possible - the logic can be called directly, with no app, no client and no server.
If a service function starts wanting to raise `HTTPException`, it should return a
value or raise a plain exception instead, and let `routes.py` translate it.

To add a component: create `src/<componentName>/` with those three files, then one
`include_router` line in `main.py`.

## Naming

Enforced by `.claude/skills/naming-conventions/SKILL.md`:

- **camelCase** - locals, functions, methods, parameters (`getUserProfile`,
  `responseData`)
- **PascalCase** - classes (`HealthResponse`, `MongoAdapter`)
- **UPPER_SNAKE_CASE** - module-level globals and constants (`MAX_RETRIES`,
  `CPU_WARNING_PERCENT`)

No snake_case anywhere else, strictly. Names are simple and descriptive, and
no line passes **80 characters**, in code or in the JSON files beside it.

Component folders are camelCase (`serverDetails`);
`DbAdapters` is PascalCase because it reads as a package of classes.

Test names follow from the same rule: `TestPascalCase` classes with
`testCamelCase` methods, which is why pytest discovery is configured explicitly
rather than left on its snake_case defaults.

## Platform: Windows in development, AWS Linux in production

Every line written here has to run on both, and the difference is not theoretical
- it has already bitten this codebase once.

- **No hardcoded separators or drive letters.** `serverDetails.service` needs the
  disk usage of the drive the OS lives on. `os.path.abspath(os.sep)` looks
  portable and is not: on Windows it resolves against the *current* drive, so
  working from `D:\Code\Rag` it reported `D:` while the system drive was `C:`.
  The fix is an explicit branch on `os.name`, reading `%SystemDrive%` on Windows
  and using `/` elsewhere.
- **Guard platform-specific APIs.** Several `psutil` calls are unavailable or
  behave differently on one of the two - `getloadavg`, `cpu_freq`, `open_files`,
  and the permissions on `net_connections`. Catch and degrade to `None`; never let
  one raise out of a request.
- **Test both branches.** Anything platform-dependent is tested for Windows *and*
  Linux by patching `os.name`, not by skipping the test on whichever platform is
  not running it. Otherwise the production branch is never exercised until it is
  in production.
