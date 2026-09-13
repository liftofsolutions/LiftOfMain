# CLAUDE.md

Guidance for working in this repository.

## Project

FastAPI service (`rag`), Python 3.12, dependencies managed with **uv**.

```
uv add <pkg>              # runtime dependency
uv add --dev <pkg>        # dev/test dependency
uv run <cmd>              # run inside the project venv
uv run fastapi dev src/main.py
```

## Platform

**Development happens on Windows; the service is deployed to AWS (Linux).**
Everything written here must run on both:

- Never hardcode a path separator or a drive letter. Branch on `os.name` when the
  platform genuinely differs, and cover both branches in the Btest.
- Do not assume a POSIX-only API exists. Some `psutil` calls behave differently or
  are unavailable on one of the two (`getloadavg`, `cpu_freq`, `open_files`,
  `net_connections` permissions) - guard them and degrade to `None` rather than
  raising.
- Anything platform-dependent gets tested for both Windows and Linux by patching
  `os.name`, not by skipping the test on the other platform.

## Naming and style

See `.claude/skills/naming-conventions/SKILL.md`. In short: **camelCase** for
locals, functions, methods and parameters; **PascalCase** for classes;
**UPPER_SNAKE_CASE** for module-level globals and constants. No snake_case
anywhere else. This is strict.

- Names are simple and descriptive: `buildChatModel`, `systemPrompt`, not
  `bcm` or `spData`.
- **Max 80 characters per line**, in code and in the JSON files beside it.
- Keep it simple: write exactly the logic asked for, nothing extra.

## Code layout

`src/` is the import root - there is no wrapper package inside it, so a component
is imported as `from serverDetails import service`, not `from rag.serverDetails`.
This is an application, not a distributable library (`package = false` in
`[tool.uv]`); `pythonpath = ["src"]` puts it on the path for pytest, and the
FastAPI CLI adds it when given `src/main.py`.

Routes and logic never live in the same file. Each feature is a component folder
directly under `src/`, and `main.py` only builds the app and wires routers in -
the standard FastAPI main-router layout.

```
src/
  main.py                      # FastAPI() + app.include_router(...) only
  <componentName>/             # camelCase folder, e.g. serverDetails/
    __init__.py
    routes.py                  # APIRouter + thin handlers that only call service
    service.py                 # all the logic; no FastAPI imports
    schemas.py                 # pydantic request/response models
```

Rules:

- `routes.py` declares the endpoints and delegates immediately. No logic in a
  handler.
- `service.py` must be importable and fully testable without FastAPI. Keep it free
  of `Request`, `HTTPException` and other framework types.
- `schemas.py` holds the pydantic models. Every route declares a `response_model`.
- Register a new component in `main.py` with
  `app.include_router(<componentName>Router)`.

Shared logic with no endpoints (`config/`, `LLM/`) skips `routes.py` and
`schemas.py`. `LLM/` builds one chat model per role from `configLLM.json` and
`systemPrompt.json` - see `docs/llm.md`.

## Testing

Two separate suites, and they do **not** overlap:

| Suite     | Purpose                    | Rule                                                    |
| --------- | -------------------------- | ------------------------------------------------------- |
| **Btest** | Component / logic tests    | Test the logic directly. **Never call a route.**        |
| **Ptest** | Product tests              | Drive the app end to end through its routes/HTTP client. |

```
Btest/<componentName>/<ComponentName>Test.py
Ptest/<componentName>/<ComponentName>Test.py
```

**How to write a Btest.** Build the data model exactly as the component is
supposed to receive it, call the processing function directly, and assert the
result matches what was expected:

1. Write small `build<Thing>(**overrides)` helpers returning a valid, healthy
   baseline, so each test overrides only the one field it exercises.
2. Call the real function - `assessOverload(...)`, not an endpoint.
3. Assert on the returned value. Where a function produces a response payload,
   validate it by instantiating the pydantic model (`HealthResponse(**payload)`) -
   that checks every field name and type in one shot.
4. Isolate the outside world with `monkeypatch` (clock, `psutil`, `os.name`), never
   with a running server.

No `TestClient`, `httpx`, `requests` or `app` import belongs in a Btest. If a test
needs the HTTP layer, it is a Ptest.

Test naming follows the project convention: `TestPascalCase` classes,
`testCamelCase` methods. Discovery is configured in `pyproject.toml`
(`python_files = ["*Test.py"]`).

```
uv run pytest Btest -q      # logic suite
uv run pytest Ptest -q      # product suite
uv run pytest -q            # both
```

Before calling work done, run the relevant suite and confirm the tests actually
fail when the logic is broken.
