---
name: api-endpoint
description: How an API endpoint is built in this service - the route, service, schema split, how errors are logged and returned, and how long work is handed to the job manager. Use when adding, changing or reviewing any endpoint.
---

# How an endpoint works here

One feature is one folder under `src/`, and an endpoint is three files in it.
`src/` is the import root, so it is `from documentProcessing import service`,
never `from rag.documentProcessing`.

```
src/<componentName>/
  routes.py     # APIRouter + handlers that only call service
  service.py    # all the logic; imports no FastAPI
  model.py      # pydantic models for the responses
```

Register it once in `main.py`:

```python
app.include_router(documentProcessingRouter)
```

## The route layer

A handler does four things and nothing else: call the service, log a failure,
return an error, or return the result.

```python
router = APIRouter(prefix="/document", tags=["document processing"])

@router.post(
    "/{accountUuid}/{projectId}/rag",
    response_model=CreateJobResponse,
    summary="Create a rag database",
)
async def createRagDatabase(
    accountUuid: str, projectId: str
) -> dict | JSONResponse:
    """Reserve a rag database under a project and open a job for it."""
    try:
        created = await service.createRagDatabase(accountUuid, projectId)
    except Exception:
        LOGGER.exception(
            "Creating a job for account %s, project %s failed.",
            accountUuid,
            projectId,
        )
        return errorResponse(STATUS_INTERNAL_ERROR, INTERNAL_ERROR_DETAIL)

    if "error" in created:
        return errorResponse(STATUS_NOT_FOUND, created["error"])
    return created
```

Rules that fall out of that:

- **Scope ids live in the path.** Every endpoint is scoped to an account and a
  project, so `accountUuid` and `projectId` are path parameters. A handler
  never takes them from the body.
- **Never raise.** No `HTTPException`, no bubbling exception. An error is
  logged and returned - `errorResponse(status, message)` builds the
  `JSONResponse`.
- **Log the backtrace.** Use `LOGGER.exception(...)` inside `except`, which
  records the traceback with the message. The caller gets
  `INTERNAL_ERROR_DETAIL`, never the exception text.
- **Every route declares a `response_model`**, or `response_model=None` when
  the return annotation is a union and there is no schema yet - FastAPI
  otherwise tries to build a model out of `dict | JSONResponse` and fails at
  import.
- Returning a `JSONResponse` bypasses `response_model`, which is what lets an
  error body differ from the success shape.
- Bodies: a pydantic model in `model.py`, or `Body()` parameters for a
  handful of fields (`documentLink: str = Body()`).

Error bodies are `{"error": "..."}`. Statuses: 404 when the service reports
something missing, 500 when the call broke.

## The service layer

`service.py` must be importable and fully testable without FastAPI - no
`Request`, no `HTTPException`, no `Response`.

- **Return errors, do not raise them.** A missing record is
  `{"error": message}` after `LOGGER.error(message)`; the route turns it into
  a 404. Reserve exceptions for what the caller cannot act on.
- **Log what happened, with every attribute.** Build the record once and log
  the object, so the log cannot drift from what was stored:

  ```python
  await store.writeRecord(jobsPath, jobId, jobRecord)
  LOGGER.info("Opened job at %s: %s", jobsPath, jobRecord)
  ```

  `%s` placeholders, not f-strings, so a suppressed level costs nothing.
- **Time goes through `utcNow()`**, so a test can freeze it.
- **Ids are drawn, not guessed.** Read the ids already taken in that slice
  once, then `getNewId(takenIds)` redraws until it is free.

## Storage

`store.py` is an in-memory dict mirrored into Firestore. Every read comes from
memory; Firestore is a mirror, not a second source of truth.

| Call | Use it when |
| --- | --- |
| `fetchRecord(path, id)` | Reading one record. |
| `fetchRecordIds(path, filters)` | The ids in a slice, in one read. |
| `writeRecord(path, id, data)` | The caller needs the write to be durable. |
| `writeRecordSoon(path, id, data)` | A reply is owed now; mirror in the background. |

`writeRecordSoon` writes to memory and returns immediately - about 0.1 ms
against 1.2 s for an awaited Firestore write - and logs a failed mirror rather
than raising it. Use it where nothing in the reply depends on the record being
durable, and accept that a write in flight at shutdown is lost.

Paths are built from constants, one root for everything this service writes:

```python
USER_PROJECTS_PATH = RAG_DATABASE_ROOT + "/userProjects/{accountUuid}"
JOBS_PATH = RAG_DATABASE_ROOT + "/Jobs/{accountUuid}/{projectId}/jobs"
```

Firestore needs an odd number of path segments for a collection.

## Work that outlives the request

An endpoint never does slow work inline. It validates, hands the work to the
job manager, and replies.

```python
getJobManager().createJob(jobId, jobData, runDocumentJob, ttl=JOB_TTL)
return {"jobId": jobId, "status": JOB_STATUS_QUEUED, "message": "..."}
```

`JobManager` exposes exactly three things: `createJob`, `getStatus` and
`interruptAndDelete`. At most `MAX_CONCURRENT_JOBS` (5) run at once; the rest
wait at `queued` until a slot frees. The runner writes its own progress onto
the stored record and re-raises on failure, so both the record and the manager
show it failed.

## Tests

Two suites that never overlap:

- `Btest/<componentName>/<Thing>Test.py` - call the service function directly.
  No `TestClient`, no `httpx`, no `app` import.
- `Ptest/<componentName>/<Thing>Test.py` - drive the endpoint over HTTP.

A Btest that touches Firestore writes its own fixtures, runs the whole flow in
**one** `asyncio.run` (the adapter's gRPC channel is bound to the loop that
opened it), and deletes what it wrote in a `finally` so a failed assertion
still leaves the database clean. Give test ids a `uuid4` suffix.

Before calling the work done, run the suite *and* break the logic to confirm
the test fails. Asserting against the constant under test
(`assert ttl == service.JOB_TTL`) proves nothing - assert the literal.

## Adding an endpoint: the checklist

1. `model.py`: a response model for the success shape.
2. `service.py`: the logic, returning a plain dict, `{"error": ...}` when
   something is missing, logging both.
3. `routes.py`: the handler - path-scoped ids, `response_model`,
   `try/except` + `LOGGER.exception`, `errorResponse`, no raise.
4. `main.py`: `include_router` if the component is new.
5. A Btest against the service, and a Ptest if the HTTP layer matters.
