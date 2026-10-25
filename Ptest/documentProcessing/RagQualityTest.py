"""Ptest for the rag endpoints: index the test documents, then rate how well
the chunks a query returns answer each test question.

Everything goes through the product the way a client would. The server is
started as its own process and reached over HTTP only; the dummy account
and project are written to Firestore by `dummyData`; the documents are
served from a local http server for the job to download; and a judge model
on the OpenAI API rates the chunks. Nothing from `src/` is imported.

In order:

1. Health (`/health`) and heartbeat (`/status`).
2. Create a job, submit `Ptest/docs/docs.rar` to it, and follow its status
   until it ends.
3. Ask every question in `Ptest/docs/question.json`, and have JUDGE_MODEL
   score each question's chunks together from 0 to 10. How many chunks a
   query returns is the server's setting: it is started with
   `RAG_QUERY_TOP_K` set to TOP_K.
4. Print the average score - overall, per document and per category - and
   how long the queries took, then save every rating and every query's
   time, with the server's log, to `Ptest/results/`. A query is timed from
   the request to the full answer, while WORKERS run at once; the judge's
   time is kept apart from it.
5. Delete the rag through the API, then the dummy records.

The server's log is read the whole time, and checked after every big step:
what it logged since the last check is printed - its warnings in full -
and an error fails the test there, with its traceback. While the job runs
and while questions are asked, an error stops the test at once.

The scores are reported, not held to a pass mark. A run spends real OpenAI
tokens, about half a million, and writes to the Firestore and Pinecone that
`test.env` names:

    uv run pytest Ptest/documentProcessing/RagQualityTest.py -s
"""

import functools
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

from dummyData import (
    DummyData,
    createDummyData,
    deleteDummyData,
    openFirestore,
    readSetting,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_FOLDER = PROJECT_ROOT / "Ptest" / "docs"
ARCHIVE_NAME = "docs.rar"
QUESTIONS_PATH = DOCS_FOLDER / "question.json"
RESULTS_FOLDER = PROJECT_ROOT / "Ptest" / "results"
RUN_STAMP = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"

JUDGE_MODEL = "gpt-6-luna"
# Chunks per query, set on the server it starts as RAG_QUERY_TOP_K.
TOP_K = 10
TOP_K_ENV_VAR = "RAG_QUERY_TOP_K"
# Questions asked and rated at the same time.
WORKERS = 8
# Questions between two progress lines.
PROGRESS_EVERY = 25
# The slowest queries named when the times are printed.
SLOWEST_SHOWN = 5

STARTUP_SECONDS = 60
INDEXING_SECONDS = 600
# How long chunks just saved to Pinecone may take to be found.
SEARCHABLE_SECONDS = 60
POLL_SECONDS = 1
REQUEST_SECONDS = 120
JUDGE_RETRIES = 5
# How long a check waits for the log lines of a step that just ended.
LOG_SETTLE_SECONDS = 0.5

JOB_STATUS_CREATED = "created"
JOB_STATUS_SUBMITTED = "submitted"
JOB_STATUS_DONE = "done"
JOB_STATUS_FAILED = "failed"
RAG_STATUS_READY = "ready"
STRATEGY_SIMPLE_RAG = "simpleRag"

# The server sets up no logging of its own, so uvicorn is handed this: every
# record from INFO up, each with its level, so an error can be told apart.
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "plain": {
            "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "plain",
            "stream": "ext://sys.stdout",
        }
    },
    "root": {"level": "INFO", "handlers": ["console"]},
}
# The start of a log record, as LOG_CONFIG formats it; one at ERROR or
# worse; and a warning.
RECORD_START = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} \| ")
ERROR_RECORD = re.compile(r"\| (ERROR|CRITICAL) \|")
WARNING_RECORD = re.compile(r"\| WARNING \|")
# Lines of the server's errors shown when a test fails; the log has all.
ERROR_LINES_SHOWN = 60

JUDGE_INSTRUCTIONS = (
    "You rate the retrieval step of a RAG system. You are given a question "
    "and the chunks the system retrieved for it. Score from 0 to 10 how "
    "well the chunks, taken together, let someone answer the question "
    "fully and correctly:\n"
    "10: every fact the answer needs is in the chunks.\n"
    "7: the main facts are there; a minor detail is missing.\n"
    "4: only part of the answer is there.\n"
    "1: the chunks are on the topic but do not hold the answer.\n"
    "0: nothing relevant.\n"
    "Judge only from what the chunks say, never from your own knowledge. "
    "In the reason, say in one sentence what is missing, or that nothing "
    "is."
)
RATING_FORMAT = {
    "type": "json_schema",
    "name": "chunkRating",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 0, "maximum": 10},
            "reason": {"type": "string"},
        },
        "required": ["score", "reason"],
        "additionalProperties": False,
    },
}


@dataclass
class Server:
    """The server under test: where it listens, and what it logged.

    The counts are moved on by every `checkLog`, so each check reports
    only what was logged since the one before.
    """

    baseUrl: str
    logPath: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    lineCount: int = 0
    checkedLineCount: int = 0
    checkedWarningCount: int = 0


class QuietHandler(SimpleHTTPRequestHandler):
    """Serves the test documents without writing a line per request."""

    def log_message(self, format, *args) -> None:
        """Keep the test's output to the test."""


def freePort() -> int:
    """A port nothing listens on, for the server to take."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def readLog(process: subprocess.Popen, server: Server) -> None:
    """Copy the server's output into its log; keep its errors and warnings.

    An error keeps the lines after it up to the next record - its
    traceback - and so does a traceback printed outside any record.
    """
    inError = False
    with server.logPath.open("w", encoding="utf-8") as log:
        for line in process.stdout:
            log.write(line)
            log.flush()
            server.lineCount += 1
            if RECORD_START.match(line):
                inError = bool(ERROR_RECORD.search(line))
                if WARNING_RECORD.search(line):
                    server.warnings.append(line.rstrip())
            elif line.startswith("Traceback"):
                inError = True
            if inError:
                server.errors.append(line.rstrip())


def waitUntilUp(server: Server, process: subprocess.Popen) -> None:
    """Return once the server answers its heartbeat; fail if it never does."""
    deadline = time.monotonic() + STARTUP_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(
                f"The server exited with code {process.returncode}; see "
                f"{server.logPath}."
            )
        try:
            response = httpx.get(f"{server.baseUrl}/status", timeout=5)
            if response.status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.5)
    pytest.fail(
        f"The server did not answer /status within {STARTUP_SECONDS}s; see "
        f"{server.logPath}."
    )


def assertNoServerErrors(server: Server) -> None:
    """Fail on any error the server has logged so far."""
    assert not server.errors, (
        f"The server logged errors (full log: {server.logPath}):\n"
        + "\n".join(server.errors[:ERROR_LINES_SHOWN])
    )


def checkLog(server: Server, step: str) -> None:
    """Report what the server logged during `step`, and fail on an error.

    Called once each big step ends. It waits a moment first, as the log
    is read while the server writes it.
    """
    time.sleep(LOG_SETTLE_SECONDS)
    newLines = server.lineCount - server.checkedLineCount
    newWarnings = server.warnings[server.checkedWarningCount :]
    server.checkedLineCount += newLines
    server.checkedWarningCount += len(newWarnings)
    print(
        f"   log after {step}: {newLines} lines, {len(newWarnings)} "
        f"warnings, {len(server.errors)} error lines"
    )
    for warning in newWarnings:
        print(f"      {warning}")
    assertNoServerErrors(server)


@pytest.fixture(scope="module")
def server() -> Iterator[Server]:
    """The rag server, started as its own process, for this module's tests.

    It runs on a free local port, reads `test.env` (the suite sets
    `RAG_ENV`) with RAG_QUERY_TOP_K set to TOP_K over it, and logs at INFO
    into `Ptest/results/`, read line by line as it goes.
    """
    RESULTS_FOLDER.mkdir(exist_ok=True)
    configFolder = Path(tempfile.mkdtemp(prefix="ragPtest"))
    logConfigPath = configFolder / "logConfig.json"
    logConfigPath.write_text(json.dumps(LOG_CONFIG), encoding="utf-8")
    port = freePort()
    started = Server(
        f"http://127.0.0.1:{port}",
        RESULTS_FOLDER / f"server-{RUN_STAMP}.log",
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--app-dir",
            "src",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-config",
            str(logConfigPath),
        ],
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            TOP_K_ENV_VAR: str(TOP_K),
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    reader = threading.Thread(
        target=readLog, args=(process, started), daemon=True
    )
    reader.start()
    try:
        waitUntilUp(started, process)
        print(f"\nServer up at {started.baseUrl}, log {started.logPath}")
        checkLog(started, "startup")
        yield started
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        reader.join(timeout=5)
        shutil.rmtree(configFolder, ignore_errors=True)


@pytest.fixture(scope="module")
def client(server: Server) -> Iterator[httpx.Client]:
    """An HTTP client on the server under test."""
    with httpx.Client(
        base_url=server.baseUrl, timeout=REQUEST_SECONDS
    ) as httpClient:
        yield httpClient


@pytest.fixture(scope="module")
def dummy() -> Iterator[DummyData]:
    """A dummy account and project in Firestore, deleted afterwards."""
    firestoreClient = openFirestore()
    created = createDummyData(firestoreClient)
    try:
        yield created
    finally:
        deleteDummyData(
            firestoreClient, created.accountUuid, created.projectId
        )
        firestoreClient.close()


@pytest.fixture(scope="module")
def documentLink() -> Iterator[str]:
    """A link to `docs.rar`, served over http for the job to download."""
    handler = functools.partial(QuietHandler, directory=str(DOCS_FOLDER))
    fileServer = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=fileServer.serve_forever, daemon=True).start()
    try:
        port = fileServer.server_address[1]
        yield f"http://127.0.0.1:{port}/{ARCHIVE_NAME}"
    finally:
        fileServer.shutdown()
        fileServer.server_close()


@pytest.fixture(scope="module")
def judge() -> Iterator[OpenAI]:
    """An OpenAI client for the judge model."""
    apiKey = readSetting("OPENAI_API_KEY")
    if not apiKey:
        pytest.fail("OPENAI_API_KEY is set in neither the env nor test.env.")
    with OpenAI(api_key=apiKey, max_retries=JUDGE_RETRIES) as openaiClient:
        yield openaiClient


def followStatus(
    client: httpx.Client, base: str, jobId: str, server: Server
) -> str:
    """Poll the job's status until it ends, and return the last one seen.

    Each change is printed as it happens. An error in the server's log
    fails the test at once, with the error, rather than when the job ends.
    """
    started = time.monotonic()
    seen = None
    while time.monotonic() - started < INDEXING_SECONDS:
        response = client.get(f"{base}/jobs/{jobId}/status")
        assert response.status_code == 200, response.text
        status = response.json()["status"]
        if status != seen:
            print(f"   {time.monotonic() - started:6.1f}s  {status}")
            seen = status
        if status in (JOB_STATUS_DONE, JOB_STATUS_FAILED):
            return status
        assertNoServerErrors(server)
        time.sleep(POLL_SECONDS)
    pytest.fail(f"Job {jobId} was still {seen} after {INDEXING_SECONDS}s.")


def indexDocuments(
    client: httpx.Client, server: Server, base: str, job: dict, link: str
) -> dict:
    """Submit `link` to the job, follow it until it ends, and return its
    rag, ready and searched in Pinecone, as the project's list shows it.
    """
    submitted = client.post(
        f"{base}/jobs/{job['jobId']}/submit", json={"documentLink": link}
    )
    assert submitted.status_code == 200, submitted.text
    assert submitted.json()["status"] == JOB_STATUS_SUBMITTED
    checkLog(server, "submit")

    status = followStatus(client, base, job["jobId"], server)
    assert status == JOB_STATUS_DONE, (
        f"The job {status}:\n" + "\n".join(server.errors)
    )
    rag = ragSummaryOf(client, base, job["ragId"])
    assert rag["status"] == RAG_STATUS_READY, rag
    assert rag["strategy"] == STRATEGY_SIMPLE_RAG, rag
    print(f"   {rag['tokenCount']} tokens in {rag['chunkCount']} chunks")
    checkLog(server, "indexing")
    return rag


def ragSummaryOf(client: httpx.Client, base: str, ragId: str) -> dict:
    """The rag as the project's list of rags shows it."""
    response = client.get(f"{base}/rags")
    assert response.status_code == 200, response.text
    [rag] = [rag for rag in response.json()["rags"] if rag["ragId"] == ragId]
    return rag


def askRag(
    client: httpx.Client, base: str, ragId: str, question: str
) -> list[dict]:
    """The chunks the rag returns for `question`, best first."""
    response = client.post(
        f"{base}/rags/{ragId}/query", json={"query": question}
    )
    assert response.status_code == 200, (
        f"{question!r}: {response.status_code} {response.text}"
    )
    return response.json()["matches"]


def waitUntilSearchable(
    client: httpx.Client, base: str, ragId: str, question: str, wanted: int
) -> None:
    """Wait until a query finds `wanted` chunks, as Pinecone may lag a save."""
    deadline = time.monotonic() + SEARCHABLE_SECONDS
    found = len(askRag(client, base, ragId, question))
    while found < wanted and time.monotonic() < deadline:
        time.sleep(POLL_SECONDS)
        found = len(askRag(client, base, ragId, question))
    if found < wanted:
        print(f"   only {found} of {wanted} chunks found after waiting")


def judgeInputOf(question: str, matches: list[dict]) -> str:
    """The question and its chunks, numbered, as the judge reads them."""
    chunks = [
        f"Chunk {number} ({match['fileName']}):\n{match['text']}"
        for number, match in enumerate(matches, 1)
    ]
    return f"Question: {question}\n\n" + "\n\n".join(chunks)


def rateChunks(judge: OpenAI, question: str, matches: list[dict]) -> dict:
    """The judge's score for the chunks, its reason, and the tokens spent."""
    response = judge.responses.create(
        model=JUDGE_MODEL,
        instructions=JUDGE_INSTRUCTIONS,
        input=judgeInputOf(question, matches),
        reasoning={"effort": "low"},
        text={"format": RATING_FORMAT},
    )
    rating = json.loads(response.output_text)
    return {
        "score": rating["score"],
        "reason": rating["reason"],
        "inputTokens": response.usage.input_tokens,
        "outputTokens": response.usage.output_tokens,
    }


def rateQuestion(
    client: httpx.Client,
    judge: OpenAI,
    server: Server,
    base: str,
    ragId: str,
    row: dict,
) -> dict:
    """Ask the rag one question, and have the judge score what came back.

    Each is timed on its own. The server's log is checked between the two,
    so an error stops the question before the judge is paid for it.
    """
    started = time.perf_counter()
    matches = askRag(client, base, ragId, row["question"])
    querySeconds = time.perf_counter() - started
    assertNoServerErrors(server)
    started = time.perf_counter()
    rating = rateChunks(judge, row["question"], matches)
    judgeSeconds = time.perf_counter() - started
    return {
        **row,
        **rating,
        "querySeconds": round(querySeconds, 3),
        "judgeSeconds": round(judgeSeconds, 3),
        "chunks": matches,
    }


def rateEveryQuestion(
    rate: Callable[[dict], dict], questions: list[dict]
) -> list[dict]:
    """Rate the questions WORKERS at a time, in order, printing progress.

    The first failure - a query, the judge, or an error in the server's
    log - cancels the questions not yet started, so no more tokens are
    spent on them, and is raised.
    """
    pool = ThreadPoolExecutor(max_workers=WORKERS)
    try:
        futures = [pool.submit(rate, row) for row in questions]
        rated = []
        for future in futures:
            rated.append(future.result())
            if len(rated) % PROGRESS_EVERY == 0:
                print(f"   rated {len(rated)} of {len(questions)}")
        return rated
    finally:
        pool.shutdown(cancel_futures=True)


def questionsOf(path: Path) -> list[dict]:
    """Every question in `path`, with the document and category it is in."""
    stored = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "documentId": document["id"],
            "documentTitle": document["title"],
            "categoryId": category["id"],
            "categoryName": category["name"],
            "questionId": question["id"],
            "question": question["question"],
        }
        for document in stored["documents"]
        for category in document["categories"]
        for question in category["questions"]
    ]


def averageOf(rows: list[dict], key: str = "score") -> float:
    """The rows' average under `key`, their score by default, to 2 places."""
    return round(statistics.mean(row[key] for row in rows), 2)


def timingOf(rows: list[dict], key: str) -> dict:
    """The spread of the rows' times under `key`, in seconds."""
    values = sorted(row[key] for row in rows)
    return {
        "min": values[0],
        "median": round(statistics.median(values), 3),
        "average": round(statistics.mean(values), 3),
        # The 95th percentile; it needs two times to cut between.
        "p95": round(
            statistics.quantiles(values, n=20, method="inclusive")[-1]
            if len(values) > 1
            else values[0],
            3,
        ),
        "max": values[-1],
    }


def groupBy(rows: list[dict], key: str) -> dict[str, list[dict]]:
    """`rows` by their value under `key`, in the order first seen."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    return groups


def reportOf(rag: dict, rated: list[dict]) -> dict:
    """The run's scores, in the documents and categories of question.json."""
    return {
        "finishedAt": datetime.now(timezone.utc).isoformat(),
        "judgeModel": JUDGE_MODEL,
        "topK": TOP_K,
        "rag": rag,
        "questionCount": len(rated),
        "averageScore": averageOf(rated),
        "averageQuerySeconds": averageOf(rated, "querySeconds"),
        "queriesAtOnce": WORKERS,
        "querySeconds": timingOf(rated, "querySeconds"),
        "judgeSeconds": timingOf(rated, "judgeSeconds"),
        "slowestQueries": [
            {"id": row["questionId"], "querySeconds": row["querySeconds"]}
            for row in sorted(
                rated, key=lambda row: row["querySeconds"], reverse=True
            )[:SLOWEST_SHOWN]
        ],
        "judgeTokens": {
            "input": sum(row["inputTokens"] for row in rated),
            "output": sum(row["outputTokens"] for row in rated),
        },
        "documents": [
            {
                "id": documentId,
                "title": documentRows[0]["documentTitle"],
                "averageScore": averageOf(documentRows),
                "averageQuerySeconds": averageOf(
                    documentRows, "querySeconds"
                ),
                "categories": [
                    {
                        "id": categoryId,
                        "name": categoryRows[0]["categoryName"],
                        "averageScore": averageOf(categoryRows),
                        "averageQuerySeconds": averageOf(
                            categoryRows, "querySeconds"
                        ),
                        "questions": [
                            {
                                "id": row["questionId"],
                                "question": row["question"],
                                "score": row["score"],
                                "reason": row["reason"],
                                "querySeconds": row["querySeconds"],
                                "judgeSeconds": row["judgeSeconds"],
                                "chunks": row["chunks"],
                            }
                            for row in categoryRows
                        ],
                    }
                    for categoryId, categoryRows in groupBy(
                        documentRows, "categoryId"
                    ).items()
                ],
            }
            for documentId, documentRows in groupBy(
                rated, "documentId"
            ).items()
        ],
    }


def saveReport(report: dict) -> Path:
    """Write the report into `Ptest/results/`, and say where."""
    path = RESULTS_FOLDER / f"ragQuality-{RUN_STAMP}.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def describeTiming(timing: dict) -> str:
    """A spread of times on one line."""
    return (
        f"min {timing['min']:.2f}s, median {timing['median']:.2f}s, "
        f"average {timing['average']:.2f}s, p95 {timing['p95']:.2f}s, "
        f"max {timing['max']:.2f}s"
    )


def printReport(report: dict, path: Path) -> None:
    """The averages - overall, per document and per category - and times.

    Each line gives the average score, then the average query time.
    """
    print(
        f"\nAverage score {report['averageScore']:.2f} / 10 over "
        f"{report['questionCount']} questions ({report['judgeModel']}, "
        f"topK {report['topK']})\n"
        f"Average query time {report['averageQuerySeconds']:.2f}s per "
        f"query ({report['queriesAtOnce']} at once)\n"
        "   score  query  document / category"
    )
    for document in report["documents"]:
        print(
            f"   {document['averageScore']:5.2f}  "
            f"{document['averageQuerySeconds']:5.2f}s  {document['title']}"
        )
        for category in document["categories"]:
            print(
                f"          {category['averageScore']:5.2f}  "
                f"{category['averageQuerySeconds']:5.2f}s  "
                f"{category['name']}"
            )
    slowest = ", ".join(
        f"{query['id']} {query['querySeconds']:.2f}s"
        for query in report["slowestQueries"]
    )
    tokens = report["judgeTokens"]
    print(
        f"Query time, {report['queriesAtOnce']} at once: "
        f"{describeTiming(report['querySeconds'])}\n"
        f"   slowest: {slowest}\n"
        f"Judge time: {describeTiming(report['judgeSeconds'])}\n"
        f"Judge tokens: {tokens['input']:,} in, {tokens['output']:,} out\n"
        f"Report: {path}"
    )


class TestRagQuality:
    """The rag endpoints, end to end, over HTTP."""

    def testHealth(self, client: httpx.Client, server: Server) -> None:
        response = client.get("/health")
        assert response.status_code == 200, response.text
        health = response.json()
        print(f"\n/health: {health['status']}, warnings {health['warnings']}")
        assert health["healthy"] is True
        checkLog(server, "health")

    def testHeartbeat(self, client: httpx.Client, server: Server) -> None:
        response = client.get("/status")
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "up"
        checkLog(server, "heartbeat")

    def testRagAnswersTheQuestions(
        self,
        client: httpx.Client,
        server: Server,
        dummy: DummyData,
        documentLink: str,
        judge: OpenAI,
    ) -> None:
        base = f"/rag/{dummy.accountUuid}/{dummy.projectId}"
        created = client.post(f"{base}/jobs")
        assert created.status_code == 200, created.text
        job = created.json()
        assert job["status"] == JOB_STATUS_CREATED
        print(f"\nJob {job['jobId']}, rag {job['ragId']}")
        try:
            checkLog(server, "create job")
            rag = indexDocuments(client, server, base, job, documentLink)

            questions = questionsOf(QUESTIONS_PATH)
            waitUntilSearchable(
                client,
                base,
                job["ragId"],
                questions[0]["question"],
                min(TOP_K, rag["chunkCount"]),
            )
            checkLog(server, "search warm-up")
            rate = functools.partial(
                rateQuestion, client, judge, server, base, job["ragId"]
            )
            rated = rateEveryQuestion(rate, questions)
            checkLog(server, f"{len(rated)} questions")
            report = reportOf(rag, rated)
            printReport(report, saveReport(report))
        finally:
            deleted = client.delete(f"{base}/rags/{job['ragId']}")
            if deleted.status_code != 200:
                print(f"   Rag {job['ragId']} was not deleted: "
                      f"{deleted.status_code} {deleted.text}")
        assert deleted.status_code == 200, deleted.text
        checkLog(server, "delete rag")
