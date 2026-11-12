"""Stress Ptest for an agent: a support agent, its rag indexed from the
test documents, asked QUERY_COUNT questions at once, each in a
conversation of its own, timed, and reported.

Everything goes through the product the way a client would, over HTTP to
the server as its own process, using `RagQualityTest`'s server, dummy
project and document server, and nothing from `src/`. The one exception
is the agent's template and its node types: no endpoint writes those yet,
so they are written by hand, to Firestore, as the backend will.

The agent is one of FLOWS, as AGENT_STRESS_FLOW in the environment
names it - `fastSupport`, unless it names another:

* `guardThenSupport` - guard passes a question on to support, or turns
  it away; support searches the agent's rag and answers:

      input -> guard -> support -> history -> output
                 |
                 +-> output   (turned away)

* `fastSupport` - the TEMPORARY `fast_support_chat` node, the same
  guard and support in one: the guard's pick and the rag search at
  once, then support's answer:

      input -> fastSupport -> history -> output

Every model is MODEL, OpenAI's, at REASONING - no reasoning, as it calls
tools, a search or a pick, only so. Support answers in
SUPPORT_MAX_TOKENS at most; history keeps the turn, and summarises the
conversation if it is past 10000 tokens, after the agent has replied.

In order:

1. The dummy account and project are written to Firestore.
2. `Ptest/docs/docs.rar` is indexed from its URL, through the rag
   endpoints, into a rag of the project.
3. The template and its node types are written to Firestore by hand.
4. The agent is configured through its endpoint, running the template,
   then given the rag through its own.
5. One warm-up query builds the agent; its time is kept apart.
6. QUERY_COUNT queries - the questions of `question.json` in turn, each
   with a new `convoId` - are sent AT_ONCE at a time, and each is timed
   from its request to its answer.
7. The report - how many were answered, the times, the throughput, the
   searches of the rag the server logged, the conversations saved, the
   tokens each model used and what they cost, and every failure - is
   printed and saved to `Ptest/results/`. The tokens are the server's
   `Model usage` lines, logged during the queries, and their cost is
   by PRICES: Pinecone's is not counted.
8. The conversations, the agent, its template and node types, then the
   rag and the dummy records are deleted.

The test fails if any query is not answered, or answered empty, or the
server logs an error, once the report is saved. A run spends real
OpenAI tokens - two or three calls a query - and writes to the
Firestore and Pinecone that `test.env` names:

    uv run pytest Ptest/agents/AgentStressTest.py -s

`AGENT_STRESS_QUERIES` and `AGENT_STRESS_AT_ONCE` in the environment ask
fewer queries, or fewer at once, for a smaller run;
`AGENT_STRESS_FLOW=guardThenSupport` stresses the agent of two nodes.
"""

import asyncio
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from google.cloud import firestore

# The rag Ptest's server and records, which this one shares.
PTEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PTEST_ROOT / "documentProcessing"))

from AgentConversationTest import (  # noqa: E402, F401
    AGENT_MEMORY,
    AGENT_TEMPLATES,
    AGENTS,
    GUARD_PROMPT,
    MODEL,
    NODE_TYPES,
    REASONING,
    SUMMARISE_AFTER_TOKENS,
    SUPPORT_PROMPT,
    deleteRecords,
    firestoreClient,
    newId,
    searchesLogged,
    writeRecords,
)
from dummyData import DummyData  # noqa: E402
from RagQualityTest import (  # noqa: E402, F401
    QUESTIONS_PATH,
    RESULTS_FOLDER,
    RUN_STAMP,
    TOP_K,
    Server,
    assertNoServerErrors,
    checkLog,
    client,
    documentLink,
    dummy,
    indexDocuments,
    questionsOf,
    server,
    waitUntilSearchable,
)

# Queries asked, and how many at once: all of them, unless the
# environment says less, for a smaller run.
QUERY_COUNT = int(os.environ.get("AGENT_STRESS_QUERIES", 1000))
AT_ONCE = int(os.environ.get("AGENT_STRESS_AT_ONCE", QUERY_COUNT))
# How long a query may wait for its answer, while all the others wait on
# the same models.
QUERY_SECONDS = 900
# The guard's model, which only picks: the same as support's.
GUARD_HANDLER = "openai_chat"
GUARD_MODEL = MODEL
# The most tokens support's answer may take.
SUPPORT_MAX_TOKENS = 300
# The agent stressed: one of FLOWS.
FLOW = os.environ.get("AGENT_STRESS_FLOW", "fastSupport")
# What each model costs, in dollars per million tokens - input, cached
# input, output - as OpenAI's pricing page gave it on 2026-10-26.
PRICES = {
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
}
# What starts a line of the server's log giving a model call's tokens,
# as `LLM.usageLog` writes it; and the tokens it counts.
USAGE_PREFIX = "Model usage: "
TOKEN_KINDS = ("input", "cachedInput", "output", "reasoning")
# How long the conversations may take to be saved, after the last reply,
# and how often they are counted meanwhile.
SAVE_SECONDS = 120
POLL_SECONDS = 1
# Firestore takes at most 500 writes in a batch.
DELETES_PER_BATCH = 500
# The slowest queries, and the failures, named in the printed report.
SLOWEST_SHOWN = 5
FAILURES_SHOWN = 10
# The percentiles of the query times reported.
PERCENTILES = (50, 90, 95, 99)


def agentRecordsOf(dummy: DummyData) -> list[tuple[str, dict]]:
    """The agent's template and its three node types, of the dummy
    project, each with its collection, as they are written by hand: the
    template last."""
    now = datetime.now(timezone.utc)
    stamps = {"created_at": now, "updated_at": now}
    chatTypeId, historyTypeId, templateId = newId(), newId(), newId()
    guardTypeId = newId()
    guardId, supportId, historyId = newId(), newId(), newId()

    def node(nodeId: str, typeId: str, name: str, config: dict) -> dict:
        return {
            "id": nodeId,
            "node_type_id": typeId,
            "name": name,
            "config": config,
            **stamps,
        }

    return [
        (
            NODE_TYPES,
            {
                "_id": guardTypeId,
                "project_id": dummy.projectId,
                "category": "chat_model",
                "handler": GUARD_HANDLER,
                "name": "Stress guard chat",
                "config": {
                    "model": GUARD_MODEL,
                    "reasoning_effort": REASONING,
                },
                "is_active": True,
                **stamps,
            },
        ),
        (
            NODE_TYPES,
            {
                "_id": chatTypeId,
                "project_id": dummy.projectId,
                "category": "chat_model",
                "handler": "openai_chat",
                "name": "Stress OpenAI chat",
                "config": {"model": MODEL, "reasoning_effort": REASONING},
                "is_active": True,
                **stamps,
            },
        ),
        (
            NODE_TYPES,
            {
                "_id": historyTypeId,
                "project_id": dummy.projectId,
                "category": "memory",
                "handler": "chat_history",
                "name": "Stress chat history",
                "config": {"provider": "openai", "model": MODEL},
                "is_active": True,
                **stamps,
            },
        ),
        (
            AGENT_TEMPLATES,
            {
                "_id": templateId,
                "project_id": dummy.projectId,
                "name": "Stress customer support",
                "nodes": [
                    node(
                        guardId,
                        guardTypeId,
                        "guard",
                        {
                            "system_prompt": GUARD_PROMPT,
                            "description": "Checks the message.",
                        },
                    ),
                    node(
                        supportId,
                        chatTypeId,
                        "support",
                        {
                            "system_prompt": SUPPORT_PROMPT,
                            "description": "Answers the customer.",
                            "rag_search": True,
                            "max_tokens": SUPPORT_MAX_TOKENS,
                        },
                    ),
                    node(
                        historyId,
                        historyTypeId,
                        "history",
                        {
                            "description": "Keeps the conversation.",
                            "summarise_after_tokens": SUMMARISE_AFTER_TOKENS,
                        },
                    ),
                ],
                "edges": [
                    {"from": "input", "to": guardId},
                    {"from": guardId, "to": supportId},
                    {"from": guardId, "to": "output"},
                    {"from": supportId, "to": historyId},
                    {"from": historyId, "to": "output"},
                ],
                "triggers": [],
                "is_active": True,
                **stamps,
            },
        ),
    ]


def fastAgentRecordsOf(dummy: DummyData) -> list[tuple[str, dict]]:
    """The fast support agent's template and its two node types, of the
    dummy project, each with its collection, as they are written by
    hand: the template last."""
    now = datetime.now(timezone.utc)
    stamps = {"created_at": now, "updated_at": now}
    fastTypeId, historyTypeId, templateId = newId(), newId(), newId()
    fastId, historyId = newId(), newId()
    return [
        (
            NODE_TYPES,
            {
                "_id": fastTypeId,
                "project_id": dummy.projectId,
                "category": "chat_model",
                "handler": "fast_support_chat",
                "name": "Stress fast support",
                "config": {"model": MODEL, "reasoning_effort": REASONING},
                "is_active": True,
                **stamps,
            },
        ),
        (
            NODE_TYPES,
            {
                "_id": historyTypeId,
                "project_id": dummy.projectId,
                "category": "memory",
                "handler": "chat_history",
                "name": "Stress chat history",
                "config": {"provider": "openai", "model": MODEL},
                "is_active": True,
                **stamps,
            },
        ),
        (
            AGENT_TEMPLATES,
            {
                "_id": templateId,
                "project_id": dummy.projectId,
                "name": "Stress fast customer support",
                "nodes": [
                    {
                        "id": fastId,
                        "node_type_id": fastTypeId,
                        "name": "fastSupport",
                        "config": {
                            "system_prompt": SUPPORT_PROMPT,
                            "description": "Answers the customer.",
                            "rag_search": True,
                            "max_tokens": SUPPORT_MAX_TOKENS,
                            "guard": {
                                "model": GUARD_MODEL,
                                "system_prompt": GUARD_PROMPT,
                                "reasoning_effort": REASONING,
                            },
                        },
                        **stamps,
                    },
                    {
                        "id": historyId,
                        "node_type_id": historyTypeId,
                        "name": "history",
                        "config": {
                            "description": "Keeps the conversation.",
                            "summarise_after_tokens": SUMMARISE_AFTER_TOKENS,
                        },
                        **stamps,
                    },
                ],
                "edges": [
                    {"from": "input", "to": fastId},
                    {"from": fastId, "to": historyId},
                    {"from": historyId, "to": "output"},
                ],
                "triggers": [],
                "is_active": True,
                **stamps,
            },
        ),
    ]


# The agents that can be stressed, by name: the records each is made of.
FLOWS = {
    "guardThenSupport": agentRecordsOf,
    "fastSupport": fastAgentRecordsOf,
}


def configureAgent(
    client: httpx.Client, dummy: DummyData, templateId: str
) -> str:
    """The id of the agent the endpoint makes of the template."""
    response = client.post(
        f"/agents/{dummy.accountUuid}/{dummy.projectId}",
        json={"name": "Stress support agent", "templateId": templateId},
    )
    assert response.status_code == 200, response.text
    configured = response.json()
    assert configured["templateId"] == templateId, configured
    assert configured["status"] == "active", configured
    return configured["agentId"]


def configureRag(
    client: httpx.Client, dummy: DummyData, agentId: str, ragId: str
) -> None:
    """Give the agent the rag, through the endpoint."""
    response = client.post(
        f"/agents/{dummy.accountUuid}/{dummy.projectId}/{agentId}/rag",
        json={"ragId": ragId},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"agentId": agentId, "ragId": ragId}


async def askOnce(
    httpClient: httpx.AsyncClient,
    gate: asyncio.Semaphore,
    path: str,
    row: dict,
) -> dict:
    """One query, in a conversation of its own, once the gate lets it
    go: its question, conversation, status, time and answer, or why it
    failed. It never raises: a failure is a result."""
    convoId = newId()
    async with gate:
        started = time.perf_counter()
        try:
            response = await httpClient.post(
                path, json={"message": row["question"], "convoId": convoId}
            )
        except httpx.HTTPError as error:
            status, body = None, f"{type(error).__name__}: {error}"
        else:
            status, body = response.status_code, response.text
        seconds = time.perf_counter() - started
    result = {
        "id": row["questionId"],
        "convoId": convoId,
        "status": status,
        "seconds": round(seconds, 3),
    }
    if status == 200:
        result["answer"] = json.loads(body)["answer"]
    else:
        result["error"] = body[:500]
    return result


async def askAll(
    baseUrl: str, path: str, rows: list[dict]
) -> tuple[list[dict], float]:
    """Every query of `rows`, AT_ONCE at a time, and the seconds from the
    first sent to the last answered."""
    gate = asyncio.Semaphore(AT_ONCE)
    limits = httpx.Limits(
        max_connections=AT_ONCE, max_keepalive_connections=AT_ONCE
    )
    async with httpx.AsyncClient(
        base_url=baseUrl, timeout=QUERY_SECONDS, limits=limits
    ) as httpClient:
        started = time.perf_counter()
        results = await asyncio.gather(
            *(askOnce(httpClient, gate, path, row) for row in rows)
        )
        return list(results), time.perf_counter() - started


def percentileOf(values: list[float], percent: int) -> float:
    """The `percent`th percentile of `values`, sorted, by nearest rank."""
    rank = max(1, round(percent / 100 * len(values)))
    return values[rank - 1]


def spreadOf(values: list[float]) -> dict:
    """The spread of `values`: least, average, PERCENTILES and most."""
    values = sorted(values)
    if not values:
        return {}
    return {
        "min": values[0],
        "average": round(statistics.mean(values), 3),
        **{
            f"p{percent}": percentileOf(values, percent)
            for percent in PERCENTILES
        },
        "max": values[-1],
    }


def timesOf(results: list[dict]) -> dict:
    """The spread of the answered queries' times, in seconds."""
    return spreadOf([result["seconds"] for result in results])


def usageLogged(lines: list[str]) -> list[dict]:
    """The tokens of each model call the server logged in `lines`."""
    return [
        json.loads(line.split(USAGE_PREFIX, 1)[1])
        for line in lines
        if USAGE_PREFIX in line
    ]


def costsOf(usages: list[dict], queries: int) -> dict:
    """The tokens each model used in `usages`, and what they cost in
    dollars by PRICES, by model; and the cost of all, and of a query."""
    byModel: dict[str, dict] = {}
    for usage in usages:
        used = byModel.setdefault(
            usage["model"],
            {"provider": usage["provider"], "calls": 0}
            | dict.fromkeys(TOKEN_KINDS, 0),
        )
        used["calls"] += 1
        for kind in TOKEN_KINDS:
            used[kind] += usage[kind]
    for model, used in byModel.items():
        prices = PRICES.get(model)
        if prices is None:
            used["dollars"] = None
            continue
        inputPrice, cachedPrice, outputPrice = prices
        uncached = used["input"] - used["cachedInput"]
        used["dollars"] = round(
            (
                uncached * inputPrice
                + used["cachedInput"] * cachedPrice
                + used["output"] * outputPrice
            )
            / 1_000_000,
            6,
        )
    dollars = sum(used["dollars"] or 0 for used in byModel.values())
    return {
        "byModel": byModel,
        "unpricedModels": [
            model for model, used in byModel.items() if used["dollars"] is None
        ],
        "dollars": round(dollars, 6),
        "dollarsPerQuery": round(dollars / queries, 8),
        "dollarsPer1000Queries": round(dollars * 1000 / queries, 4),
    }


def failuresByStatus(failures: list[dict]) -> dict[str, int]:
    """How many queries failed with each status, a dropped connection
    as "none"."""
    counts: dict[str, int] = {}
    for failure in failures:
        status = str(failure["status"] or "none")
        counts[status] = counts.get(status, 0) + 1
    return counts


def reportOf(
    rag: dict,
    warmUp: dict,
    results: list[dict],
    wallSeconds: float,
    searches: int,
    saved: tuple[int, float],
    costs: dict,
    server: Server,
) -> dict:
    """The run's report: what was asked, how fast it was answered, what
    it cost, and what failed."""
    answered = [result for result in results if result["status"] == 200]
    failures = [result for result in results if result["status"] != 200]
    empty = [result for result in answered if not result["answer"].strip()]
    return {
        "finishedAt": datetime.now(timezone.utc).isoformat(),
        "flow": FLOW,
        "model": MODEL,
        "guardModel": GUARD_MODEL,
        "supportMaxTokens": SUPPORT_MAX_TOKENS,
        "rag": rag,
        "queries": len(results),
        "atOnce": AT_ONCE,
        "answered": len(answered),
        "emptyAnswers": len(empty),
        "failed": len(failures),
        "failuresByStatus": failuresByStatus(failures),
        "wallSeconds": round(wallSeconds, 3),
        "queriesPerSecond": round(len(answered) / wallSeconds, 3),
        "warmUpSeconds": warmUp["seconds"],
        "seconds": timesOf(answered),
        "searchesOfTheRag": searches,
        "conversationsSaved": saved[0],
        "savedSecondsAfterTheLastReply": saved[1],
        "costs": costs,
        "serverErrorLines": len(server.errors),
        "slowest": sorted(
            answered, key=lambda result: result["seconds"], reverse=True
        )[:SLOWEST_SHOWN],
        "failures": failures,
        "results": results,
    }


def saveReport(report: dict) -> Path:
    """Write the report into `Ptest/results/`, and say where."""
    path = RESULTS_FOLDER / f"agentStress-{RUN_STAMP}.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def printReport(report: dict, path: Path) -> None:
    """The report's numbers, its slowest queries, and its first
    failures."""
    times = report["seconds"]
    spread = ", ".join(f"{name} {value:.2f}s" for name, value in times.items())
    print(
        f"\nStress report, {report['flow']} (guard {report['guardModel']}, "
        f"support {report['model']}, at most {report['supportMaxTokens']} "
        "tokens)\n"
        f"   queries:        {report['queries']}, {report['atOnce']} at once\n"
        f"   answered:       {report['answered']}, "
        f"{report['emptyAnswers']} of them empty\n"
        f"   failed:         {report['failed']} "
        f"{report['failuresByStatus'] or ''}\n"
        f"   wall time:      {report['wallSeconds']:.2f}s\n"
        f"   throughput:     {report['queriesPerSecond']:.2f} answers/s\n"
        f"   warm-up query:  {report['warmUpSeconds']:.2f}s\n"
        f"   answer times:   {spread}\n"
        f"   rag searches:   {report['searchesOfTheRag']}\n"
        f"   conversations:  {report['conversationsSaved']} saved, "
        f"{report['savedSecondsAfterTheLastReply']:.2f}s after the last "
        "reply\n"
        f"   server errors:  {report['serverErrorLines']} log lines"
    )
    printCosts(report["costs"])
    for result in report["slowest"]:
        print(f"   slow: {result['seconds']:.2f}s  {result['id']}")
    for failure in report["failures"][:FAILURES_SHOWN]:
        print(
            f"   failed: {failure['status']}  {failure['id']}  "
            f"{failure['error'][:100]}"
        )
    print(f"Report: {path}")


def printCosts(costs: dict) -> None:
    """The tokens each model used, and what they cost, then the total."""
    print("   usage and cost:")
    for model, used in costs["byModel"].items():
        dollars = (
            "not priced" if used["dollars"] is None
            else f"${used['dollars']:.6f}"
        )
        print(
            f"      {model}: {used['calls']} calls, {used['input']} in "
            f"({used['cachedInput']} cached), {used['output']} out "
            f"({used['reasoning']} reasoning) -> {dollars}"
        )
    print(
        f"      total ${costs['dollars']:.6f}: "
        f"${costs['dollarsPerQuery']:.8f} a query, "
        f"${costs['dollarsPer1000Queries']:.4f} per 1000 queries "
        "(Pinecone not counted)"
    )


def countConversations(
    firestoreClient: firestore.Client, agentId: str
) -> int:
    """How many conversations the agent has saved."""
    conversations = firestoreClient.collection(AGENT_MEMORY).where(
        filter=firestore.FieldFilter("agent_id", "==", agentId)
    )
    [[counted]] = conversations.count().get()
    return counted.value


def waitForConversations(
    firestoreClient: firestore.Client, agentId: str, wanted: int
) -> tuple[int, float]:
    """Wait, SAVE_SECONDS at most, until the agent has saved `wanted`
    conversations - each is saved after its reply - and return how many
    it has, and how long that took."""
    started = time.monotonic()
    saved = countConversations(firestoreClient, agentId)
    while saved < wanted and time.monotonic() - started < SAVE_SECONDS:
        time.sleep(POLL_SECONDS)
        saved = countConversations(firestoreClient, agentId)
    return saved, round(time.monotonic() - started, 3)


def deleteConversations(
    firestoreClient: firestore.Client, agentId: str
) -> int:
    """Delete every conversation the agent saved, DELETES_PER_BATCH at a
    time; how many there were."""
    conversations = firestoreClient.collection(AGENT_MEMORY).where(
        filter=firestore.FieldFilter("agent_id", "==", agentId)
    )
    references = [document.reference for document in conversations.stream()]
    for start in range(0, len(references), DELETES_PER_BATCH):
        batch = firestoreClient.batch()
        for reference in references[start : start + DELETES_PER_BATCH]:
            batch.delete(reference)
        batch.commit()
    return len(references)


class TestAgentStress:
    """A support agent, asked QUERY_COUNT questions at once, over HTTP."""

    def testAgentAnswersEveryQueryAtOnce(
        self,
        client: httpx.Client,
        server: Server,
        dummy: DummyData,
        documentLink: str,
        firestoreClient: firestore.Client,
    ) -> None:
        ragBase = f"/rag/{dummy.accountUuid}/{dummy.projectId}"
        created = client.post(f"{ragBase}/jobs")
        assert created.status_code == 200, created.text
        job = created.json()
        ragId = job["ragId"]
        print(f"\nJob {job['jobId']}, rag {ragId}")
        records: list[tuple[str, dict]] = []
        agentId = None
        try:
            checkLog(server, "create job")
            rag = indexDocuments(client, server, ragBase, job, documentLink)
            questions = questionsOf(QUESTIONS_PATH)
            waitUntilSearchable(
                client,
                ragBase,
                ragId,
                questions[0]["question"],
                min(TOP_K, rag["chunkCount"]),
            )
            checkLog(server, "search warm-up")

            records = FLOWS[FLOW](dummy)
            writeRecords(firestoreClient, records)
            templateId = records[-1][1]["_id"]
            agentId = configureAgent(client, dummy, templateId)
            configureRag(client, dummy, agentId, ragId)
            print(f"   template {templateId}, agent {agentId}")
            checkLog(server, "configure")

            path = f"/agents/{dummy.projectId}/{agentId}/query"
            rows = [
                questions[index % len(questions)]
                for index in range(QUERY_COUNT)
            ]
            [warmUp], _ = asyncio.run(askAll(server.baseUrl, path, rows[:1]))
            assert warmUp["status"] == 200, warmUp
            print(f"   warm-up: {warmUp['seconds']:.2f}s")
            checkLog(server, "warm-up")

            searchesBefore = searchesLogged(server, ragId)
            linesBefore = server.lineCount
            print(
                f"   asking {QUERY_COUNT} questions of {FLOW}, {AT_ONCE} "
                "at once"
            )
            results, wallSeconds = asyncio.run(
                askAll(server.baseUrl, path, rows)
            )
            # Each conversation is saved after its reply: the warm-up's,
            # and one for each query.
            savedConversations, savedSeconds = waitForConversations(
                firestoreClient, agentId, QUERY_COUNT + 1
            )
            searches = searchesLogged(server, ragId) - searchesBefore
            lines = server.logPath.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()[linesBefore:]
            costs = costsOf(usageLogged(lines), QUERY_COUNT)
            deleteConversations(firestoreClient, agentId)
            report = reportOf(
                rag,
                warmUp,
                results,
                wallSeconds,
                searches,
                (savedConversations, savedSeconds),
                costs,
                server,
            )
            printReport(report, saveReport(report))
            checkLog(server, f"{QUERY_COUNT} queries")
            assert report["failed"] == 0, report["failuresByStatus"]
            assert report["emptyAnswers"] == 0
            assert savedConversations == QUERY_COUNT + 1
        finally:
            if agentId is not None:
                deleteConversations(firestoreClient, agentId)
                records.append((AGENTS, {"_id": agentId}))
            if records:
                documents = [
                    (collection, record["_id"])
                    for collection, record in records
                ]
                deleteRecords(firestoreClient, documents)
            deleted = client.delete(f"{ragBase}/rags/{ragId}")
            if deleted.status_code != 200:
                print(f"   Rag {ragId} was not deleted: "
                      f"{deleted.status_code} {deleted.text}")
        assert deleted.status_code == 200, deleted.text
        assertNoServerErrors(server)
