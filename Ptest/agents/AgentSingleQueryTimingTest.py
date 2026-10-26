"""Timing Ptest for an agent: two support agents of one rag, each asked
one question, with where its time went.

The rag and the setup are `AgentStressTest`'s, over HTTP to the server
as its own process. The two agents, FLOWS:

* `guardThenSupport` - the stress test's agent, its guard, then support
  searching the rag:

      input -> guard -> support -> history -> output
                 |
                 +-> output   (turned away)

* `fastSupport` - the TEMPORARY `fast_support_chat` node, the same
  guard and support in one, the guard's pick and the rag search at once:

      input -> fastSupport -> history -> output

The dummy account and project are written to Firestore, and
`Ptest/docs/docs.rar` is indexed from its URL into a rag of the project.
Then, for each agent, in order:

1. The template and its node types are written to Firestore by hand; the
   agent is configured through its endpoint, then given the rag.
2. One warm-up query builds the agent; its time is printed apart.
3. The question - AGENT_TIMING_QUESTION in the environment, or the first
   of `question.json` - is asked once, in a new conversation, and timed
   from its request to its answer.
4. What the server logged from the query's arrival on is printed, each
   line with the seconds since: each model call ends on an `httpx` line,
   the search on its `Queried rag` line, and the answer on the route's.
   Then how long the conversation took to be saved, after the answer.

Both timings are saved to `Ptest/results/`, and everything made is
deleted. The test fails if a query is not answered, or answered empty,
or the server logs an error. A run spends a few DeepSeek and OpenAI
calls, and writes to the Firestore and Pinecone that `test.env` names:

    uv run pytest Ptest/agents/AgentSingleQueryTimingTest.py -s
"""

import json
import os
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
    AGENT_TEMPLATES,
    AGENTS,
    GUARD_PROMPT,
    MODEL,
    NODE_TYPES,
    SUMMARISE_AFTER_TOKENS,
    SUPPORT_PROMPT,
    deleteRecords,
    firestoreClient,
    newId,
    searchesLogged,
    writeRecords,
)
from AgentStressTest import (  # noqa: E402
    GUARD_MODEL,
    SUPPORT_MAX_TOKENS,
    agentRecordsOf,
    configureAgent,
    configureRag,
    deleteConversations,
    waitForConversations,
)
from dummyData import DummyData  # noqa: E402
from RagQualityTest import (  # noqa: E402, F401
    QUESTIONS_PATH,
    RECORD_START,
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

# The question asked, unless the environment names another.
QUESTION_ENV_VAR = "AGENT_TIMING_QUESTION"
# How long the query may wait for its answer.
QUERY_SECONDS = 300
# How a log record's time is written, as `LOG_CONFIG` formats it.
LOG_STAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
# The most of a log record's text printed in the timeline.
TIMELINE_TEXT_LENGTH = 110


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
                "name": "Timing fast support",
                "config": {"model": MODEL},
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
                "name": "Timing chat history",
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
                "name": "Timing fast customer support",
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


# The agents timed, by name: the records each is made of.
FLOWS = {
    "guardThenSupport": agentRecordsOf,
    "fastSupport": fastAgentRecordsOf,
}


def askTimed(
    client: httpx.Client, path: str, question: str, convoId: str
) -> tuple[str, float]:
    """The agent's answer to `question`, and the seconds it took."""
    started = time.perf_counter()
    response = client.post(
        path,
        json={"message": question, "convoId": convoId},
        timeout=QUERY_SECONDS,
    )
    seconds = time.perf_counter() - started
    assert response.status_code == 200, (
        f"{question!r}: {response.status_code} {response.text}"
    )
    answer = response.json()["answer"]
    assert answer.strip(), response.text
    return answer, seconds


def timelineOf(lines: list[str], path: str) -> list[tuple[float, str]]:
    """The server's log records from the query's arrival at `path` on,
    each with the seconds since it arrived, and its logger and text."""
    records = []
    for line in lines:
        if not RECORD_START.match(line):
            continue
        stamp, _, name, text = line.rstrip().split(" | ", 3)
        records.append(
            (datetime.strptime(stamp, LOG_STAMP_FORMAT), f"{name} | {text}")
        )
    arrived = next(
        stamp
        for stamp, text in records
        if f"POST {path} received" in text
    )
    return [
        ((stamp - arrived).total_seconds(), text)
        for stamp, text in records
        if stamp >= arrived
    ]


def timeOneQuery(
    client: httpx.Client,
    server: Server,
    firestoreClient: firestore.Client,
    path: str,
    agentId: str,
    ragId: str,
    question: str,
) -> dict:
    """The agent's timing of `question`, asked at `path`: a warm-up
    query, then the one timed - its answer, the searches of the rag, how
    long its conversation took to be saved after it, and the server's
    timeline of it."""
    _, warmUpSeconds = askTimed(client, path, question, newId())
    checkLog(server, "warm-up")
    linesBefore = server.lineCount
    searchesBefore = searchesLogged(server, ragId)
    answer, seconds = askTimed(client, path, question, newId())
    saved, savedSeconds = waitForConversations(firestoreClient, agentId, 2)
    checkLog(server, "query")
    assert saved == 2, f"{saved} of 2 conversations saved"
    lines = server.logPath.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()[linesBefore:]
    return {
        "question": question,
        "answer": answer,
        "warmUpSeconds": round(warmUpSeconds, 3),
        "seconds": round(seconds, 3),
        "searchesOfTheRag": searchesLogged(server, ragId) - searchesBefore,
        "savedSecondsAfterTheAnswer": savedSeconds,
        "timeline": timelineOf(lines, path),
    }


def printTiming(flow: str, timing: dict) -> None:
    """The query's times, then the server's timeline of it."""
    print(
        f"\n{flow} (guard {GUARD_MODEL}, support {MODEL}, at most "
        f"{SUPPORT_MAX_TOKENS} tokens)\n"
        f"   question:       {timing['question']}\n"
        f"   answer:         {timing['answer']}\n"
        f"   warm-up query:  {timing['warmUpSeconds']:.2f}s\n"
        f"   answer time:    {timing['seconds']:.2f}s\n"
        f"   rag searches:   {timing['searchesOfTheRag']}\n"
        f"   saved:          {timing['savedSecondsAfterTheAnswer']:.2f}s "
        "after the answer\n"
        "   server timeline:"
    )
    for seconds, text in timing["timeline"]:
        print(f"   {seconds:7.3f}s  {text[:TIMELINE_TEXT_LENGTH]}")


def saveTiming(timings: dict) -> Path:
    """Write the timings into `Ptest/results/`, and say where."""
    path = RESULTS_FOLDER / f"agentTiming-{RUN_STAMP}.json"
    path.write_text(
        json.dumps(timings, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


class TestAgentSingleQueryTiming:
    """Each support agent of FLOWS, asked one question over HTTP, and
    timed."""

    def testOneQueryIsAnsweredAndTimed(
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
        agentIds: list[str] = []
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

            question = os.environ.get(
                QUESTION_ENV_VAR, questions[0]["question"]
            )
            timings = {
                "model": MODEL,
                "guardModel": GUARD_MODEL,
                "supportMaxTokens": SUPPORT_MAX_TOKENS,
            }
            for flow, recordsOf in FLOWS.items():
                flowRecords = recordsOf(dummy)
                records.extend(flowRecords)
                writeRecords(firestoreClient, flowRecords)
                templateId = flowRecords[-1][1]["_id"]
                agentId = configureAgent(client, dummy, templateId)
                agentIds.append(agentId)
                configureRag(client, dummy, agentId, ragId)
                print(f"   {flow}: template {templateId}, agent {agentId}")
                checkLog(server, f"configure {flow}")
                timings[flow] = timeOneQuery(
                    client,
                    server,
                    firestoreClient,
                    f"/agents/{dummy.projectId}/{agentId}/query",
                    agentId,
                    ragId,
                    question,
                )
                printTiming(flow, timings[flow])
            timings["finishedAt"] = datetime.now(timezone.utc).isoformat()
            print(f"Timing: {saveTiming(timings)}")
        finally:
            for agentId in agentIds:
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
