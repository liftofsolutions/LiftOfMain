"""Ptest for an agent: a customer support agent, its knowledge base the
test documents, asked ten questions one after another in one
conversation.

Everything goes through the product the way a client would, as in
`RagQualityTest`, whose server, dummy project, document server and judge
this module uses: the server is its own process, reached over HTTP; the
documents are indexed through the rag endpoints; and the agent's records
- its node types, its template and the agent - are written to Firestore
with the Google client, as the backend will write them. Nothing from
`src/` is imported.

The agent's template wires it so:

    input -> guard -> support -> review -> history -> output
                        ^           |
                        +-----------+  three times at most

guard passes a question on to support, or turns it away; support
searches the knowledge base and answers; review passes the answer or
sends it back with what to fix; history keeps the conversation,
summarised past 10000 tokens. Every model is MODEL.

In order:

1. Index `Ptest/docs/docs.rar` into a rag of the dummy project, and wait
   until Pinecone finds its chunks.
2. Write the agent's records, the template naming that rag.
3. Ask QUESTION_IDS of `question.json`, one at a time, in one
   conversation, each once the last is answered. Each answer must come
   back in the conversation, and support must have searched the rag for
   it: the server logs every search of the rag, and the log is read.
4. Have JUDGE_MODEL score each answer from 0 to 10 against what the rag
   holds for its question, through the rag's own query endpoint.
5. Check the conversation's chat history in `agent_memory`: every turn
   kept, or summarised with the last answer kept; print the scores and
   times, and save them to `Ptest/results/`.
6. Delete the agent's records and its conversation, then the rag.

The scores are reported, not held to a pass mark. A run spends real
OpenAI tokens and writes to the Firestore and Pinecone that `test.env`
names:

    uv run pytest Ptest/agents/AgentConversationTest.py -s
"""

import json
import sys
import textwrap
import time
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from google.cloud import firestore
from openai import OpenAI

# The rag Ptest's server and records, which this one shares.
PTEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PTEST_ROOT / "documentProcessing"))

from dummyData import DummyData, openFirestore  # noqa: E402
from RagQualityTest import (  # noqa: E402, F401
    JUDGE_MODEL,
    QUESTIONS_PATH,
    RESULTS_FOLDER,
    RUN_STAMP,
    TOP_K,
    Server,
    askRag,
    averageOf,
    checkLog,
    client,
    describeTiming,
    documentLink,
    dummy,
    indexDocuments,
    judge,
    judgeInputOf,
    questionsOf,
    server,
    timingOf,
    waitUntilSearchable,
)

MODEL = "gpt-5.4-mini"
# The questions asked, in this order, in one conversation.
QUESTION_IDS = [
    "product1.1.1",
    "product1.1.2",
    "product1.2.2",
    "product2.1.2",
    "product4.1.2",
    "warranty.1.1",
    "warranty.1.2",
    "warranty.2.1",
    "generalPolicy.1.1",
    "generalPolicy.1.2",
]
# An agent's answer waits on several models in turn, and on searches.
QUERY_SECONDS = 300
# How long the log may take to show the searches of an answer just given.
LOG_SETTLE_SECONDS = 1

AGENTS = "agents"
AGENT_TEMPLATES = "agent_templates"
NODE_TYPES = "node_types"
AGENT_MEMORY = "agent_memory"
SUMMARISE_AFTER_TOKENS = 10_000

# The guard is told what Aether sells, and which choice support is - its
# choices are node ids - or it turns most of the questions away.
GUARD_PROMPT = (
    "You guard Aether Dynamics' customer support, and never answer a "
    "question yourself. Aether makes phones and tablets - the Pulse, Flow, "
    "Horizon and Terra lines - and their accessories, and sells them with "
    "its own warranty, return and privacy policies. Any question about "
    "these - a spec, a price, the box, a warranty, a return, a policy, an "
    "order - goes on to support: pick the choice described as support, "
    "your message one line saying what is asked. Only for a message with "
    "nothing to do with Aether, pick output, and politely say you can "
    "only help with Aether."
)
SUPPORT_PROMPT = (
    "You are Aether Dynamics' customer support. Always search the "
    "knowledge base before you answer, as often as the question needs, "
    "and answer only from what it holds: every figure and name exactly "
    "as it is there. If it does not hold the answer, say so."
)
REVIEW_PROMPT = (
    "You review support's last reply to the customer against what its "
    "knowledge base searches found. If the reply is right, complete and "
    "polite, pick history; else pick support, and say exactly what to fix."
)

JUDGE_INSTRUCTIONS = (
    "You rate a customer support agent's answer. You are given the "
    "question, passages its knowledge base holds for it, and the agent's "
    "answer. Score from 0 to 10 how correct and complete the answer is, "
    "by the passages:\n"
    "10: every fact the question needs is there, and right.\n"
    "7: the main facts are right; a minor detail is missing.\n"
    "4: only part of the answer is there, or a detail is wrong.\n"
    "1: on the topic, but it does not answer, or is mostly wrong.\n"
    "0: wrong, or not an answer.\n"
    "Take off for any claim the passages do not support. Judge only from "
    "the passages, never from your own knowledge. In the reason, say in "
    "one sentence what is wrong or missing, or that nothing is."
)
ANSWER_RATING_FORMAT = {
    "type": "json_schema",
    "name": "answerRating",
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


def newId() -> str:
    """A fresh UUID v4, as the schemas' ids are."""
    return str(uuid.uuid4())


def agentRecordsOf(dummy: DummyData, ragId: str) -> list[tuple[str, dict]]:
    """The support agent's records, each with its collection: its two
    node types, of the dummy project; its template, searching `ragId`;
    and the agent."""
    now = datetime.now(timezone.utc)
    stamps = {"created_at": now, "updated_at": now}
    chatTypeId, historyTypeId = newId(), newId()
    guardId, supportId, reviewId, historyId = (newId() for _ in range(4))
    templateId = newId()

    def node(nodeId: str, typeId: str, name: str, config: dict) -> dict:
        return {
            "id": nodeId,
            "node_type_id": typeId,
            "name": name,
            "config": config,
            **stamps,
        }

    template = {
        "_id": templateId,
        "project_id": dummy.projectId,
        "name": "Ptest customer support",
        "rag_enabled": True,
        "rag_id": ragId,
        "nodes": [
            node(
                guardId,
                chatTypeId,
                "guard",
                {
                    "system_prompt": GUARD_PROMPT,
                    "description": "Checks the message is one to answer.",
                },
            ),
            node(
                supportId,
                chatTypeId,
                "support",
                {
                    "system_prompt": SUPPORT_PROMPT,
                    "description": "Answers the customer, or answers again.",
                    "rag_search": True,
                },
            ),
            node(
                reviewId,
                chatTypeId,
                "review",
                {
                    "system_prompt": REVIEW_PROMPT,
                    "description": "Checks support's reply.",
                    "pick_limits": {supportId: 3},
                },
            ),
            node(
                historyId,
                historyTypeId,
                "history",
                {
                    "description": "The reply passed: it goes to the customer.",
                    "summarise_after_tokens": SUMMARISE_AFTER_TOKENS,
                },
            ),
        ],
        "edges": [
            {"from": "input", "to": guardId},
            {"from": guardId, "to": supportId},
            {"from": guardId, "to": "output"},
            {"from": supportId, "to": reviewId},
            {"from": reviewId, "to": supportId},
            {"from": reviewId, "to": historyId},
            {"from": historyId, "to": "output"},
        ],
        "triggers": [],
        "is_active": True,
        **stamps,
    }
    return [
        (
            NODE_TYPES,
            {
                "_id": chatTypeId,
                "project_id": dummy.projectId,
                "category": "chat_model",
                "handler": "openai_chat",
                "name": "Ptest OpenAI chat",
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
                "name": "Ptest chat history",
                "config": {"provider": "openai", "model": MODEL},
                "is_active": True,
                **stamps,
            },
        ),
        (AGENT_TEMPLATES, template),
        (
            AGENTS,
            {
                "_id": newId(),
                "project_id": dummy.projectId,
                "template_id": templateId,
                "name": "Ptest support agent",
                "status": "active",
                "created_by": dummy.accountUuid,
                **stamps,
            },
        ),
    ]


@pytest.fixture(scope="module")
def firestoreClient() -> Iterator[firestore.Client]:
    """A client on the Firestore database the server reads."""
    openedClient = openFirestore()
    try:
        yield openedClient
    finally:
        openedClient.close()


def writeRecords(
    firestoreClient: firestore.Client, records: list[tuple[str, dict]]
) -> None:
    """Write `records`, each under its `_id`, in one batch."""
    batch = firestoreClient.batch()
    for collection, record in records:
        document = firestoreClient.collection(collection).document(
            record["_id"]
        )
        batch.set(document, record)
    batch.commit()


def deleteRecords(
    firestoreClient: firestore.Client, documents: list[tuple[str, str]]
) -> None:
    """Delete each of `documents`, by collection and id, in one batch."""
    batch = firestoreClient.batch()
    for collection, documentId in documents:
        document = firestoreClient.collection(collection).document(
            documentId
        )
        batch.delete(document)
    batch.commit()


def searchesLogged(server: Server, ragId: str) -> int:
    """How many searches of the rag the server has logged so far."""
    log = server.logPath.read_text(encoding="utf-8", errors="replace")
    return log.count(f"Queried rag {ragId} ")


def askAgent(
    client: httpx.Client, path: str, question: str, convoId: str
) -> tuple[str, float]:
    """The agent's answer to `question` in the conversation, and how long
    it took."""
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
    reply = response.json()
    assert reply["convoId"] == convoId, reply
    assert reply["answer"].strip(), reply
    return reply["answer"], seconds


def printExchange(question: str, answer: str, seconds: float) -> None:
    """The question asked, the agent's answer, each line indented under
    its own, and how long the answer took."""
    answerLines = answer.replace("\n", "\n   ")
    exchange = f"Q: {question}\nA: {answerLines}\nT: {seconds:.2f}s"
    print(textwrap.indent(exchange, " " * 7))


def rateAnswer(
    judge: OpenAI, question: str, matches: list[dict], answer: str
) -> dict:
    """The judge's score for the answer, by the passages, and its
    reason."""
    response = judge.responses.create(
        model=JUDGE_MODEL,
        instructions=JUDGE_INSTRUCTIONS,
        input=f"{judgeInputOf(question, matches)}\n\nAnswer:\n{answer}",
        reasoning={"effort": "low"},
        text={"format": ANSWER_RATING_FORMAT},
    )
    return json.loads(response.output_text)


def checkChatHistory(chatHistory: dict, answers: list[str]) -> str:
    """Fail unless the history kept the conversation: each question and
    answer, or a summary with the last answer kept. What it holds, in a
    line."""
    messages = chatHistory["messages"]
    assert messages[-1] == {"role": "ai", "text": answers[-1]}, messages[-1]
    if chatHistory["summary"]:
        return (
            f"summarised: {len(chatHistory['summary'])} characters, then "
            f"{len(messages)} messages"
        )
    assert [message["text"] for message in messages[1::2]] == answers
    assert [message["role"] for message in messages] == (
        ["human", "ai"] * len(answers)
    )
    return f"{len(messages)} messages, no summary"


def saveReport(report: dict) -> Path:
    """Write the report into `Ptest/results/`, and say where."""
    path = RESULTS_FOLDER / f"agentConversation-{RUN_STAMP}.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def printReport(report: dict, path: Path) -> None:
    """Each question's score, searches and time; then the averages."""
    print("\n   score  searches  seconds  question")
    for row in report["questions"]:
        print(
            f"   {row['score']:5d}  {row['searches']:8d}  "
            f"{row['seconds']:7.1f}  {row['id']}"
        )
    print(
        f"Average score {report['averageScore']:.2f} / 10 over "
        f"{len(report['questions'])} questions ({report['model']}, judged "
        f"by {report['judgeModel']})\n"
        f"Answer time: {describeTiming(report['seconds'])}\n"
        f"Chat history: {report['chatHistory']}\n"
        f"Report: {path}"
    )


class TestAgentConversation:
    """A support agent, asked ten questions in one conversation, over
    HTTP."""

    def testAgentAnswersTenQuestionsFromTheRag(
        self,
        client: httpx.Client,
        server: Server,
        dummy: DummyData,
        documentLink: str,
        judge: OpenAI,
        firestoreClient: firestore.Client,
    ) -> None:
        ragBase = f"/rag/{dummy.accountUuid}/{dummy.projectId}"
        created = client.post(f"{ragBase}/jobs")
        assert created.status_code == 200, created.text
        job = created.json()
        ragId = job["ragId"]
        print(f"\nJob {job['jobId']}, rag {ragId}")
        records: list[tuple[str, dict]] = []
        convoId = newId()
        try:
            checkLog(server, "create job")
            rag = indexDocuments(client, server, ragBase, job, documentLink)
            byId = {
                row["questionId"]: row for row in questionsOf(QUESTIONS_PATH)
            }
            questions = [byId[questionId] for questionId in QUESTION_IDS]
            waitUntilSearchable(
                client,
                ragBase,
                ragId,
                questions[0]["question"],
                min(TOP_K, rag["chunkCount"]),
            )
            checkLog(server, "search warm-up")

            records = agentRecordsOf(dummy, ragId)
            writeRecords(firestoreClient, records)
            agentId = records[-1][1]["_id"]
            agentPath = f"/agents/{dummy.projectId}/{agentId}/query"
            print(f"   agent {agentId}, conversation {convoId}")

            rows = []
            for number, row in enumerate(questions, 1):
                searchesBefore = searchesLogged(server, ragId)
                answer, seconds = askAgent(
                    client, agentPath, row["question"], convoId
                )
                time.sleep(LOG_SETTLE_SECONDS)
                searches = searchesLogged(server, ragId) - searchesBefore
                print(
                    f"   {number:2d}. {row['questionId']}: {seconds:.1f}s, "
                    f"{searches} searches"
                )
                printExchange(row["question"], answer, seconds)
                checkLog(server, f"question {number}")
                assert searches > 0, (
                    f"{row['questionId']} was answered without a search of "
                    f"the rag: {answer!r}"
                )
                rows.append(
                    {
                        "id": row["questionId"],
                        "question": row["question"],
                        "answer": answer,
                        "searches": searches,
                        "seconds": round(seconds, 3),
                    }
                )

            for row in rows:
                matches = askRag(client, ragBase, ragId, row["question"])
                rating = rateAnswer(
                    judge, row["question"], matches, row["answer"]
                )
                row.update(rating)
            checkLog(server, "judging")

            stored = firestoreClient.collection(AGENT_MEMORY).document(
                f"{agentId}_{convoId}"
            ).get()
            assert stored.exists, "The conversation's history was not saved."
            chatHistory = checkChatHistory(
                stored.to_dict()["chat_history"],
                [row["answer"] for row in rows],
            )
            report = {
                "finishedAt": datetime.now(timezone.utc).isoformat(),
                "model": MODEL,
                "judgeModel": JUDGE_MODEL,
                "rag": rag,
                "averageScore": averageOf(rows),
                "seconds": timingOf(rows, "seconds"),
                "chatHistory": chatHistory,
                "questions": rows,
            }
            printReport(report, saveReport(report))
        finally:
            documents = [
                (collection, record["_id"]) for collection, record in records
            ]
            if records:
                documents.append(
                    (AGENT_MEMORY, f"{records[-1][1]['_id']}_{convoId}")
                )
                deleteRecords(firestoreClient, documents)
            deleted = client.delete(f"{ragBase}/rags/{ragId}")
            if deleted.status_code != 200:
                print(f"   Rag {ragId} was not deleted: "
                      f"{deleted.status_code} {deleted.text}")
        assert deleted.status_code == 200, deleted.text
        checkLog(server, "delete rag")
