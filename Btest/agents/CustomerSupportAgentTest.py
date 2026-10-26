"""Btest for a customer support agent, run as its template wires it:

    input -> guard -> support -> review -> history -> output
                        ^           |
                        +-----------+  three times at most

* input - where a message from the web comes in, as it is.
* guard - a chat model that turns away what support should not answer,
  going out with its own reply, or passes the message on to support.
* support - a chat model that answers the customer, searching the
  agent's knowledge base, the rag its record names, as it needs.
* review - a chat model that passes support's reply, or sends it back
  with what to change: three times at most, by its `pick_limits`,
  counted in its `nodeDetails`.
* history - the chat history node: the turn is kept in the
  conversation's chat history, and past 10000 tokens the history is
  summarised, keeping the summary and the last message.

Firestore is the `FakeFirestore` of `AgentManagerTest`, holding the
agent, its template, their node types, the rag and the conversation's
chat history; Pinecone is `FakePinecone`, which finds RAG_CHUNKS; and
each node's model is a `ScriptedModel`, which answers as its test says.
Everything between is real: `agentStore`, `loadAgent`, the node manager,
the nodes, LangGraph and `searchRag`.

1. A run: the guard turning a message away goes out with its reply,
   and nothing after it runs. A message passed on is answered by
   support, from what it found in the rag; the review passes it, and the
   answer is support's, not the review's.
2. Reviews: the review sends the reply back with what to change, and
   support answers again, reading it; after three, the next reply goes
   on unread, as the review has no other way to send it; and each run
   counts its own three.
3. Chat history: each turn is kept, and the next run's nodes read it.
   Past 10000 tokens, the history is summarised - all but its last
   message, the old summary with it - and keeps that summary and the
   last message; at 10000, it is kept whole. The next run's nodes read
   the summary after their system prompts.
4. Refused as built: a node searching a rag, of an agent with none;
   pick limits that are not counts, name a node with no edge, or limit
   every way on; a node picking that cannot - no chat model, or one
   searching a rag; and a history node with no count of tokens.

    uv run pytest Btest/agents -q
"""

import asyncio
import logging

import pytest

# The fakes and fixtures the agent manager's Btest has: `memoryDb` and
# `firestore` are fixtures of this module too, as imported.
from AgentManagerTest import (  # noqa: F401
    AGENT_ID,
    CHAT_TYPE_ID,
    CONVO_ID,
    PROJECT_ID,
    FakeFirestore,
    SettlingAgentManager,
    runWithin,
    buildAgent,
    buildTemplate,
    chatHistoryIdOf,
    firestore,
    memoryDb,
    storedChatHistory,
)
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from agents.agent import AgentNotRunnableError
from agents.agentManager import AgentManager
from agents.agentStore import (
    AGENT_MEMORY,
    AGENT_TEMPLATES,
    AGENTS,
    NODE_TYPES,
)
from agents.nodeManager import NodeManager
from agents.nodes import INPUT_NODE_ID, OUTPUT_NODE_ID
from agents.nodes import chatHistoryNode
from agents.nodes.chatModelNode import RAG_TOOL_NAME, SUMMARY_ABOUT
from config import ServerConfig
from DbAdapters import QueryMatch
from documentProcessing import service as ragService
from documentProcessing import utilities
from documentProcessing.ragStrategy import (
    PINECONE_TEXT_FIELD,
    RAG_STATUS_READY,
    STRATEGY_SIMPLE_RAG,
)
from LLM import agentChatModel
from LLM.agentChatModel import getSystemPrompt

RAG_ID = "3700255f-a07c-4820-86eb-ba61fb732f77"
HISTORY_TYPE_ID = "b5e1c7d2-4a3f-4e8b-9c6d-2f7a1e0b3d58"

GUARD_PROMPT = "Pass on what Acme's support answers; turn the rest away."
SUPPORT_PROMPT = "You are Acme's support. Answer from the knowledge base."
REVIEW_PROMPT = "Pass a reply that is right and polite; else say what to fix."

QUESTION = "Do you deliver to Pune?"
TURNED_AWAY = "Sorry, I can only help with Acme orders."
PASSED = "A delivery question: passed to support."
SEARCH_QUERY = "delivery to Pune"
RAG_CHUNKS = [
    "Acme delivers across Maharashtra, Pune included.",
    "Deliveries take 2-3 business days.",
]
ANSWER = "Yes, we deliver to Pune in 2-3 business days."
LOOKS_RIGHT = "The reply is right and polite."
SEND_BACK = ("Say how long delivery takes.", "support")
SUMMARY = "The customer asked about a large order to Mumbai."


def searchFor(query: str, callId: str = "search-1") -> AIMessage:
    """A model's answer that searches the knowledge base for `query`."""
    return AIMessage(
        "",
        tool_calls=[
            {"name": RAG_TOOL_NAME, "args": {"query": query}, "id": callId}
        ],
    )


class ScriptedModel:
    """Stands in for a provider's chat model, answering as its `script`
    says, in turn, and keeping everything it was sent.

    A plain call answers with the next message of the script; asked to
    answer in a shape, as a picking node asks, with the next `(message,
    next)` pair, in that shape - so a pick outside the shape's choices
    fails, as a real model's constrained answer cannot be. An error in
    the script is raised instead of answering, and given a `gate`, a
    plain call answers once the gate is set. It keeps each list of
    messages it was sent, each shape, and each binding of tools, with its
    settings.
    """

    def __init__(self) -> None:
        """A model with nothing to say yet, and sent nothing."""
        self.script: list = []
        self.received: list[list] = []
        self.shapes: list[type] = []
        self.bindings: list[tuple[list, dict]] = []
        self.gate: asyncio.Event | None = None

    async def ainvoke(self, messages: list) -> AIMessage:
        """Keep `messages`, and answer with the next of the script - once
        the gate is set, if there is one."""
        self.received.append(list(messages))
        if self.gate is not None:
            await self.gate.wait()
        answer = self.script.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def bind_tools(self, tools: list, **settings) -> "ScriptedModel":
        """The model, the tools bound: kept, with their settings."""
        self.bindings.append((tools, settings))
        return self

    def with_structured_output(self, shape: type) -> "ShapedModel":
        """The model, answering in `shape`."""
        self.shapes.append(shape)
        return ShapedModel(self, shape)

    def choicesOffered(self) -> list[list[str]]:
        """The ids it could pick, in each shape it answered in."""
        return [
            shape.model_json_schema()["properties"]["next"]["enum"]
            for shape in self.shapes
        ]


class ShapedModel:
    """A `ScriptedModel` answering in a shape: the next pair of its
    script."""

    def __init__(self, model: ScriptedModel, shape: type) -> None:
        """`model`, answering in `shape`."""
        self.model = model
        self.shape = shape

    async def ainvoke(self, messages: list) -> object:
        """Keep `messages`, and answer in the shape."""
        self.model.received.append(list(messages))
        message, nextNodeId = self.model.script.pop(0)
        return self.shape(message=message, next=nextNodeId)


class FakePinecone:
    """Stands in for the Pinecone queue: finds RAG_CHUNKS for any query,
    and keeps every search."""

    def __init__(self) -> None:
        """Searched for nothing yet."""
        self.searches: list[tuple[str, str]] = []

    async def search(
        self, namespace: str, text: str, topK: int, fields: list[str]
    ) -> list[QueryMatch]:
        """RAG_CHUNKS, closest first, the search kept."""
        self.searches.append((namespace, text))
        return [
            QueryMatch(
                id=f"{namespace}#{index}",
                score=0.9 - index / 10,
                metadata={PINECONE_TEXT_FIELD: text, "chunk": index},
            )
            for index, text in enumerate(RAG_CHUNKS)
        ]


def buildHistoryNodeType(**overrides) -> dict:
    """A platform node type for the chat history node."""
    nodeType = {
        "_id": HISTORY_TYPE_ID,
        "category": "memory",
        "handler": "chat_history",
        "name": "Chat history",
        "config": {"provider": "openai", "model": "gpt-5.4-mini"},
        "is_active": True,
    }
    nodeType.update(overrides)
    return nodeType


def buildNode(nodeId: str, config: dict, typeId: str = CHAT_TYPE_ID) -> dict:
    """A node of the template, of id and name `nodeId`."""
    return {
        "id": nodeId,
        "node_type_id": typeId,
        "name": nodeId,
        "config": config,
    }


def buildSupportNodes(**configs) -> list[dict]:
    """The guard, support, review and history nodes, each config updated
    as `configs` says, by node id."""
    nodes = [
        buildNode(
            "guard",
            {
                "system_prompt": GUARD_PROMPT,
                "description": "Checks the message is one to answer.",
            },
        ),
        buildNode(
            "support",
            {
                "system_prompt": SUPPORT_PROMPT,
                "description": "Answers the customer, or answers again.",
                "rag_search": True,
            },
        ),
        buildNode(
            "review",
            {
                "system_prompt": REVIEW_PROMPT,
                "description": "Checks support's reply.",
                "pick_limits": {"support": 3},
            },
        ),
        buildNode(
            "history",
            {"description": "The reply passed: it goes to the customer."},
            HISTORY_TYPE_ID,
        ),
    ]
    for node in nodes:
        node["config"].update(configs.get(node["id"], {}))
    return nodes


# The support agent's graph.
SUPPORT_EDGES = [
    {"from": INPUT_NODE_ID, "to": "guard"},
    {"from": "guard", "to": "support"},
    {"from": "guard", "to": OUTPUT_NODE_ID},
    {"from": "support", "to": "review"},
    {"from": "review", "to": "support"},
    {"from": "review", "to": "history"},
    {"from": "history", "to": OUTPUT_NODE_ID},
]


def buildSupportTemplate(**overrides) -> dict:
    """The support agent's template: its nodes, wired by SUPPORT_EDGES."""
    template = {"nodes": buildSupportNodes(), "edges": SUPPORT_EDGES}
    template.update(overrides)
    return buildTemplate(**template)


def buildRag(**overrides) -> dict:
    """The project's rag, ready, searched in Pinecone."""
    rag = {
        "_id": RAG_ID,
        "project_id": PROJECT_ID,
        "namespace": RAG_ID,
        "status": RAG_STATUS_READY,
        "strategy": STRATEGY_SIMPLE_RAG,
    }
    rag.update(overrides)
    return rag


def putChatHistory(firestore: FakeFirestore, chatHistory: dict) -> None:
    """The conversation's chat history, as an earlier answer saved it."""
    firestore.records[AGENT_MEMORY][chatHistoryIdOf(CONVO_ID)] = {
        "agent_id": AGENT_ID,
        "project_id": PROJECT_ID,
        "convo_id": CONVO_ID,
        "chat_history": chatHistory,
    }


def textsOf(messages: list) -> list[str]:
    """The text of each of `messages`."""
    return [message.text for message in messages]


def passedAndAnswered(answer: str = ANSWER) -> dict[str, list]:
    """The scripts of a run the guard passes on, support answers without
    searching, and the review passes; the history, if it summarises,
    summarised as SUMMARY."""
    return {
        "guard": [(PASSED, "support")],
        "support": [AIMessage(answer)],
        "review": [(LOOKS_RIGHT, "history")],
        "history": [AIMessage(SUMMARY)],
    }


@pytest.fixture
def pinecone(monkeypatch) -> FakePinecone:
    """Pinecone, finding RAG_CHUNKS; the rag's search settings are the
    defaults."""
    fakePinecone = FakePinecone()
    monkeypatch.setattr(ragService, "getPineconeDb", lambda: fakePinecone)
    monkeypatch.setattr(ragService, "getServerConfig", ServerConfig)
    return fakePinecone


@pytest.fixture
def supportManager(monkeypatch, firestore, pinecone) -> AgentManager:
    """A manager with no agent kept, the support agent's records in
    Firestore - the agent searching RAG_ID - and every model a
    `ScriptedModel`."""
    firestore.put(AGENTS, buildAgent(rag_enabled=True, rag_id=RAG_ID))
    firestore.put(AGENT_TEMPLATES, buildSupportTemplate())
    firestore.put(NODE_TYPES, buildHistoryNodeType())
    firestore.put(utilities.RAGS, buildRag())
    serverConfig = ServerConfig(llmApiKeys={"openai": "platform-test-key"})
    monkeypatch.setattr(
        agentChatModel, "getServerConfig", lambda: serverConfig
    )
    monkeypatch.setattr(
        agentChatModel,
        "init_chat_model",
        lambda model, **settings: ScriptedModel(),
    )
    return SettlingAgentManager(NodeManager())


def modelsOf(agentManager: AgentManager) -> dict[str, ScriptedModel]:
    """The model of each node of the support agent, built if it is not
    kept, by node id."""
    agent = asyncio.run(agentManager.getAgent(PROJECT_ID, AGENT_ID))
    return {node.nodeId: node.chatModel.llm for node in agent.nodes}


def ask(
    agentManager: AgentManager,
    scripts: dict[str, list],
    message: str = QUESTION,
) -> tuple[str, dict[str, ScriptedModel]]:
    """The support agent's answer to `message` in the conversation, each
    node's model answering as `scripts` says, by node id; and the models,
    each script left as the run left it."""
    models = modelsOf(agentManager)
    for nodeId, model in models.items():
        model.script = list(scripts.get(nodeId, []))
        model.received.clear()
        model.shapes.clear()
        model.bindings.clear()
    reply = asyncio.run(
        agentManager.query(PROJECT_ID, AGENT_ID, message, CONVO_ID)
    )
    return reply.answer, models


class TestRun:
    """A message turned away, or answered from the rag and passed."""

    def testGuardTurnsAMessageAwayAndNothingAfterRuns(
        self, supportManager, firestore, pinecone
    ):
        answer, models = ask(
            supportManager, {"guard": [(TURNED_AWAY, OUTPUT_NODE_ID)]}
        )

        assert answer == TURNED_AWAY
        assert models["guard"].choicesOffered() == [
            ["support", OUTPUT_NODE_ID]
        ]
        ran = [nodeId for nodeId, model in models.items() if model.received]
        assert ran == ["guard"]
        assert pinecone.searches == []
        [guardSent] = models["guard"].received
        assert textsOf(guardSent) == [GUARD_PROMPT, QUESTION]
        assert storedChatHistory(firestore, CONVO_ID)["chat_history"] == {}

    def testSupportAnswersFromTheRagAndItsAnswerGoesOut(
        self, supportManager, pinecone
    ):
        answer, models = ask(
            supportManager,
            {
                "guard": [(PASSED, "support")],
                "support": [searchFor(SEARCH_QUERY), AIMessage(ANSWER)],
                "review": [(LOOKS_RIGHT, "history")],
            },
        )

        assert answer == ANSWER
        assert pinecone.searches == [(RAG_ID, SEARCH_QUERY)]
        support = models["support"]
        [(tools, settings)] = support.bindings
        assert [tool["function"]["name"] for tool in tools] == [RAG_TOOL_NAME]
        assert settings == {}
        firstSent, secondSent = support.received
        assert textsOf(firstSent) == [SUPPORT_PROMPT, QUESTION, PASSED]
        found = secondSent[-1]
        assert isinstance(found, ToolMessage)
        assert found.tool_call_id == "search-1"
        assert found.text == "\n\n".join(RAG_CHUNKS)
        [reviewSent] = models["review"].received
        assert textsOf(reviewSent) == [
            REVIEW_PROMPT,
            QUESTION,
            PASSED,
            "",
            "\n\n".join(RAG_CHUNKS),
            ANSWER,
        ]
        assert models["review"].choicesOffered() == [["support", "history"]]

    def testSearchOfARagNotReadyTellsTheModelSo(
        self, supportManager, firestore, pinecone
    ):
        firestore.put(utilities.RAGS, buildRag(status="indexing"))

        answer, models = ask(
            supportManager,
            {
                "guard": [(PASSED, "support")],
                "support": [searchFor(SEARCH_QUERY), AIMessage(ANSWER)],
                "review": [(LOOKS_RIGHT, "history")],
            },
        )

        assert answer == ANSWER
        assert pinecone.searches == []
        found = models["support"].received[1][-1]
        assert "only a ready rag can be queried" in found.text

    def testRagOfAnotherProjectIsNotSearched(
        self, supportManager, firestore, pinecone
    ):
        firestore.put(utilities.RAGS, buildRag(project_id="someone-else"))

        _, models = ask(
            supportManager,
            {
                "guard": [(PASSED, "support")],
                "support": [searchFor(SEARCH_QUERY), AIMessage(ANSWER)],
                "review": [(LOOKS_RIGHT, "history")],
            },
        )

        assert pinecone.searches == []
        found = models["support"].received[1][-1]
        assert f"has no rag {RAG_ID!r}" in found.text

    def testModelStillSearchingIsAskedToAnswerWithWhatItHas(
        self, supportManager, pinecone
    ):
        searches = [searchFor(f"query {index}") for index in range(3)]

        answer, models = ask(
            supportManager,
            {
                "guard": [(PASSED, "support")],
                "support": [*searches, AIMessage(ANSWER)],
                "review": [(LOOKS_RIGHT, "history")],
            },
        )

        assert answer == ANSWER
        assert len(pinecone.searches) == 3
        settings = [settings for _, settings in models["support"].bindings]
        assert settings == [{}, {"tool_choice": "none"}]


class TestReviews:
    """The review sending support's reply back, three times at most."""

    def testSupportAnswersAgainReadingWhatToChange(self, supportManager):
        answer, models = ask(
            supportManager,
            {
                "guard": [(PASSED, "support")],
                "support": [AIMessage("We might."), AIMessage(ANSWER)],
                "review": [
                    ("Say how long delivery takes.", "support"),
                    (LOOKS_RIGHT, "history"),
                ],
            },
        )

        assert answer == ANSWER
        lastSent = models["support"].received[-1]
        assert textsOf(lastSent)[-2:] == [
            "We might.",
            "Say how long delivery takes.",
        ]

    def testAfterThreeReviewsTheReplyGoesOnUnread(
        self, supportManager, firestore
    ):
        drafts = [AIMessage(f"Draft {index}.") for index in range(4)]

        answer, models = ask(
            supportManager,
            {
                "guard": [(PASSED, "support")],
                "support": drafts,
                "review": [SEND_BACK] * 3,
            },
        )

        assert answer == "Draft 3."
        assert len(models["support"].received) == 4
        assert models["review"].choicesOffered() == [
            ["support", "history"]
        ] * 3
        assert len(models["review"].received) == 3
        stored = storedChatHistory(firestore, CONVO_ID)["chat_history"]
        assert stored["messages"][-1] == {"role": "ai", "text": "Draft 3."}

    def testEachRunCountsItsOwnReviews(self, supportManager):
        scripts = {
            "guard": [(PASSED, "support")],
            "support": [AIMessage(ANSWER)] * 4,
            "review": [SEND_BACK] * 3,
        }
        ask(supportManager, scripts)

        _, models = ask(supportManager, scripts)

        assert len(models["support"].received) == 4
        assert models["review"].choicesOffered() == [
            ["support", "history"]
        ] * 3


class TestChatHistory:
    """Each turn kept, read by the next run, and summarised when long."""

    def testTurnIsKeptAndTheNextRunReadsIt(self, supportManager, firestore):
        ask(supportManager, passedAndAnswered())

        stored = storedChatHistory(firestore, CONVO_ID)["chat_history"]
        assert stored == {
            "summary": None,
            "messages": [
                {"role": "human", "text": QUESTION},
                {"role": "ai", "text": ANSWER},
            ],
        }

        _, models = ask(
            supportManager,
            passedAndAnswered("Two to three days."),
            "How long?",
        )

        [supportSent] = models["support"].received
        assert textsOf(supportSent) == [
            SUPPORT_PROMPT,
            QUESTION,
            ANSWER,
            "How long?",
            PASSED,
        ]
        stored = storedChatHistory(firestore, CONVO_ID)["chat_history"]
        assert [message["text"] for message in stored["messages"]] == [
            QUESTION,
            ANSWER,
            "How long?",
            "Two to three days.",
        ]

    def testHistoryPastTenThousandTokensKeepsItsSummaryAndLastMessage(
        self, supportManager, firestore
    ):
        earlier = [
            {"role": "human", "text": "Can you ship 500 units? " * 1000},
            {"role": "ai", "text": "Yes, to Mumbai, in a week. " * 1000},
        ]
        putChatHistory(firestore, {"summary": None, "messages": earlier})
        earlierTokens = chatHistoryNode.countTokens(
            chatHistoryNode.transcriptOf(None, earlier)
        )
        assert earlierTokens > 10_000

        answer, models = ask(supportManager, passedAndAnswered())

        assert answer == ANSWER
        stored = storedChatHistory(firestore, CONVO_ID)["chat_history"]
        assert stored == {
            "summary": SUMMARY,
            "messages": [{"role": "ai", "text": ANSWER}],
        }
        [summarySent] = models["history"].received
        systemPrompt, transcript = summarySent
        assert systemPrompt.text == getSystemPrompt(
            "summarizer", "openai", "gpt-5.4-mini"
        )
        assert transcript.text.endswith(f"human: {QUESTION}")
        assert "ai: Yes, to Mumbai" in transcript.text
        assert ANSWER not in transcript.text

    def testSummaryHoldsTheOldSummary(
        self, monkeypatch, supportManager, firestore
    ):
        putChatHistory(firestore, {"summary": "Earlier talk.", "messages": []})
        monkeypatch.setattr(chatHistoryNode, "countTokens", lambda text: 10_001)

        _, models = ask(supportManager, passedAndAnswered())

        [summarySent] = models["history"].received
        assert summarySent[1].text.splitlines()[-2:] == [
            "Summary so far: Earlier talk.",
            f"human: {QUESTION}",
        ]

    @pytest.mark.parametrize(
        ("tokens", "summarised"), [(10_000, False), (10_001, True)]
    )
    def testHistoryIsSummarisedOnlyPastTenThousandTokens(
        self, monkeypatch, supportManager, firestore, tokens, summarised
    ):
        counted = []

        def countTokens(text: str) -> int:
            counted.append(text)
            return tokens

        monkeypatch.setattr(chatHistoryNode, "countTokens", countTokens)

        _, models = ask(supportManager, passedAndAnswered())

        assert counted == [f"human: {QUESTION}\nai: {ANSWER}"]
        assert len(models["history"].received) == (1 if summarised else 0)
        stored = storedChatHistory(firestore, CONVO_ID)["chat_history"]
        assert (stored["summary"] == SUMMARY) is summarised
        assert len(stored["messages"]) == (1 if summarised else 2)

    def testNodesReadTheSummaryAfterTheirPrompts(
        self, supportManager, firestore
    ):
        putChatHistory(
            firestore,
            {"summary": SUMMARY, "messages": [{"role": "ai", "text": ANSWER}]},
        )

        _, models = ask(supportManager, passedAndAnswered(), "And Pune?")

        for nodeId in ("guard", "support", "review"):
            systemPrompt, *messages = models[nodeId].received[0]
            assert isinstance(systemPrompt, SystemMessage)
            assert systemPrompt.text.endswith(f"{SUMMARY_ABOUT}\n{SUMMARY}")
            assert textsOf(messages)[:2] == [ANSWER, "And Pune?"]

    def testAnswerGoesBackBeforeTheHistoryIsSummarisedAndSaved(
        self, monkeypatch, supportManager, firestore
    ):
        monkeypatch.setattr(chatHistoryNode, "countTokens", lambda text: 10_001)
        models = modelsOf(supportManager)
        for nodeId, script in passedAndAnswered().items():
            models[nodeId].script = list(script)
        summarising = asyncio.Event()
        models["history"].gate = summarising

        async def askThenLetItSummarise() -> tuple:
            reply = await AgentManager.query(
                supportManager, PROJECT_ID, AGENT_ID, QUESTION, CONVO_ID
            )
            savedAtReply = storedChatHistory(firestore, CONVO_ID)
            summarising.set()
            await supportManager.settle()
            return reply, savedAtReply

        reply, savedAtReply = runWithin(askThenLetItSummarise())

        assert reply.answer == ANSWER
        assert savedAtReply is None
        stored = storedChatHistory(firestore, CONVO_ID)["chat_history"]
        assert stored == {
            "summary": SUMMARY,
            "messages": [{"role": "ai", "text": ANSWER}],
        }

    def testRunFailingAfterItsAnswerIsLoggedAndSavesNothing(
        self, monkeypatch, supportManager, firestore, caplog
    ):
        monkeypatch.setattr(chatHistoryNode, "countTokens", lambda text: 10_001)
        scripts = passedAndAnswered()
        scripts["history"] = [RuntimeError("The summarizer is down.")]

        with caplog.at_level(logging.ERROR, logger="agents.agent"):
            answer, _ = ask(supportManager, scripts)

        assert answer == ANSWER
        assert "then failed; its chat history is not saved" in caplog.text
        assert "The summarizer is down." in caplog.text
        assert storedChatHistory(firestore, CONVO_ID) is None


def withConfig(nodeId: str, **config) -> list[tuple[str, dict]]:
    """The support template, the node `nodeId`'s config updated with
    `config`."""
    nodes = buildSupportNodes(**{nodeId: config})
    return [(AGENT_TEMPLATES, buildSupportTemplate() | {"nodes": nodes})]


# What a change of records does to a support agent that cannot then be
# built, and a part of the refusal's message.
NOT_BUILT_CASES = {
    "ragSearchWithNoRag": (
        [(AGENTS, buildAgent())],
        "searches a rag, and agent",
    ),
    "ragNotEnabled": (
        [(AGENTS, buildAgent(rag_enabled=False, rag_id=RAG_ID))],
        "searches a rag, and agent",
    ),
    "pickLimitNotACount": (
        withConfig("review", pick_limits={"support": "3"}),
        "pick limits that are not counts",
    ),
    "pickLimitOfNone": (
        withConfig("review", pick_limits={"support": 0}),
        "pick limits that are not counts",
    ),
    "pickLimitsNotByNode": (
        withConfig("review", pick_limits=[3]),
        "pick limits that are not counts",
    ),
    "pickLimitOfANodeWithNoEdge": (
        withConfig("review", pick_limits={"guard": 3}),
        "limits its picks of 'guard', to which it has no edge",
    ),
    "pickLimitsOfEveryWayOn": (
        withConfig("review", pick_limits={"support": 3, "history": 1}),
        "limits its picks of every node it goes on to",
    ),
    "pickLimitOfTheOneWayOn": (
        withConfig("support", pick_limits={"review": 3}),
        "limits its picks of every node it goes on to",
    ),
    "pickingNodeSearchesARag": (
        withConfig("review", rag_search=True),
        "a node that picks does not search a rag",
    ),
    "historyWithNoCountOfTokens": (
        withConfig("history", summarise_after_tokens=0),
        "summarise_after_tokens is 0",
    ),
    "historyOfAModelNotListed": (
        withConfig("history", model="gpt-unlisted"),
        "cannot be reserved: 'gpt-unlisted'",
    ),
}


class TestNotBuilt:
    """Support agents that cannot be built as their records say."""

    @pytest.mark.parametrize(
        ("changes", "refusal"),
        NOT_BUILT_CASES.values(),
        ids=NOT_BUILT_CASES.keys(),
    )
    def testAgentIsRefusedWithNoNodeLeftReserved(
        self, supportManager, firestore, changes, refusal
    ):
        for collection, record in changes:
            firestore.put(collection, record)

        with pytest.raises(AgentNotRunnableError, match=refusal):
            asyncio.run(supportManager.getAgent(PROJECT_ID, AGENT_ID))

        assert supportManager.nodeManager.reservedNodes == set()

    def testPickingNodeWithNoModelIsRefused(self, supportManager, firestore):
        edges = [
            *SUPPORT_EDGES,
            {"from": "history", "to": "support"},
        ]
        firestore.put(AGENT_TEMPLATES, buildSupportTemplate(edges=edges))

        with pytest.raises(AgentNotRunnableError, match="no model to pick"):
            asyncio.run(supportManager.getAgent(PROJECT_ID, AGENT_ID))

    def testRagTheConfigNamesIsNotTheOneSearched(
        self, supportManager, firestore
    ):
        sneaked = {"project_id": "someone-else", "rag_id": "their-rag"}
        nodes = buildSupportNodes(
            support={"rag": sneaked}, review={"rag": sneaked}
        )
        firestore.put(
            AGENT_TEMPLATES, buildSupportTemplate() | {"nodes": nodes}
        )

        agent = asyncio.run(supportManager.getAgent(PROJECT_ID, AGENT_ID))

        configs = {node.nodeId: node.config for node in agent.nodes}
        assert configs["support"]["rag"] == {
            "project_id": PROJECT_ID,
            "rag_id": RAG_ID,
        }
        assert "rag" not in configs["review"]
