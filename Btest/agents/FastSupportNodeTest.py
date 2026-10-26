"""Btest for `agents.nodes.fastSupportNode`, the TEMPORARY node that is a
guard and support in one, run as its template wires it:

    input -> fast -> history -> output

Firestore, Pinecone and the models are `CustomerSupportAgentTest`'s
fakes: each model a `ScriptedModel`, Pinecone finding RAG_CHUNKS.
Everything between is real: `loadAgent`, the node manager, the node,
the chat history node, LangGraph and `searchRag`.

1. The guard's pick and the search of the message run at once: neither
   goes on until the other has started.
2. A message passed on is answered by support in one call, from the
   search added to the conversation as its own; support may search
   again. The guard is sent its own prompt, and the message.
3. A message turned away goes out with the guard's message; support is
   not asked, and the turn is kept.
4. The guard runs on DeepSeek with its thinking off; a node with no
   guard is not built; and a released node holds no model.

    uv run pytest Btest/agents -q
"""

import asyncio

import pytest
from AgentManagerTest import (  # noqa: F401
    AGENT_ID,
    CONVO_ID,
    PROJECT_ID,
    SettlingAgentManager,
    buildAgent,
    buildTemplate,
    firestore,
    memoryDb,
    runWithin,
    storedChatHistory,
)
from CustomerSupportAgentTest import (  # noqa: F401
    ANSWER,
    GUARD_PROMPT,
    PASSED,
    QUESTION,
    RAG_CHUNKS,
    RAG_ID,
    SUPPORT_PROMPT,
    TURNED_AWAY,
    ScriptedModel,
    buildHistoryNodeType,
    buildNode,
    buildRag,
    pinecone,
    searchFor,
    textsOf,
)
from langchain_core.messages import AIMessage, ToolMessage

from agents.agent import AgentNotRunnableError
from agents.agentManager import AgentManager
from agents.agentStore import AGENT_TEMPLATES, AGENTS, NODE_TYPES
from agents.nodeManager import NodeManager
from agents.nodes import INPUT_NODE_ID, OUTPUT_NODE_ID
from agents.nodes.fastSupportNode import EARLY_SEARCH_ID, FastSupportNode
from config import ServerConfig
from documentProcessing import utilities
from LLM import agentChatModel

FAST_TYPE_ID = "c4a9e2b7-1d6f-4b3a-8e5c-7f2d9a0b6e14"
HISTORY_TYPE_ID = "b5e1c7d2-4a3f-4e8b-9c6d-2f7a1e0b3d58"
GUARD_MODEL = "deepseek-v4-flash"

FAST_EDGES = [
    {"from": INPUT_NODE_ID, "to": "fast"},
    {"from": "fast", "to": "history"},
    {"from": "history", "to": OUTPUT_NODE_ID},
]


def buildFastNodeType(**overrides) -> dict:
    """A platform node type for the fast support node."""
    nodeType = {
        "_id": FAST_TYPE_ID,
        "category": "chat_model",
        "handler": "fast_support_chat",
        "name": "Fast support",
        "config": {"model": "gpt-5.5"},
        "is_active": True,
    }
    nodeType.update(overrides)
    return nodeType


def buildFastTemplate(**fastConfig) -> dict:
    """The fast support agent's template: the fast node, its config
    updated with `fastConfig`, and the history node."""
    config = {
        "system_prompt": SUPPORT_PROMPT,
        "description": "Answers the customer.",
        "rag_search": True,
        "guard": {"model": GUARD_MODEL, "system_prompt": GUARD_PROMPT},
    }
    config.update(fastConfig)
    return buildTemplate(
        nodes=[
            buildNode("fast", config, FAST_TYPE_ID),
            buildNode(
                "history", {"description": "Keeps it."}, HISTORY_TYPE_ID
            ),
        ],
        edges=FAST_EDGES,
    )


class MeetingPinecone:
    """Wraps the fake Pinecone: a search goes on once the guard has been
    asked too, at the barrier they share."""

    def __init__(self, pinecone) -> None:
        """`pinecone`, meeting the guard at no barrier yet."""
        self.pinecone = pinecone
        self.barrier: asyncio.Barrier | None = None

    async def search(self, *args) -> list:
        """Wait at the barrier, if there is one, then search."""
        if self.barrier is not None:
            await self.barrier.wait()
        return await self.pinecone.search(*args)


class MeetingModel(ScriptedModel):
    """A `ScriptedModel` whose shaped answer waits at the barrier first,
    until the search has started too."""

    def __init__(self) -> None:
        """A model meeting at no barrier yet."""
        super().__init__()
        self.barrier: asyncio.Barrier | None = None

    def with_structured_output(self, shape: type) -> "MeetingShapedModel":
        """The model, answering in `shape` once both meet."""
        self.shapes.append(shape)
        return MeetingShapedModel(self, shape)


class MeetingShapedModel:
    """A `MeetingModel` answering in a shape, after the barrier."""

    def __init__(self, model: MeetingModel, shape: type) -> None:
        """`model`, answering in `shape`."""
        self.model = model
        self.shape = shape

    async def ainvoke(self, messages: list) -> object:
        """Keep `messages`, wait at the barrier, and answer in the
        shape."""
        self.model.received.append(list(messages))
        if self.model.barrier is not None:
            await self.model.barrier.wait()
        message, nextNodeId = self.model.script.pop(0)
        return self.shape(message=message, next=nextNodeId)


@pytest.fixture
def builds(monkeypatch) -> list[tuple[str, dict]]:
    """Every model built, with its settings; each a `MeetingModel`."""
    built: list[tuple[str, dict]] = []

    def initChatModel(model: str, **settings) -> MeetingModel:
        built.append((model, settings))
        return MeetingModel()

    serverConfig = ServerConfig(
        llmApiKeys={"openai": "platform-test-key", "deepseek": "test-key"}
    )
    monkeypatch.setattr(
        agentChatModel, "getServerConfig", lambda: serverConfig
    )
    monkeypatch.setattr(agentChatModel, "init_chat_model", initChatModel)
    return built


@pytest.fixture
def fastManager(firestore, pinecone, builds) -> AgentManager:
    """A manager with no agent kept, the fast support agent's records in
    Firestore - the agent searching RAG_ID."""
    firestore.put(AGENTS, buildAgent(rag_enabled=True, rag_id=RAG_ID))
    firestore.put(AGENT_TEMPLATES, buildFastTemplate())
    firestore.put(NODE_TYPES, buildFastNodeType())
    firestore.put(NODE_TYPES, buildHistoryNodeType())
    firestore.put(utilities.RAGS, buildRag())
    return SettlingAgentManager(NodeManager())


def fastNodeOf(agentManager: AgentManager) -> FastSupportNode:
    """The fast node of the agent, built if it is not kept."""
    agent = asyncio.run(agentManager.getAgent(PROJECT_ID, AGENT_ID))
    [node] = [node for node in agent.nodes if node.nodeId == "fast"]
    return node


def ask(
    agentManager: AgentManager,
    guardScript: list,
    supportScript: list,
    meeting: MeetingPinecone | None = None,
) -> tuple[str, ScriptedModel, ScriptedModel]:
    """The agent's answer to QUESTION in the conversation, the guard and
    support answering as their scripts say; and the guard's model and
    support's. Given `meeting`, the guard and the search wait for each
    other."""
    node = fastNodeOf(agentManager)
    guard, support = node.guardModel.llm, node.chatModel.llm
    guard.script, support.script = list(guardScript), list(supportScript)

    async def query() -> str:
        if meeting is not None:
            meeting.barrier = guard.barrier = asyncio.Barrier(2)
        reply = await agentManager.query(
            PROJECT_ID, AGENT_ID, QUESTION, CONVO_ID
        )
        return reply.answer

    return runWithin(query()), guard, support


class TestAtOnce:
    """The guard's pick and the search, at the same time."""

    def testGuardAndSearchEachWaitForTheOtherToStart(
        self, fastManager, pinecone, monkeypatch
    ):
        meeting = MeetingPinecone(pinecone)
        monkeypatch.setattr(
            "documentProcessing.service.getPineconeDb", lambda: meeting
        )

        answer, _, _ = ask(
            fastManager,
            [(PASSED, "support")],
            [AIMessage(ANSWER)],
            meeting,
        )

        assert answer == ANSWER
        assert pinecone.searches == [(RAG_ID, QUESTION)]


class TestAnswer:
    """A message passed on, answered from the search; or turned away."""

    def testPassedMessageIsAnsweredInOneCallFromTheSearch(
        self, fastManager, firestore, pinecone
    ):
        answer, guard, support = ask(
            fastManager, [(PASSED, "support")], [AIMessage(ANSWER)]
        )

        assert answer == ANSWER
        assert pinecone.searches == [(RAG_ID, QUESTION)]
        [guardSent] = guard.received
        assert textsOf(guardSent) == [GUARD_PROMPT, QUESTION]
        assert guard.choicesOffered() == [["support", OUTPUT_NODE_ID]]
        [supportSent] = support.received
        assert textsOf(supportSent) == [
            SUPPORT_PROMPT,
            QUESTION,
            "",
            "\n\n".join(RAG_CHUNKS),
        ]
        searched, found = supportSent[-2:]
        assert searched.tool_calls[0]["args"] == {"query": QUESTION}
        assert isinstance(found, ToolMessage)
        assert found.tool_call_id == EARLY_SEARCH_ID
        stored = storedChatHistory(firestore, CONVO_ID)["chat_history"]
        assert stored["messages"][-1] == {"role": "ai", "text": ANSWER}

    def testSupportMaySearchAgain(self, fastManager, pinecone):
        answer, _, support = ask(
            fastManager,
            [(PASSED, "support")],
            [searchFor("delivery times"), AIMessage(ANSWER)],
        )

        assert answer == ANSWER
        assert pinecone.searches == [
            (RAG_ID, QUESTION),
            (RAG_ID, "delivery times"),
        ]
        assert len(support.received) == 2

    def testTurnedAwayMessageGoesOutWithTheGuardsMessage(
        self, fastManager, firestore
    ):
        answer, _, support = ask(
            fastManager, [(TURNED_AWAY, OUTPUT_NODE_ID)], []
        )

        assert answer == TURNED_AWAY
        assert support.received == []
        stored = storedChatHistory(firestore, CONVO_ID)["chat_history"]
        assert stored["messages"] == [
            {"role": "human", "text": QUESTION},
            {"role": "ai", "text": TURNED_AWAY},
        ]


class TestBuilt:
    """The guard's model, a node with no guard, and a released node."""

    def testGuardRunsOnDeepSeekWithItsThinkingOff(
        self, fastManager, builds
    ):
        fastNodeOf(fastManager)

        [guardSettings] = [
            settings for model, settings in builds if model == GUARD_MODEL
        ]
        assert guardSettings["model_provider"] == "deepseek"
        assert guardSettings["extra_body"] == {
            "thinking": {"type": "disabled"}
        }

    def testNodeWithNoGuardIsNotBuilt(self, fastManager, firestore):
        template = buildFastTemplate()
        del template["nodes"][0]["config"]["guard"]
        firestore.put(AGENT_TEMPLATES, template)

        with pytest.raises(AgentNotRunnableError, match="cannot be reserved"):
            fastNodeOf(fastManager)

    def testReleasedNodeHoldsNoModel(self, fastManager):
        node = fastNodeOf(fastManager)

        node.deconfigure()

        assert node.guardModel is None
        assert node.chatModel is None
        assert node.config == {}
