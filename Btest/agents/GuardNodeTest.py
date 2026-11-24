"""Btest for `agents.nodes.guardNode`: a customer agent's guardrails,
which search the rag for the message as they pick, run as the quality
Ptest's agent wires it:

    input -> guard -> support -> output
               |
               +-> output   (turned away)

Firestore, Pinecone and the models are `CustomerSupportAgentTest`'s
fakes: each model a `ScriptedModel`, Pinecone finding RAG_CHUNKS.
Everything between is real: `loadAgent`, the node manager, the guard,
the support chat node, LangGraph and `searchRag`.

1. The guard's pick and its search of the message run at once: neither
   goes on until the other has started. The search goes on while the
   conversation's chat history loads; the pick waits for it, and reads
   it.
2. A message passed on is answered by support from the guard's search,
   added to the conversation as support's own first search; support may
   search again. The guard is sent its own prompt and the message, and
   picks support or the way out.
3. A message turned away goes out with the guard's reply; support is not
   asked, and the search is dropped.
4. A guard with no rag is refused as it is built.

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
)
from CustomerSupportAgentTest import (  # noqa: F401
    ANSWER,
    putChatHistory,
    GUARD_PROMPT,
    PASSED,
    QUESTION,
    RAG_CHUNKS,
    RAG_ID,
    SUPPORT_PROMPT,
    TURNED_AWAY,
    ScriptedModel,
    buildNode,
    buildRag,
    pinecone,
    searchFor,
    textsOf,
)
from langchain_core.messages import AIMessage, ToolMessage

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
from agents.nodes.guardNode import GUARD_SEARCH_ID
from config import ServerConfig
from documentProcessing import utilities
from LLM import agentChatModel

GUARD_TYPE_ID = "6d2f8a41-3c7e-4b95-a0d8-1e5f9c2b7a63"

GUARD_EDGES = [
    {"from": INPUT_NODE_ID, "to": "guard"},
    {"from": "guard", "to": "support"},
    {"from": "guard", "to": OUTPUT_NODE_ID},
    {"from": "support", "to": OUTPUT_NODE_ID},
]


def buildGuardNodeType(**overrides) -> dict:
    """A platform node type for the guard."""
    nodeType = {
        "_id": GUARD_TYPE_ID,
        "category": "chat_model",
        "handler": "openai_guard",
        "name": "Guard",
        "config": {"model": "gpt-5.5"},
        "is_active": True,
    }
    nodeType.update(overrides)
    return nodeType


def buildGuardTemplate(**guardConfig) -> dict:
    """The agent's template: the guard, its config updated with
    `guardConfig`, and support, wired by GUARD_EDGES."""
    config = {
        "system_prompt": GUARD_PROMPT,
        "description": "Checks the message is one to answer.",
        "rag_search": True,
    }
    config.update(guardConfig)
    return buildTemplate(
        nodes=[
            buildNode("guard", config, GUARD_TYPE_ID),
            buildNode(
                "support",
                {
                    "system_prompt": SUPPORT_PROMPT,
                    "description": "Answers the customer.",
                    "rag_search": True,
                },
            ),
        ],
        edges=GUARD_EDGES,
    )


class MeetingPinecone:
    """Wraps the fake Pinecone: a search goes on once the guard's model
    has been asked too, at the barrier they share."""

    def __init__(self, pinecone) -> None:
        """`pinecone`, meeting the guard at no barrier yet."""
        self.pinecone = pinecone
        self.barrier: asyncio.Barrier | None = None

    async def search(self, *arguments) -> list:
        """Wait at the barrier, if there is one, then search."""
        if self.barrier is not None:
            await self.barrier.wait()
        return await self.pinecone.search(*arguments)


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
def guardManager(monkeypatch, firestore, pinecone) -> AgentManager:
    """A manager with no agent kept, the guarded agent's records in
    Firestore - the agent searching RAG_ID - and every model a
    `MeetingModel`."""
    firestore.put(AGENTS, buildAgent(rag_enabled=True, rag_id=RAG_ID))
    firestore.put(AGENT_TEMPLATES, buildGuardTemplate())
    firestore.put(NODE_TYPES, buildGuardNodeType())
    firestore.put(utilities.RAGS, buildRag())
    serverConfig = ServerConfig(llmApiKeys={"openai": "platform-test-key"})
    monkeypatch.setattr(
        agentChatModel, "getServerConfig", lambda: serverConfig
    )
    monkeypatch.setattr(
        agentChatModel,
        "init_chat_model",
        lambda model, **settings: MeetingModel(),
    )
    return SettlingAgentManager(NodeManager())


def ask(
    agentManager: AgentManager,
    guardScript: list,
    supportScript: list,
    meeting: MeetingPinecone | None = None,
) -> tuple[str, MeetingModel, MeetingModel]:
    """The agent's answer to QUESTION in the conversation, the guard and
    support answering as their scripts say; and the guard's model and
    support's. Given `meeting`, the guard and the search wait for each
    other."""
    agent = asyncio.run(agentManager.getAgent(PROJECT_ID, AGENT_ID))
    models = {node.nodeId: node.chatModel.llm for node in agent.nodes}
    guard, support = models["guard"], models["support"]
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
    """The guard's pick and its search, at the same time."""

    def testGuardAndSearchEachWaitForTheOtherToStart(
        self, guardManager, pinecone, monkeypatch
    ):
        meeting = MeetingPinecone(pinecone)
        monkeypatch.setattr(
            "documentProcessing.service.getPineconeDb", lambda: meeting
        )

        answer, _, _ = ask(
            guardManager,
            [(PASSED, "support")],
            [AIMessage(ANSWER)],
            meeting,
        )

        assert answer == ANSWER
        assert pinecone.searches == [(RAG_ID, QUESTION)]

    def testSearchGoesOnWhileTheChatHistoryLoads(
        self, guardManager, firestore, pinecone, monkeypatch
    ):
        putChatHistory(
            firestore,
            {
                "messages": [
                    {"role": "human", "text": "Hi"},
                    {"role": "ai", "text": "Hello! How can I help?"},
                ]
            },
        )
        loading = asyncio.Event()
        fetchDocument = firestore.fetchDocument

        async def fetchOnceLoading(collection: str, documentId: str):
            if collection == AGENT_MEMORY:
                await loading.wait()
            return await fetchDocument(collection, documentId)

        monkeypatch.setattr(firestore, "fetchDocument", fetchOnceLoading)
        agent = asyncio.run(guardManager.getAgent(PROJECT_ID, AGENT_ID))
        models = {node.nodeId: node.chatModel.llm for node in agent.nodes}
        guard, support = models["guard"], models["support"]
        guard.script = [(PASSED, "support")]
        support.script = [AIMessage(ANSWER)]

        async def askWhileLoading() -> tuple:
            query = asyncio.create_task(
                guardManager.query(PROJECT_ID, AGENT_ID, QUESTION, CONVO_ID)
            )
            while not pinecone.searches:
                await asyncio.sleep(0)
            pickedMeanwhile = list(guard.received)
            loading.set()
            return pickedMeanwhile, (await query).answer

        pickedMeanwhile, answer = runWithin(askWhileLoading())

        assert answer == ANSWER
        # Searched before the history loaded; the pick waited for it.
        assert pickedMeanwhile == []
        [guardSent] = guard.received
        assert textsOf(guardSent) == [
            GUARD_PROMPT,
            "Hi",
            "Hello! How can I help?",
            QUESTION,
        ]


class TestAnswer:
    """A message passed on, answered from the guard's search; or turned
    away."""

    def testPassedMessageIsAnsweredFromTheGuardsSearch(
        self, guardManager, pinecone
    ):
        answer, guard, support = ask(
            guardManager, [(PASSED, "support")], [AIMessage(ANSWER)]
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
            PASSED,
        ]
        searched, found = supportSent[2:4]
        assert searched.tool_calls[0]["args"] == {"query": QUESTION}
        assert isinstance(found, ToolMessage)
        assert found.tool_call_id == GUARD_SEARCH_ID

    def testSupportMaySearchAgain(self, guardManager, pinecone):
        answer, _, support = ask(
            guardManager,
            [(PASSED, "support")],
            [searchFor("delivery times"), AIMessage(ANSWER)],
        )

        assert answer == ANSWER
        assert pinecone.searches == [
            (RAG_ID, QUESTION),
            (RAG_ID, "delivery times"),
        ]
        assert len(support.received) == 2

    def testTurnedAwayMessageGoesOutWithTheGuardsReply(
        self, guardManager, pinecone
    ):
        answer, _, support = ask(
            guardManager, [(TURNED_AWAY, OUTPUT_NODE_ID)], []
        )

        assert answer == TURNED_AWAY
        assert support.received == []
        # Searched as the guard picked, then dropped.
        assert pinecone.searches == [(RAG_ID, QUESTION)]


class TestBuilt:
    """What a guard needs to be built."""

    def testGuardWithNoRagIsRefused(self, guardManager, firestore):
        firestore.put(AGENT_TEMPLATES, buildGuardTemplate(rag_search=False))

        with pytest.raises(AgentNotRunnableError, match="has no rag"):
            asyncio.run(guardManager.getAgent(PROJECT_ID, AGENT_ID))
