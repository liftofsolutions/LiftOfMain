"""Btest for the agent manager: agents built from their records, kept in
an LRU of its own, and run on queries.

Firestore is a `FakeFirestore`, holding the records of `db/agent-schema`
an agent is built from - an agent; its template, with a reply node and a
WhatsApp node; their node types - and the chat history of the agent's
conversations, keeping every read and write. memoryDb is a fresh one for
each test,
with the agent caches registered, and the chat model's provider is
`RecordingModel`, which answers with ANSWER.
Everything between is real: `agentStore`, `loadAgent`, the node manager,
the chat model node, LangGraph and the service.

1. Loading: an agent is built from its records - its reply node's config
   merged over its node type's, its graph the one its template's edges
   wire, the WhatsApp node left out - and holds its node reserved; an
   agent not in the project is not found, and everything that cannot
   run, its graph included, is refused, with no node left reserved.
2. Graphs: nodes run as the edges wire them, by id, one at a time - in
   a line, each reading the answers before it; a node with several edges
   out goes on to the one its model picks, out of the nodes they lead
   to, each read by its name and description - so a loop runs until its
   node picks the way out - and a repeated edge is one edge. Nodes may
   share a name, and only the nodes the edges name are reserved. A run
   comes in at the input node and goes out at the output node, and one
   going on past the step limit is refused. Triggers: one can start the
   agent if its template has it, enabled, on one of its nodes.
3. Records: they are cached in memoryDb, at most 100 a collection - and
   2000 conversations' chat history - the least recently used dropped
   first, so building again reads nothing.
4. Kept agents: a built agent is kept, ten at most, the least recently
   used dropped first and its node released - once its running query is
   answered; a dropped one is built again on its next query.
5. Queries: the chat model answers, on the node reserved at build;
   queries at once share one load, even one that fails; a cancelled
   query leaves the load to the others; an agent dropped while its query
   waited is loaded again; and an agent not kept is built.
6. Conversations: a query of none keeps nothing; a conversation's first
   query starts its chat history empty, and the history a run ends with
   is saved, to Firestore and memoryDb, so the next query reads it from
   memoryDb - or Firestore, once it is dropped. Nodes - `ScriptedNode`s
   here - read the history and change it, whole or in place; a run that
   fails saves nothing and leaves the cached history as it was, and a
   conversation is one agent's alone. Each node keeps its details for
   the run under its id.
7. Service: anyone gets the agent's answer, in a conversation or none -
   no account, and no project record, is read - agent errors come back
   as error kinds, and every answer fits the route's response model.

    uv run pytest Btest/agents -q
"""

import asyncio
import copy
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from itertools import pairwise

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START

from agents import agentStore, service
from agents.agent import (
    MAX_RUN_STEPS,
    AgentNotFoundError,
    AgentNotRunnableError,
    Reply,
    mergeConfig,
    mergeNodeDetails,
)
from agents.agentManager import AgentManager
from agents.agentStore import (
    AGENT_MEMORY,
    AGENT_TEMPLATES,
    AGENTS,
    NODE_TYPES,
)
from agents.model import QueryAgentResponse
from agents.nodeManager import NodeManager
from agents.nodes import (
    INPUT_NODE_ID,
    NODE_CLASSES,
    OUTPUT_ABOUT,
    OUTPUT_NODE_ID,
    NodeBase,
)
from config import ServerConfig
from DbAdapters import Document
from DbAdapters.memoryDb import EVICTION_LRU, MemoryDb
from documentProcessing import utilities
from LLM import agentChatModel

# The ids of `agents.example.json`, and a few more.
PROJECT_ID = "a7c3e2f1-9b4d-4e6a-8c2f-5d1b0e9a7f34"
OTHER_PROJECT_ID = "5d1b0e9a-7f34-4e6a-8c2f-a7c3e2f19b4d"
ACCOUNT_ID = "8f3c2a1e-5b7d-4c9a-a1f2-3e6b9d0c7a41"
AGENT_ID = "88f95300-7453-4609-b685-2c424cc181d5"
TEMPLATE_ID = "8f32ea63-10e7-4105-a05d-ea0c7c02ff52"
CHAT_TYPE_ID = "9788e43c-dbf2-48d9-bed0-5da33fdc566a"
WHATSAPP_TYPE_ID = "1b06fcd1-73a2-4703-a9bc-00ae5a96a10a"
SEARCH_TYPE_ID = "69f06d0f-56e5-4778-9887-2d24c1d824e6"
SCRIPTED_TYPE_ID = "3c5f1d8a-2b47-4e9c-9a61-7f0e8d2c4b15"
REPLY_NODE_ID = "473b3fd6-5886-40fc-bd5a-a5d414591bf2"
WHATSAPP_NODE_ID = "9e60cc8c-e30f-44f7-b025-2c613f181635"
SEARCH_NODE_ID = "bffe1424-267a-4ef6-9c05-1cc9c9a62f15"
CREDENTIAL_ID = "f833e685-a764-47c5-a765-5a1bda2c8ce4"
TRIGGER_ID = "d9d81456-5f87-4b46-80e1-0c4d39b36ec7"
MISSING_ID = "00000000-0000-4000-8000-000000000000"
CONVO_ID = "6b0e3c1a-2d4f-4b8e-9a7c-5e1f3d2b4a69"
DELETED_AT = datetime(2026, 10, 10, tzinfo=timezone.utc)

QUESTION = "Do you deliver to Pune?"
ANSWER = "Yes, we deliver to Pune in 2-3 business days."
SYSTEM_PROMPT = "You are Acme's sales assistant. Answer briefly."


class FakeFirestore:
    """Stands in for the Firestore adapter: records by collection and id,
    and every read and write made, in order."""

    def __init__(self) -> None:
        """No records, and no reads or writes yet."""
        self.records: defaultdict[str, dict[str, dict]] = defaultdict(dict)
        self.reads: list[tuple[str, str]] = []
        self.writes: list[tuple[str, str]] = []

    def put(self, collection: str, record: dict) -> None:
        """Store `record` under its `_id`, replacing what is there."""
        self.records[collection][record["_id"]] = record

    async def fetchDocument(
        self, collection: str, documentId: str
    ) -> Document | None:
        """A copy of the stored record, as the Firestore adapter hands one
        back - after a wait, as a real read waits on the network."""
        self.reads.append((collection, documentId))
        await asyncio.sleep(0)
        record = self.records[collection].get(documentId)
        if record is None:
            return None
        return Document(documentId, copy.deepcopy(record))

    async def upsertDocument(
        self, collection: str, document: Document, *, merge: bool = False
    ) -> None:
        """Store a copy of `document`'s fields under its id, replacing what
        is there - after a wait, as a real write waits on the network."""
        self.writes.append((collection, document.id))
        await asyncio.sleep(0)
        self.records[collection][document.id] = copy.deepcopy(document.data)


class RecordingModel:
    """Stands in for a provider's chat model.

    Answers every call with `answer` - ANSWER, unless a test sets another
    - and keeps every list of messages it was sent. Asked to answer in a
    shape, as a node picking where a run goes next asks, it keeps the
    shape, and picks the first of `picks` left - a test sets them. Given
    a `gate`, it holds each answer until the gate is set.
    """

    def __init__(self) -> None:
        """A model that has been sent nothing yet, and holds nothing."""
        self.received: list[list] = []
        self.answer = ANSWER
        self.picks: list[str] = []
        self.shapes: list[type] = []
        self.gate: asyncio.Event | None = None

    async def ainvoke(self, messages: list) -> AIMessage:
        """Keep `messages`, and answer - once the gate is set, if there is
        one."""
        self.received.append(list(messages))
        if self.gate is not None:
            await self.gate.wait()
        return AIMessage(self.answer)

    def with_structured_output(self, shape: type) -> "ShapedModel":
        """The model, answering in `shape`."""
        self.shapes.append(shape)
        return ShapedModel(self, shape)


class ShapedModel:
    """A `RecordingModel` answering in a shape: its answer, and its next
    pick."""

    def __init__(self, model: RecordingModel, shape: type) -> None:
        """`model`, answering in `shape`."""
        self.model = model
        self.shape = shape

    async def ainvoke(self, messages: list) -> object:
        """Keep `messages`, and answer in the shape, with the next pick."""
        self.model.received.append(list(messages))
        return self.shape(
            message=self.model.answer, next=self.model.picks.pop(0)
        )


class ScriptedNode(NodeBase):
    """Stands in for a node of the run's own, keeping a copy of every
    state it was run on. It counts, in the chat history itself, each key
    its config's `count` lists - as a node changing the history in place
    would - and returns its config's `update`."""

    handler = "scripted"

    def __init__(self, nodeId: str, name: str) -> None:
        """The node, run on no state yet."""
        super().__init__(nodeId, name)
        self.seen: list[dict] = []

    async def __call__(self, state: dict) -> dict:
        """Keep the state, count in the history, and return the update."""
        self.seen.append(copy.deepcopy(state))
        chatHistory = state["chatHistory"]
        for key in self.config.get("count", []):
            chatHistory[key] = chatHistory.get(key, 0) + 1
        return copy.deepcopy(self.config.get("update", {}))


def buildChatNodeType(**overrides) -> dict:
    """A platform node type for OpenAI chat."""
    nodeType = {
        "_id": CHAT_TYPE_ID,
        "category": "chat_model",
        "handler": "openai_chat",
        "name": "OpenAI chat",
        "config": {"model": "gpt-5.5", "temperature": 0.7, "max_tokens": 1024},
        "is_active": True,
    }
    nodeType.update(overrides)
    return nodeType


def buildWhatsAppNodeType(**overrides) -> dict:
    """A platform node type for WhatsApp, an io node."""
    nodeType = {
        "_id": WHATSAPP_TYPE_ID,
        "category": "io",
        "handler": "whatsapp",
        "name": "WhatsApp",
        "config": {"api_version": "v21.0"},
        "is_active": True,
    }
    nodeType.update(overrides)
    return nodeType


def buildSearchNodeType(**overrides) -> dict:
    """A platform node type for web search, a tool node."""
    nodeType = {
        "_id": SEARCH_TYPE_ID,
        "category": "tool",
        "handler": "tavily_search",
        "name": "Tavily search",
        "config": {"max_results": 5},
        "is_active": True,
    }
    nodeType.update(overrides)
    return nodeType


def buildReplyNode(**overrides) -> dict:
    """The template's reply node: OpenAI chat, its own temperature and
    system prompt over the node type's."""
    node = {
        "id": REPLY_NODE_ID,
        "node_type_id": CHAT_TYPE_ID,
        "name": "reply",
        "config": {"temperature": 0.3, "system_prompt": SYSTEM_PROMPT},
    }
    node.update(overrides)
    return node


def buildWhatsAppNode(**overrides) -> dict:
    """The template's WhatsApp node, with its credential."""
    node = {
        "id": WHATSAPP_NODE_ID,
        "node_type_id": WHATSAPP_TYPE_ID,
        "credential_id": CREDENTIAL_ID,
        "name": "whatsapp",
        "config": {},
    }
    node.update(overrides)
    return node


def buildSearchNode(**overrides) -> dict:
    """A web search node, a tool node."""
    node = {
        "id": SEARCH_NODE_ID,
        "node_type_id": SEARCH_TYPE_ID,
        "name": "web_search",
        "config": {},
    }
    node.update(overrides)
    return node


def buildChatNode(
    nodeId: str, name: str | None = None, description: str | None = None
) -> dict:
    """A chat model node of id `nodeId`, with a system prompt of its own;
    named `name`, which other nodes may share, else like its id; and
    described by `description`, if there is one."""
    config = {"system_prompt": f"You are {nodeId}."}
    if description:
        config["description"] = description
    return buildReplyNode(id=nodeId, name=name or nodeId, config=config)


def buildScriptedNodeType() -> dict:
    """A node type of `ScriptedNode`."""
    return {
        "_id": SCRIPTED_TYPE_ID,
        "category": "tool",
        "handler": ScriptedNode.handler,
        "name": "Scripted",
        "config": {},
        "is_active": True,
    }


def buildScriptedNode(
    nodeId: str, update: dict | None = None, count: list[str] | None = None
) -> dict:
    """A scripted node of id `nodeId`, counting `count` in the chat
    history and returning `update` as it runs."""
    return {
        "id": nodeId,
        "node_type_id": SCRIPTED_TYPE_ID,
        "name": nodeId,
        "config": {"update": update or {}, "count": count or []},
    }


def chatHistoryIdOf(convoId: str, agentId: str = AGENT_ID) -> str:
    """The id of the chat history record of the conversation `convoId`."""
    return f"{agentId}_{convoId}"


def storedChatHistory(
    firestore: "FakeFirestore", convoId: str, agentId: str = AGENT_ID
) -> dict | None:
    """The chat history record of the conversation `convoId`, in
    Firestore."""
    return firestore.records[AGENT_MEMORY].get(
        chatHistoryIdOf(convoId, agentId)
    )


def askInConvo(
    agentManager: AgentManager, message: str, convoId: str = CONVO_ID
) -> Reply:
    """The agent's reply to `message`, in the conversation `convoId`."""
    return asyncio.run(
        agentManager.query(PROJECT_ID, AGENT_ID, message, convoId)
    )


def putScriptedTemplate(firestore: "FakeFirestore", *nodes: dict) -> None:
    """The template, running `nodes` - scripted nodes - then its reply
    node, one after another."""
    nodeIds = [node["id"] for node in nodes]
    firestore.put(
        AGENT_TEMPLATES,
        buildTemplate(
            nodes=[*nodes, buildReplyNode()],
            edges=lineOf(*nodeIds, REPLY_NODE_ID),
        ),
    )


def lineOf(*nodeIds: str) -> list[dict]:
    """Edges that run the nodes of `nodeIds` one after another, from the
    input node to the output node."""
    return [
        {"from": source, "to": target}
        for source, target in pairwise(
            [INPUT_NODE_ID, *nodeIds, OUTPUT_NODE_ID]
        )
    ]


def buildTrigger(**overrides) -> dict:
    """The template's trigger: a message on its WhatsApp node starts the
    agent."""
    trigger = {
        "id": TRIGGER_ID,
        "node_id": WHATSAPP_NODE_ID,
        "event": "message.received",
        "enabled": True,
    }
    trigger.update(overrides)
    return trigger


def buildTemplate(**overrides) -> dict:
    """The project's template: a reply node, which its graph runs, and a
    WhatsApp node, which it leaves out."""
    template = {
        "_id": TEMPLATE_ID,
        "project_id": PROJECT_ID,
        "name": "Acme sales assistant",
        "rag_enabled": False,
        "nodes": [buildReplyNode(), buildWhatsAppNode()],
        "edges": lineOf(REPLY_NODE_ID),
        "triggers": [],
        "is_active": True,
    }
    template.update(overrides)
    return template


def templateWith(**overrides) -> list[tuple[str, dict]]:
    """The change of records that gives the agent's template `overrides`."""
    return [(AGENT_TEMPLATES, buildTemplate(**overrides))]


def buildAgent(**overrides) -> dict:
    """The project's active agent, running the template."""
    agent = {
        "_id": AGENT_ID,
        "project_id": PROJECT_ID,
        "template_id": TEMPLATE_ID,
        "name": "Acme Sales Bot",
        "status": "active",
        "created_by": ACCOUNT_ID,
    }
    agent.update(overrides)
    return agent


def buildWorld() -> list[tuple[str, dict]]:
    """Every record of a healthy agent, with its collection; and a tool
    node type no node uses."""
    return [
        (AGENTS, buildAgent()),
        (AGENT_TEMPLATES, buildTemplate()),
        (NODE_TYPES, buildChatNodeType()),
        (NODE_TYPES, buildWhatsAppNodeType()),
        (NODE_TYPES, buildSearchNodeType()),
    ]


def putAgents(firestore: "FakeFirestore", count: int) -> list[str]:
    """`count` more agents of the template, each under a new id."""
    agentIds = [str(uuid.uuid4()) for _ in range(count)]
    for agentId in agentIds:
        firestore.put(AGENTS, buildAgent(_id=agentId))
    return agentIds


async def queryAtOnce(
    agentManager: AgentManager, projectIds: list[str]
) -> list:
    """The agent queried at once under each of `projectIds`: each answer,
    or the error raised instead."""
    replies = await asyncio.gather(
        *(
            agentManager.query(projectId, AGENT_ID, QUESTION)
            for projectId in projectIds
        ),
        return_exceptions=True,
    )
    return [answerOf(reply) for reply in replies]


def answerOf(reply: Reply | BaseException) -> str | BaseException:
    """The answer a reply gives, or the error raised instead of one."""
    return reply if isinstance(reply, BaseException) else reply.answer


# What a change of records does to an agent that cannot then run, and a
# part of the refusal's message.
NOT_RUNNABLE_CASES = {
    "paused": ([(AGENTS, buildAgent(status="paused"))], "is paused"),
    "templateGone": (
        [(AGENTS, buildAgent(template_id=MISSING_ID))],
        "is gone",
    ),
    "anotherProjectsTemplate": (
        templateWith(project_id=OTHER_PROJECT_ID),
        "is not of project",
    ),
    "platformTemplate": (templateWith(project_id=None), "is not of project"),
    "inactiveTemplate": (templateWith(is_active=False), "is inactive"),
    "twoNodesOneId": (
        templateWith(nodes=[buildReplyNode(), buildReplyNode(name="summary")]),
        f"two nodes of id {REPLY_NODE_ID!r}",
    ),
    "nodeIdInput": (
        templateWith(nodes=[buildReplyNode(id=INPUT_NODE_ID)]),
        "has a node of id 'input'",
    ),
    "nodeIdLangGraphKeeps": (
        templateWith(nodes=[buildReplyNode(id=END)]),
        "has a node of id '__end__'",
    ),
    "noEdges": (templateWith(edges=[]), "has no edges"),
    "edgeWithoutTo": (
        templateWith(edges=[{"from": INPUT_NODE_ID}]),
        "needs a from and a to",
    ),
    "edgeNotAnObject": (
        templateWith(edges=[INPUT_NODE_ID]),
        "needs a from and a to",
    ),
    "edgeToANumber": (
        templateWith(edges=[{"from": INPUT_NODE_ID, "to": 7}]),
        "needs a from and a to",
    ),
    "edgeFromSeveralNodes": (
        templateWith(
            edges=[
                *lineOf(REPLY_NODE_ID),
                {"from": [INPUT_NODE_ID, REPLY_NODE_ID], "to": OUTPUT_NODE_ID},
            ]
        ),
        "needs a from and a to",
    ),
    "edgeToLabelledNodes": (
        templateWith(
            edges=[
                *lineOf(REPLY_NODE_ID),
                {"from": REPLY_NODE_ID, "to": {"again": REPLY_NODE_ID}},
            ]
        ),
        "needs a from and a to",
    ),
    "edgeNamesNoNode": (
        templateWith(edges=lineOf("summary")),
        "'summary', which is no node's id",
    ),
    "edgeNamesANodeByName": (
        templateWith(edges=lineOf("reply")),
        "'reply', which is no node's id",
    ),
    "langGraphsStartAndEnd": (
        templateWith(
            edges=[
                {"from": START, "to": REPLY_NODE_ID},
                {"from": REPLY_NODE_ID, "to": END},
            ]
        ),
        "'__start__', which is no node's id",
    ),
    "noWayIn": (
        templateWith(edges=[{"from": REPLY_NODE_ID, "to": OUTPUT_NODE_ID}]),
        "no edge from 'input'",
    ),
    "twoWaysIn": (
        templateWith(
            nodes=[buildReplyNode(), buildChatNode("summary")],
            edges=[*lineOf(REPLY_NODE_ID), *lineOf("summary")],
        ),
        "edges from 'input' to 2 nodes",
    ),
    "noWayOut": (
        templateWith(
            edges=[
                {"from": INPUT_NODE_ID, "to": REPLY_NODE_ID},
                {"from": REPLY_NODE_ID, "to": REPLY_NODE_ID},
            ]
        ),
        "no edge to 'output'",
    ),
    "nodeWithNoEdgeOut": (
        templateWith(
            nodes=[buildReplyNode(), buildChatNode("summary")],
            edges=[
                *lineOf(REPLY_NODE_ID),
                {"from": REPLY_NODE_ID, "to": "summary"},
            ],
        ),
        "'summary' of template",
    ),
    "edgeFromOutput": (
        templateWith(
            edges=[
                *lineOf(REPLY_NODE_ID),
                {"from": OUTPUT_NODE_ID, "to": REPLY_NODE_ID},
            ]
        ),
        "leaves from 'output'",
    ),
    "edgeToInput": (
        templateWith(
            edges=[
                *lineOf(REPLY_NODE_ID),
                {"from": REPLY_NODE_ID, "to": INPUT_NODE_ID},
            ]
        ),
        "leads to 'input'",
    ),
    "nodeTypeGone": (
        templateWith(nodes=[buildReplyNode(node_type_id=MISSING_ID)]),
        "has no node type",
    ),
    "anotherProjectsNodeType": (
        [(NODE_TYPES, buildChatNodeType(project_id=OTHER_PROJECT_ID))],
        "another project's node type",
    ),
    "inactiveNodeType": (
        [(NODE_TYPES, buildChatNodeType(is_active=False))],
        "inactive node type",
    ),
    "ioNodeInTheGraph": (
        templateWith(edges=lineOf(REPLY_NODE_ID, WHATSAPP_NODE_ID)),
        "No node has the handler 'whatsapp'",
    ),
    "toolNodeInTheGraph": (
        templateWith(
            nodes=[buildReplyNode(), buildSearchNode()],
            edges=lineOf(REPLY_NODE_ID, SEARCH_NODE_ID),
        ),
        "No node has the handler 'tavily_search'",
    ),
    "handlerNoNodeClassRuns": (
        [(NODE_TYPES, buildChatNodeType(handler="anthropic_chat"))],
        "No node has the handler 'anthropic_chat'",
    ),
    "nameNotASlug": (
        templateWith(nodes=[buildReplyNode(name="Reply Bot")]),
        "not a slug",
    ),
    "modelNotListed": (
        [(NODE_TYPES, buildChatNodeType(config={"model": "gpt-unlisted"}))],
        "cannot be reserved: 'gpt-unlisted'",
    ),
}

# classify picks where the run goes on to: sales, or handoff.
PICK_EDGES = [
    {"from": INPUT_NODE_ID, "to": "classify"},
    {"from": "classify", "to": "sales"},
    {"from": "classify", "to": "handoff"},
    {"from": "sales", "to": OUTPUT_NODE_ID},
    {"from": "handoff", "to": OUTPUT_NODE_ID},
]

# draft writes, and review picks: draft, to write again, or the way out.
LOOP_EDGES = [
    {"from": INPUT_NODE_ID, "to": "draft"},
    {"from": "draft", "to": "review"},
    {"from": "review", "to": "draft"},
    {"from": "review", "to": OUTPUT_NODE_ID},
]


def runGraph(
    agentManager: AgentManager,
    firestore: FakeFirestore,
    edges: list[dict],
    answers: dict[str, str],
    names: dict[str, str] | None = None,
    picks: dict[str, list[str]] | None = None,
) -> tuple[str, dict[str, RecordingModel]]:
    """Ask the agent QUESTION, its template a chat model node for each id
    of `answers` - named as `names` says, else like its id - wired by
    `edges`, each node answering as `answers` says, and picking where the
    run goes next, in turn, as `picks` says.

    The agent's answer, and each node's model, by node id.
    """
    names = names or {}
    picks = picks or {}
    nodes = [buildChatNode(nodeId, names.get(nodeId)) for nodeId in answers]
    firestore.put(AGENT_TEMPLATES, buildTemplate(nodes=nodes, edges=edges))
    agent = asyncio.run(agentManager.getAgent(PROJECT_ID, AGENT_ID))
    models = {node.nodeId: node.chatModel.llm for node in agent.nodes}
    for nodeId, model in models.items():
        model.answer = answers[nodeId]
        model.picks = list(picks.get(nodeId, []))
    reply = asyncio.run(agentManager.query(PROJECT_ID, AGENT_ID, QUESTION))
    return reply.answer, models


def contentsOf(messages: list) -> list[str]:
    """The content of each of `messages`."""
    return [message.content for message in messages]


@pytest.fixture
def memoryDb(monkeypatch) -> MemoryDb:
    """A fresh memoryDb for every record read, with the agent caches."""
    freshMemoryDb = MemoryDb()
    monkeypatch.setattr(utilities, "getMemoryDb", lambda: freshMemoryDb)
    monkeypatch.setattr(agentStore, "getMemoryDb", lambda: freshMemoryDb)
    agentStore.registerCaches()
    return freshMemoryDb


@pytest.fixture
def firestore(monkeypatch, memoryDb) -> FakeFirestore:
    """Firestore, holding a healthy project and agent."""
    fakeFirestore = FakeFirestore()
    for collection, record in buildWorld():
        fakeFirestore.put(collection, record)
    monkeypatch.setattr(
        utilities, "getFirestoreAdapter", lambda: fakeFirestore
    )
    return fakeFirestore


@pytest.fixture
def builtModels(monkeypatch) -> list[RecordingModel]:
    """Every chat model the nodes build, in order, in place of a real
    one; the platform's keys are test ones."""
    models = []

    def buildRecordingModel(model: str, **settings) -> RecordingModel:
        recordingModel = RecordingModel()
        models.append(recordingModel)
        return recordingModel

    serverConfig = ServerConfig(llmApiKeys={"openai": "platform-test-key"})
    monkeypatch.setattr(
        agentChatModel, "getServerConfig", lambda: serverConfig
    )
    monkeypatch.setattr(
        agentChatModel, "init_chat_model", buildRecordingModel
    )
    return models


@pytest.fixture
def agentManager(firestore, builtModels) -> AgentManager:
    """A manager with no agent kept, on a node manager of its own."""
    return AgentManager(NodeManager())


@pytest.fixture
def scriptedManager(firestore, builtModels) -> AgentManager:
    """A manager whose node manager reserves scripted nodes too, their
    node type in Firestore."""
    firestore.put(NODE_TYPES, buildScriptedNodeType())
    nodeClasses = {**NODE_CLASSES, ScriptedNode.handler: ScriptedNode}
    return AgentManager(NodeManager(nodeClasses=nodeClasses))


@pytest.fixture
def serviceManager(monkeypatch, agentManager) -> AgentManager:
    """`agentManager`, as the one the service hands its calls to."""
    monkeypatch.setattr(service, "getAgentManager", lambda: agentManager)
    return agentManager


class TestLoading:
    """An agent built from its records, and what cannot be."""

    def testAgentIsBuiltFromItsRecords(self, agentManager, firestore):
        agent = asyncio.run(agentManager.getAgent(PROJECT_ID, AGENT_ID))

        assert (
            agent.agentId,
            agent.name,
            agent.projectId,
            agent.templateId,
        ) == (AGENT_ID, "Acme Sales Bot", PROJECT_ID, TEMPLATE_ID)
        [node] = agent.nodes
        assert (node.handler, node.nodeId, node.name) == (
            "openai_chat",
            REPLY_NODE_ID,
            "reply",
        )
        assert node.config == {
            "model": "gpt-5.5",
            "temperature": 0.3,
            "max_tokens": 1024,
            "system_prompt": SYSTEM_PROMPT,
        }
        edges = agent.graph.get_graph().edges
        assert {(edge.source, edge.target) for edge in edges} == {
            (START, INPUT_NODE_ID),
            (INPUT_NODE_ID, REPLY_NODE_ID),
            (REPLY_NODE_ID, OUTPUT_NODE_ID),
            (OUTPUT_NODE_ID, END),
        }
        assert firestore.reads == [
            (AGENTS, AGENT_ID),
            (AGENT_TEMPLATES, TEMPLATE_ID),
            (NODE_TYPES, CHAT_TYPE_ID),
        ]

    def testNodeConfigIsMergedDeeplyOverItsNodeTypes(self):
        base = {
            "model": "gpt-5.5",
            "retry": {"attempts": 3, "delay": 1},
            "tags": ["a", "b"],
        }
        override = {"retry": {"attempts": 5}, "tags": ["c"]}

        merged = mergeConfig(base, override)

        assert merged == {
            "model": "gpt-5.5",
            "retry": {"attempts": 5, "delay": 1},
            "tags": ["c"],
        }
        assert base["retry"] == {"attempts": 3, "delay": 1}

    def testBuiltAgentHoldsItsNodeReserved(self, agentManager, builtModels):
        agent = asyncio.run(agentManager.getAgent(PROJECT_ID, AGENT_ID))

        assert agentManager.nodeManager.reservedNodes == set(agent.nodes)
        [model] = builtModels
        assert model.received == []

    @pytest.mark.parametrize(
        "agentId, agentChanges",
        [
            (MISSING_ID, {}),
            (AGENT_ID, {"project_id": OTHER_PROJECT_ID}),
            (AGENT_ID, {"deleted_at": DELETED_AT}),
        ],
        ids=["missing", "anotherProjects", "deleted"],
    )
    def testAgentNotOfTheProjectIsNotFound(
        self, agentManager, firestore, agentId, agentChanges
    ):
        firestore.put(AGENTS, buildAgent(**agentChanges))

        with pytest.raises(AgentNotFoundError) as raised:
            asyncio.run(agentManager.getAgent(PROJECT_ID, agentId))

        assert f"has no agent {agentId!r}" in str(raised.value)
        assert agentManager.agents == {}

    @pytest.mark.parametrize(
        "changes, expected",
        list(NOT_RUNNABLE_CASES.values()),
        ids=list(NOT_RUNNABLE_CASES),
    )
    def testAgentThatCannotRunIsRefused(
        self, agentManager, firestore, changes, expected
    ):
        for collection, record in changes:
            firestore.put(collection, record)

        with pytest.raises(AgentNotRunnableError) as raised:
            asyncio.run(agentManager.getAgent(PROJECT_ID, AGENT_ID))

        assert expected in str(raised.value)
        assert agentManager.agents == {}
        assert agentManager.nodeManager.reservedNodes == set()


class TestGraphs:
    """Agents of several nodes, run as their template's edges wire them."""

    def testEachNodeReadsTheAnswersBeforeIt(self, agentManager, firestore):
        answer, models = runGraph(
            agentManager,
            firestore,
            lineOf("draft", "review", "reply"),
            {"draft": "A draft.", "review": "A review.", "reply": ANSWER},
        )

        assert answer == ANSWER
        [replySent] = models["reply"].received
        assert contentsOf(replySent) == [
            "You are reply.",
            QUESTION,
            "A draft.",
            "A review.",
        ]
        assert [model.shapes for model in models.values()] == [[], [], []]

    @pytest.mark.parametrize("picked", ["sales", "handoff"])
    def testNodeWithSeveralEdgesOutGoesOnToTheOneItPicks(
        self, agentManager, firestore, picked
    ):
        answers = {
            "classify": "A sales question.",
            "sales": "Sales here.",
            "handoff": "A person will help.",
        }

        answer, models = runGraph(
            agentManager,
            firestore,
            PICK_EDGES,
            answers,
            picks={"classify": [picked]},
        )

        assert answer == answers[picked]
        ran = [nodeId for nodeId, model in models.items() if model.received]
        assert ran == ["classify", picked]
        [pickedSent] = models[picked].received
        assert contentsOf(pickedSent)[1:] == [QUESTION, "A sales question."]

    def testNodePicksOutOfTheNodesItsEdgesLeadTo(
        self, agentManager, firestore
    ):
        nodes = [
            buildChatNode("classify"),
            buildChatNode("sales", "seller", "Answers sales questions."),
        ]
        edges = [
            {"from": INPUT_NODE_ID, "to": "classify"},
            {"from": "classify", "to": "sales"},
            {"from": "classify", "to": OUTPUT_NODE_ID},
            {"from": "sales", "to": OUTPUT_NODE_ID},
        ]
        firestore.put(AGENT_TEMPLATES, buildTemplate(nodes=nodes, edges=edges))
        agent = asyncio.run(agentManager.getAgent(PROJECT_ID, AGENT_ID))
        classify = agent.nodes[0].chatModel.llm
        classify.picks = [OUTPUT_NODE_ID]

        reply = asyncio.run(agentManager.query(PROJECT_ID, AGENT_ID, QUESTION))

        [shape] = classify.shapes
        nextStep = shape.model_json_schema()["properties"]["next"]
        assert nextStep["enum"] == ["sales", OUTPUT_NODE_ID]
        assert nextStep["description"].splitlines()[1:] == [
            "sales: seller - Answers sales questions.",
            f"output: {OUTPUT_ABOUT}",
        ]
        assert reply.answer == ANSWER
        picks = {
            (edge.source, edge.target)
            for edge in agent.graph.get_graph().edges
            if edge.conditional
        }
        assert picks == {("classify", "sales"), ("classify", OUTPUT_NODE_ID)}

    def testLoopRunsUntilItsNodePicksTheWayOut(self, agentManager, firestore):
        answer, models = runGraph(
            agentManager,
            firestore,
            LOOP_EDGES,
            {"draft": "A draft.", "review": "A review."},
            picks={"review": ["draft", "draft", OUTPUT_NODE_ID]},
        )

        # The draft's, as the review only picks.
        assert answer == "A draft."
        assert len(models["review"].received) == 3
        assert len(models["draft"].received) == 3
        lastDraftSent = models["draft"].received[-1]
        assert contentsOf(lastDraftSent)[1:] == [
            QUESTION,
            "A draft.",
            "A review.",
            "A draft.",
            "A review.",
        ]

    def testRepeatedEdgeIsOneEdge(self, agentManager, firestore):
        edges = [*lineOf("reply"), {"from": "reply", "to": OUTPUT_NODE_ID}]

        answer, models = runGraph(
            agentManager, firestore, edges, {"reply": ANSWER}
        )

        assert answer == ANSWER
        assert models["reply"].shapes == []

    def testNodesMayShareAName(self, agentManager, firestore):
        answer, models = runGraph(
            agentManager,
            firestore,
            lineOf("first", "second"),
            {"first": "A first review.", "second": ANSWER},
            names={"first": "reviewer", "second": "reviewer"},
        )

        assert answer == ANSWER
        [secondSent] = models["second"].received
        assert contentsOf(secondSent) == [
            "You are second.",
            QUESTION,
            "A first review.",
        ]

    def testOnlyTheNodesTheEdgesNameAreReserved(self, agentManager, firestore):
        nodes = [buildChatNode("reply"), buildChatNode("unused")]
        firestore.put(
            AGENT_TEMPLATES, buildTemplate(nodes=nodes, edges=lineOf("reply"))
        )

        agent = asyncio.run(agentManager.getAgent(PROJECT_ID, AGENT_ID))

        assert [node.nodeId for node in agent.nodes] == ["reply"]
        assert agentManager.nodeManager.reservedNodes == set(agent.nodes)

    def testRunComesInAtInputAndGoesOutAtOutput(self, agentManager):
        agent = asyncio.run(agentManager.getAgent(PROJECT_ID, AGENT_ID))

        async def stepsOfARun() -> list[dict]:
            return [
                step
                async for step in agent.graph.astream(
                    {"inputMessage": QUESTION}, stream_mode="updates"
                )
            ]

        steps = asyncio.run(stepsOfARun())

        assert [list(step) for step in steps] == [
            [INPUT_NODE_ID],
            [REPLY_NODE_ID],
            [OUTPUT_NODE_ID],
        ]
        assert contentsOf(steps[0][INPUT_NODE_ID]["messages"]) == [QUESTION]
        assert steps[2][OUTPUT_NODE_ID] == {"answer": ANSWER}

    def testRunGoingOnPastTheStepLimitIsRefused(
        self, agentManager, firestore
    ):
        # review never picks the way out; it has picks for every step.
        picks = {"review": ["draft"] * MAX_RUN_STEPS}

        with pytest.raises(AgentNotRunnableError) as raised:
            runGraph(
                agentManager,
                firestore,
                LOOP_EDGES,
                {"draft": "A draft.", "review": "Again."},
                picks=picks,
            )

        assert f"went past {MAX_RUN_STEPS} steps" in str(raised.value)
        agent = agentManager.agents[AGENT_ID]
        [_, review] = [node.chatModel.llm for node in agent.nodes]
        # The input node is a step, then draft and review a step each.
        assert len(review.received) == (MAX_RUN_STEPS - 1) // 2
        assert agent.runningQueries == 0


class TestTriggers:
    """Which of its template's triggers can start an agent."""

    @pytest.mark.parametrize(
        "triggerChanges, triggerId, available",
        [
            ({}, TRIGGER_ID, True),
            ({"enabled": False}, TRIGGER_ID, False),
            ({"node_id": MISSING_ID}, TRIGGER_ID, False),
            ({}, MISSING_ID, False),
        ],
        ids=["enabled", "disabled", "onNoNodeOfTheTemplate", "notItsTrigger"],
    )
    def testTriggerIsAvailableEnabledOnANodeOfTheTemplate(
        self, agentManager, firestore, triggerChanges, triggerId, available
    ):
        trigger = buildTrigger(**triggerChanges)
        firestore.put(AGENT_TEMPLATES, buildTemplate(triggers=[trigger]))

        agent = asyncio.run(agentManager.getAgent(PROJECT_ID, AGENT_ID))

        assert agent.isTriggerAvailable(triggerId) is available


class TestRecords:
    """The records an agent is built from, cached in memoryDb."""

    @pytest.mark.parametrize(
        "collection", [AGENTS, AGENT_TEMPLATES, NODE_TYPES]
    )
    def testEachCollectionCachesAHundredLeastUsedFirst(
        self, memoryDb, collection
    ):
        stats = memoryDb.getCollectionStats(collection)

        assert stats.maxRecords == 100
        assert stats.evictionPolicy == EVICTION_LRU

    def testChatHistoryCachesTwoThousandLeastUsedFirst(self, memoryDb):
        stats = memoryDb.getCollectionStats(AGENT_MEMORY)

        assert stats.maxRecords == 2000
        assert stats.evictionPolicy == EVICTION_LRU

    def testBuildingAgainReadsNothingFromFirestore(
        self, firestore, builtModels
    ):
        firstManager = AgentManager(NodeManager())
        asyncio.run(firstManager.getAgent(PROJECT_ID, AGENT_ID))
        readsBefore = len(firestore.reads)

        secondManager = AgentManager(NodeManager())
        asyncio.run(secondManager.getAgent(PROJECT_ID, AGENT_ID))

        assert len(firestore.reads) == readsBefore


class TestKeptAgents:
    """The agents kept: ten at most, the least recently used dropped."""

    def testBuiltAgentIsKept(self, agentManager, firestore):
        first = asyncio.run(agentManager.getAgent(PROJECT_ID, AGENT_ID))
        readsBefore = len(firestore.reads)

        second = asyncio.run(agentManager.getAgent(PROJECT_ID, AGENT_ID))

        assert second is first
        assert len(firestore.reads) == readsBefore
        assert list(agentManager.agents) == [AGENT_ID]

    def testTenAreKeptTheLeastRecentlyUsedDroppedFirst(
        self, agentManager, firestore
    ):
        agentIds = putAgents(firestore, 11)

        for agentId in agentIds:
            asyncio.run(agentManager.getAgent(PROJECT_ID, agentId))

        assert list(agentManager.agents) == agentIds[1:]

    def testDroppedAgentReleasesItsNode(self, agentManager, firestore):
        agentIds = putAgents(firestore, 11)

        dropped, *kept = [
            asyncio.run(agentManager.getAgent(PROJECT_ID, agentId))
            for agentId in agentIds
        ]

        assert dropped.nodes == []
        assert dropped.graph is None
        assert agentManager.nodeManager.reservedNodes == {
            node for agent in kept for node in agent.nodes
        }

    def testDroppedAgentHoldsItsNodeUntilItsQueryIsAnswered(
        self, agentManager, firestore, builtModels
    ):
        agentIds = putAgents(firestore, 10)

        async def dropWhileQuerying() -> tuple:
            agent = await agentManager.getAgent(PROJECT_ID, AGENT_ID)
            [node] = agent.nodes
            builtModels[0].gate = asyncio.Event()
            query = asyncio.create_task(
                agentManager.query(PROJECT_ID, AGENT_ID, QUESTION)
            )
            for agentId in agentIds:
                await agentManager.getAgent(PROJECT_ID, agentId)
            heldWhileQuerying = node in agentManager.nodeManager.reservedNodes
            builtModels[0].gate.set()
            return node, heldWhileQuerying, (await query).answer

        node, heldWhileQuerying, answer = asyncio.run(dropWhileQuerying())

        assert AGENT_ID not in agentManager.agents
        assert heldWhileQuerying
        assert answer == ANSWER
        assert node not in agentManager.nodeManager.reservedNodes

    def testQueryMakesAnAgentTheMostRecentlyUsed(
        self, agentManager, firestore
    ):
        agentIds = putAgents(firestore, 11)
        for agentId in agentIds[:10]:
            asyncio.run(agentManager.getAgent(PROJECT_ID, agentId))

        asyncio.run(agentManager.query(PROJECT_ID, agentIds[0], QUESTION))
        asyncio.run(agentManager.getAgent(PROJECT_ID, agentIds[10]))

        assert agentIds[0] in agentManager.agents
        assert agentIds[1] not in agentManager.agents

    def testDroppedAgentIsBuiltAgainOnItsNextQuery(
        self, agentManager, firestore
    ):
        agentIds = putAgents(firestore, 11)
        for agentId in agentIds:
            asyncio.run(agentManager.getAgent(PROJECT_ID, agentId))
        readsBefore = len(firestore.reads)

        reply = asyncio.run(
            agentManager.query(PROJECT_ID, agentIds[0], QUESTION)
        )

        assert reply.answer == ANSWER
        assert list(agentManager.agents) == agentIds[2:] + agentIds[:1]
        assert len(firestore.reads) == readsBefore

    def testKeptAgentIsNotFoundForAnotherProject(self, agentManager):
        asyncio.run(agentManager.getAgent(PROJECT_ID, AGENT_ID))

        with pytest.raises(AgentNotFoundError):
            asyncio.run(agentManager.getAgent(OTHER_PROJECT_ID, AGENT_ID))


class TestQueries:
    """What a query does, alone and with others."""

    def testQueryIsAnsweredByTheChatModel(self, agentManager, builtModels):
        reply = asyncio.run(agentManager.query(PROJECT_ID, AGENT_ID, QUESTION))

        assert reply.answer == ANSWER
        [model] = builtModels
        [sent] = model.received
        assert [(type(message), message.content) for message in sent] == [
            (SystemMessage, SYSTEM_PROMPT),
            (HumanMessage, QUESTION),
        ]

    def testQueriesRunOnTheNodeReservedAtBuild(
        self, agentManager, builtModels
    ):
        for _ in range(2):
            asyncio.run(agentManager.query(PROJECT_ID, AGENT_ID, QUESTION))

        [model] = builtModels
        assert len(model.received) == 2
        [agent] = agentManager.agents.values()
        assert agentManager.nodeManager.reservedNodes == set(agent.nodes)

    def testQueriesAtOnceShareOneLoad(self, agentManager, firestore):
        answers = asyncio.run(queryAtOnce(agentManager, [PROJECT_ID] * 3))

        assert answers == [ANSWER] * 3
        assert firestore.reads.count((AGENTS, AGENT_ID)) == 1
        [agent] = agentManager.agents.values()
        assert agentManager.nodeManager.reservedNodes == set(agent.nodes)
        assert agentManager.loading == {}

    def testQueriesAtOnceShareTheLoadThatFails(
        self, agentManager, firestore
    ):
        firestore.put(AGENTS, buildAgent(status="paused"))

        errors = asyncio.run(queryAtOnce(agentManager, [PROJECT_ID] * 3))

        assert [type(error) for error in errors] == [AgentNotRunnableError] * 3
        assert firestore.reads.count((AGENTS, AGENT_ID)) == 1
        assert agentManager.loading == {}

    def testCancelledQueryLeavesTheLoadToTheOthers(self, agentManager):
        async def cancelOneOfTwo() -> list:
            queries = [
                asyncio.create_task(
                    agentManager.query(PROJECT_ID, AGENT_ID, QUESTION)
                )
                for _ in range(2)
            ]
            await asyncio.sleep(0)
            queries[0].cancel()
            return await asyncio.gather(*queries, return_exceptions=True)

        cancelled, answered = asyncio.run(cancelOneOfTwo())

        assert isinstance(cancelled, asyncio.CancelledError)
        assert answered.answer == ANSWER

    def testQueryUnderAnotherProjectSharesNoLoad(self, agentManager):
        wrongProject, rightProject = asyncio.run(
            queryAtOnce(agentManager, [OTHER_PROJECT_ID, PROJECT_ID])
        )

        assert isinstance(wrongProject, AgentNotFoundError)
        assert rightProject == ANSWER

    def testAgentDroppedWhileItsQueryWaitedIsLoadedAgain(
        self, firestore, builtModels
    ):
        agentManager = AgentManager(NodeManager(), maxAgents=1)
        [otherAgentId] = putAgents(firestore, 1)

        async def queryBothAtOnce() -> list:
            return await asyncio.gather(
                agentManager.query(PROJECT_ID, AGENT_ID, QUESTION),
                agentManager.query(PROJECT_ID, otherAgentId, QUESTION),
                return_exceptions=True,
            )

        replies = asyncio.run(queryBothAtOnce())

        assert [answerOf(reply) for reply in replies] == [ANSWER, ANSWER]
        [kept] = agentManager.agents.values()
        assert agentManager.nodeManager.reservedNodes == set(kept.nodes)

    def testQueryBuildsAnAgentNotKept(self, agentManager):
        asyncio.run(agentManager.query(PROJECT_ID, AGENT_ID, QUESTION))

        assert list(agentManager.agents) == [AGENT_ID]


class TestConvos:
    """Queries in conversations, each one's chat history kept for its
    nodes - and of none; and each node's details for a run."""

    def testQueryOfNoConvoKeepsNothing(self, scriptedManager, firestore):
        putScriptedTemplate(
            firestore,
            buildScriptedNode("notes", {"chatHistory": {"reports": 1}}),
        )

        reply = asyncio.run(
            scriptedManager.query(PROJECT_ID, AGENT_ID, QUESTION)
        )

        assert reply == Reply(ANSWER, None)
        [notes, _] = scriptedManager.agents[AGENT_ID].nodes
        assert notes.seen[0]["chatHistory"] == {}
        assert firestore.writes == []

    def testConvosFirstQueryStartsItsChatHistoryEmpty(
        self, scriptedManager, firestore
    ):
        putScriptedTemplate(firestore, buildScriptedNode("notes"))

        reply = askInConvo(scriptedManager, QUESTION)

        assert reply == Reply(ANSWER, CONVO_ID)
        [notes, _] = scriptedManager.agents[AGENT_ID].nodes
        assert notes.seen[0]["chatHistory"] == {}
        record = storedChatHistory(firestore, CONVO_ID)
        assert {
            key: record[key]
            for key in ("agent_id", "project_id", "convo_id", "chat_history")
        } == {
            "agent_id": AGENT_ID,
            "project_id": PROJECT_ID,
            "convo_id": CONVO_ID,
            "chat_history": {},
        }

    def testChatHistoryTheRunEndsWithIsSavedAndCached(
        self, scriptedManager, firestore, memoryDb
    ):
        chatHistory = {"summary": "Asked about delivery to Pune."}
        putScriptedTemplate(
            firestore,
            buildScriptedNode("notes", {"chatHistory": chatHistory}),
        )

        askInConvo(scriptedManager, QUESTION)

        record = storedChatHistory(firestore, CONVO_ID)
        assert record["chat_history"] == chatHistory
        cached = memoryDb.getRecord(AGENT_MEMORY, chatHistoryIdOf(CONVO_ID))
        assert cached["chat_history"] == chatHistory

    def testConvosNextQueryReadsItsChatHistoryFromTheCache(
        self, scriptedManager, firestore
    ):
        putScriptedTemplate(
            firestore, buildScriptedNode("notes", count=["questions"])
        )
        askInConvo(scriptedManager, QUESTION)

        askInConvo(scriptedManager, "And?")

        [notes, _] = scriptedManager.agents[AGENT_ID].nodes
        assert notes.seen[1]["chatHistory"] == {"questions": 1}
        record = storedChatHistory(firestore, CONVO_ID)
        assert record["chat_history"] == {"questions": 2}
        # Read once, before the first answer, when there was none.
        chatHistoryId = chatHistoryIdOf(CONVO_ID)
        assert firestore.reads.count((AGENT_MEMORY, chatHistoryId)) == 1

    def testChatHistoryDroppedFromTheCacheIsReadFromFirestore(
        self, scriptedManager, firestore, memoryDb
    ):
        putScriptedTemplate(
            firestore, buildScriptedNode("notes", count=["questions"])
        )
        askInConvo(scriptedManager, QUESTION)
        chatHistoryId = chatHistoryIdOf(CONVO_ID)
        memoryDb.deleteRecord(AGENT_MEMORY, chatHistoryId)

        askInConvo(scriptedManager, "And?")

        assert firestore.reads.count((AGENT_MEMORY, chatHistoryId)) == 2
        record = storedChatHistory(firestore, CONVO_ID)
        assert record["chat_history"] == {"questions": 2}
        assert memoryDb.getRecord(AGENT_MEMORY, chatHistoryId) is not None

    def testRunThatFailsSavesNothingAndLeavesTheChatHistory(
        self, scriptedManager, firestore, memoryDb
    ):
        # notes counts in the history in place, then review loops past the
        # step limit on the second query.
        nodes = [
            buildScriptedNode("notes", count=["questions"]),
            buildChatNode("draft"),
            buildChatNode("review"),
        ]
        edges = [
            {"from": INPUT_NODE_ID, "to": "notes"},
            {"from": "notes", "to": "draft"},
            *LOOP_EDGES[1:],
        ]
        firestore.put(AGENT_TEMPLATES, buildTemplate(nodes=nodes, edges=edges))
        agent = asyncio.run(scriptedManager.getAgent(PROJECT_ID, AGENT_ID))
        review = agent.nodes[2].chatModel.llm
        review.picks = [OUTPUT_NODE_ID]
        askInConvo(scriptedManager, QUESTION)
        review.picks = ["draft"] * MAX_RUN_STEPS

        with pytest.raises(AgentNotRunnableError):
            askInConvo(scriptedManager, "Again?")

        chatHistoryId = chatHistoryIdOf(CONVO_ID)
        assert firestore.writes == [(AGENT_MEMORY, chatHistoryId)]
        cached = memoryDb.getRecord(AGENT_MEMORY, chatHistoryId)
        assert cached["chat_history"] == {"questions": 1}

    def testSaveFirestoreRefusesLeavesTheCachedChatHistory(
        self, scriptedManager, firestore, memoryDb, monkeypatch
    ):
        putScriptedTemplate(
            firestore, buildScriptedNode("notes", count=["questions"])
        )
        askInConvo(scriptedManager, QUESTION)

        async def refuseTheWrite(*arguments, **keywords) -> None:
            raise ConnectionError("Firestore is out of reach.")

        monkeypatch.setattr(firestore, "upsertDocument", refuseTheWrite)

        with pytest.raises(ConnectionError):
            askInConvo(scriptedManager, "And?")

        cached = memoryDb.getRecord(AGENT_MEMORY, chatHistoryIdOf(CONVO_ID))
        assert cached["chat_history"] == {"questions": 1}

    def testConvoIsOfOneAgentAlone(self, scriptedManager, firestore):
        putScriptedTemplate(
            firestore, buildScriptedNode("notes", count=["questions"])
        )
        askInConvo(scriptedManager, QUESTION)
        [otherAgentId] = putAgents(firestore, 1)

        asyncio.run(
            scriptedManager.query(PROJECT_ID, otherAgentId, QUESTION, CONVO_ID)
        )

        [notes, _] = scriptedManager.agents[otherAgentId].nodes
        assert notes.seen[0]["chatHistory"] == {}
        for agentId in (AGENT_ID, otherAgentId):
            record = storedChatHistory(firestore, CONVO_ID, agentId)
            assert record["chat_history"] == {"questions": 1}

    def testEachNodeKeepsItsDetailsForTheRunUnderItsId(
        self, scriptedManager, firestore
    ):
        nodes = [
            buildScriptedNode(
                "first", {"nodeDetails": {"first": {"count": 1}}}
            ),
            buildScriptedNode(
                "second", {"nodeDetails": {"first": {"seenBy": "second"}}}
            ),
            buildScriptedNode("third", {}),
            buildChatNode("reply"),
        ]
        firestore.put(
            AGENT_TEMPLATES,
            buildTemplate(
                nodes=nodes, edges=lineOf("first", "second", "third", "reply")
            ),
        )

        for _ in range(2):
            asyncio.run(scriptedManager.query(PROJECT_ID, AGENT_ID, QUESTION))

        first, second, third, _ = scriptedManager.agents[AGENT_ID].nodes
        assert first.seen[1]["nodeDetails"] == {}
        assert second.seen[0]["nodeDetails"] == {"first": {"count": 1}}
        assert third.seen[0]["nodeDetails"] == {
            "first": {"count": 1, "seenBy": "second"}
        }

    def testNodeDetailsMergeIntoEachNodesOwn(self):
        nodeDetails = {"first": {"count": 1}}

        merged = mergeNodeDetails(
            nodeDetails,
            {"first": {"seenBy": "second"}, "second": {"count": 2}},
        )

        assert merged == {
            "first": {"count": 1, "seenBy": "second"},
            "second": {"count": 2},
        }
        assert nodeDetails == {"first": {"count": 1}}


class TestService:
    """The service: free to use, and answers fit for the route."""

    def testAnyoneGetsTheAgentsAnswer(self, serviceManager, firestore):
        payload = asyncio.run(
            service.queryAgent(PROJECT_ID, AGENT_ID, QUESTION)
        )

        assert QueryAgentResponse(**payload) == QueryAgentResponse(
            agentId=AGENT_ID, convoId=None, answer=ANSWER
        )
        readCollections = {collection for collection, _ in firestore.reads}
        assert readCollections == {AGENTS, AGENT_TEMPLATES, NODE_TYPES}

    def testConvoGoesOnThroughTheService(self, serviceManager, firestore):
        payloads = [
            asyncio.run(
                service.queryAgent(PROJECT_ID, AGENT_ID, message, CONVO_ID)
            )
            for message in (QUESTION, "And?")
        ]

        assert [QueryAgentResponse(**payload) for payload in payloads] == [
            QueryAgentResponse(
                agentId=AGENT_ID, convoId=CONVO_ID, answer=ANSWER
            )
        ] * 2
        chatHistoryId = chatHistoryIdOf(CONVO_ID)
        assert firestore.writes == [(AGENT_MEMORY, chatHistoryId)] * 2

    @pytest.mark.parametrize(
        "agentId, agentChanges, errorKind",
        [
            (MISSING_ID, {}, "notFound"),
            (AGENT_ID, {"project_id": OTHER_PROJECT_ID}, "notFound"),
            (AGENT_ID, {"status": "paused"}, "conflict"),
        ],
        ids=["missing", "anotherProjects", "paused"],
    )
    def testAgentErrorsComeBackAsErrorKinds(
        self, serviceManager, firestore, agentId, agentChanges, errorKind
    ):
        firestore.put(AGENTS, buildAgent(**agentChanges))

        result = asyncio.run(
            service.queryAgent(PROJECT_ID, agentId, QUESTION)
        )

        assert result["errorKind"] == errorKind
        assert agentId in result["error"]
        assert serviceManager.agents == {}
