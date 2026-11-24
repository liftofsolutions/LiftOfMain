"""An agent: who it is, the nodes it reserved, and the LangGraph graph its
template wires them into, which answers its queries.

    agent = await loadAgent(nodeManager, projectId, agentId)
    agent.isTriggerAvailable(triggerId)
    reply = await agent.query("Do you deliver to Pune?", convoId)
    del agent    # destroyed with its nodes, once nothing holds it

An `Agent` holds who it is - agentId, name, projectId and templateId,
from its record in `agents`, its description and metadata, and its
knowledge base, the record's `rag_id` when `rag_enabled` - and the
node manager it reserves its nodes from. `build` readies it to run,
once:

1. It fetches its template from `agent_templates`, and parses the graph
   in it - below - checking all of it before anything is reserved.
2. It fetches, from `node_types`, the type of each node the graph runs:
   the handler that runs the node, and the default config the node's own
   is merged over.
3. It reserves those nodes from the node manager, each configured as it
   will run. A chat model takes no credential: it runs on the platform's
   key, which the node manager's chat model nodes hold. A node whose
   config has `rag_search` is handed the agent's rag, in `rag`, to
   search: so two agents of one template may each search their own.
   Every node is handed the agent's details, `detailsOf`, in `agent`
   - never a template's own - and a chat model fills the placeholders
   of its base prompt with them: `{{persona}}` becomes the agent's.
   A copy, not the agent, so no node holds it alive.
4. It builds the LangGraph graph of them, wired as the template says,
   and keeps it compiled, in `graph`.

It keeps its template's triggers too: `isTriggerAvailable` says whether
one can start the agent - the template has it, enabled, on one of its
nodes.

`query` runs the graph on a message - in a conversation, if it names
one - and answers with what the run goes out with. Each run is handed
`LLM.USAGE_LOG`, which logs the tokens of every chat model call made in
it.

Its life
--------

1. Built - `loadAgent` reads its records, reserves its nodes, each built
   new, and builds its graph of them.
2. Kept - the agent manager keeps it, up to its limit of agents, and it
   answers queries, many at once, sharing its graph and nodes. A run
   goes on past its reply, to save its chat history.
3. Destroyed - once nothing holds it: the manager has dropped it, and
   every query it was running is answered, its run over, as a query
   and its run hold the agent while they go on. Python destroys it
   then, and its destructor, `__del__`, hands each node back to the
   node manager, which wipes its credential and config; its graph, its
   triggers and the rest of it go with it.

What nodes keep
---------------

Nodes keep two things in the run's state, each as they will:

* `chatHistory` - what is remembered of the conversation, a plain dict.
  A query names its conversation by a `convoId`, and a copy of its chat
  history - empty, the first time - loads into the run while it starts,
  in parallel: a node reads it once `awaitChatHistory` has waited for
  it, then reads and changes it, returning `{"chatHistory": ...}` whole,
  or changing it in place. Once the run is answered, the history it ends
  with is saved - to memoryDb's cache first, so the next query reads it
  at once, then to Firestore - and a run that fails saves nothing. A
  query with no `convoId` - a cron agent's report, say - is of no
  conversation: its history starts empty, and is not saved.
* `nodeDetails` - each node's own data for the run, under its id,
  `state["nodeDetails"][nodeId]`, which a node changes by returning
  `{"nodeDetails": {nodeId: {key: value}}}`, merged into what is there.
  They start empty each run, and are not saved. A node picking where
  the run goes next counts its picks there, `{"picks": {nodeId: n}}`.

A `chat_history` node keeps the conversation in `chatHistory`, in the
shape every chat model node reads - its summary, and its messages - and
summarises it once it grows long: see `agents.nodes.chatHistoryNode`.

The template's graph
--------------------

LangGraph builds a graph in code, not from JSON, so a template carries
its own: `nodes`, as `db/agent-schema` has them, and `edges`, which wire
them by id, each from one node to one node - which node runs after
which, so where each node's output goes:

    {"from": "<draft>", "to": "<review>"}
        review runs after draft - <draft> standing here for the id of a
        node named draft, and so on.

Edges name nodes by id as no two nodes of a template share one; names
are labels, and two reviewers may both be named "reviewer". A node's
output is what it adds to the conversation, a chat model its answer,
and every node reads the whole conversation so far.

Two nodes are in every graph, the special nodes of
`agents.nodes.specialNodes`, built by the agent rather than reserved;
edges name them by their ids, "input" and "output". A run comes in at
the input node, which starts the conversation with the message the run
was given - by a trigger, or a query - and goes out at the output node,
with the run's answer: the last reply of a node that answers, not of
one that picks - a reviewer passing a reply is not the one heard - or,
in a run where none answered, its last message.

A run goes on one node at a time. A node with one edge out goes on to
the node it leads to; a node with several picks one of the nodes they
lead to, run as a `ChoosingNode`. Its model answers as structured
output, in a fixed shape: its
answer, and the id of the node it picks, which can be none but theirs.
It reads each of them by its name and what it does - its config's
`description` - and the output node as the end of the run, its own
answer going out as the agent's. A node's `pick_limits` cap how often
it picks each node in a run, as its config says, counting its picks in
its `nodeDetails`. So an if-else is a node with an edge each way, and a
loop an edge back, capped or not:

    "edges": [
        {"from": "input", "to": "<draft>"},
        {"from": "<draft>", "to": "<review>"},
        {"from": "<review>", "to": "<draft>"},
        {"from": "<review>", "to": "output"}
    ]

        review picks: draft, to write it again, or output - output
        alone, once it has picked draft as often as
        `"pick_limits": {"<draft>": 3}` lets it.

A graph has one edge from "input", where no model picks, and one edge
or more to "output"; nothing leads to "input" and nothing leaves
"output", and every other node has an edge out. So a run goes out at
"output", unless it goes on past MAX_RUN_STEPS steps, a node run each,
and is refused. Only nodes the edges name are part of the graph: io
nodes - WhatsApp, email, a schedule - receive and send messages on their
own channels, and are left out.

What cannot run is refused as it is built, with no node left reserved:

* `AgentNotFoundError` - no such agent in the project: missing, deleted,
  or another project's, all answered alike.
* `AgentNotRunnableError` - the agent is paused; its template is missing,
  inactive or another project's; two of its nodes share an id, or one
  has an id the graph keeps - "input", "output", or LangGraph's
  "__start__" and "__end__"; its edges are missing or malformed, name an
  id no node has, lead from "input" to no node or to several, never
  reach "output", or leave a node with no edge out; or a node the graph
  runs has a type that is missing, inactive or another project's, or
  cannot be reserved - a handler no node class runs, as no io or tool
  node does yet, a name that is not a slug, a model `configLLM.json`
  does not list; a node searches a rag, and the agent has none; or a
  node has several edges out but cannot pick - it is no chat model node,
  or it searches a rag and is no guard, which searches as it picks - or
  its pick limits are not counts, name a node it has no edge to, or
  limit every node it goes on to.
"""

import asyncio
import copy
import logging
from typing import Annotated, NamedTuple

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents import agentStore
from agents.nodeManager import NodeManager, NodeManagerError
from agents.nodes import (
    INPUT_NODE_ID,
    OUTPUT_ABOUT,
    OUTPUT_NODE_ID,
    RUN_CHAT_HISTORY,
    RUN_REPLY,
    SPECIAL_NODE_IDS,
    AfterReplyNode,
    ChatModelNode,
    ChoosingNode,
    InputNode,
    NodeBase,
    OutputNode,
)
from agents.nodes.chatModelNode import PICK_LIMITS
from agents.nodes.promptTemplate import AGENT_DETAILS
from LLM import USAGE_LOG

LOGGER = logging.getLogger(__name__)

AGENT_STATUS_ACTIVE = "active"

# Ids no node of a template may have: the special nodes', and LangGraph's
# own.
KEPT_NODE_IDS = (*SPECIAL_NODE_IDS, START, END)

# The most steps a run takes, a node run each, before it is refused: a
# bound on loops, as LangGraph's own - 10007, unless set - is hardly one.
MAX_RUN_STEPS = 25


class AgentNotFoundError(LookupError):
    """No such agent in the project: missing, deleted, or another's."""


class AgentNotRunnableError(RuntimeError):
    """The agent is there, but cannot be run as it is."""


class Reply(NamedTuple):
    """An agent's answer, and the id of the conversation it was given in,
    if any."""

    answer: str
    convoId: str | None


def mergeNodeDetails(
    nodeDetails: dict[str, dict], changes: dict[str, dict]
) -> dict[str, dict]:
    """`nodeDetails` with the details nodes returned, each node's merged
    into its own, by its id."""
    merged = dict(nodeDetails)
    for nodeId, details in changes.items():
        merged[nodeId] = {**merged.get(nodeId, {}), **details}
    return merged


class AgentState(MessagesState):
    """A run's state: the message it comes in with, the conversation, and
    the answer it goes out with; and, for its nodes, the conversation's
    chat history and each node's details."""

    inputMessage: str
    # The last reply of a node that answers; once the run goes out, its
    # answer.
    answer: str
    # The conversation's chat history, as the nodes keep it: loaded before
    # the run, and saved after it.
    chatHistory: dict
    # Each node's own details for the run, by its id; not saved.
    nodeDetails: Annotated[dict[str, dict], mergeNodeDetails]


def mergeConfig(base: dict, override: dict) -> dict:
    """`override` merged over `base`, as `db/agent-schema` merges a node's
    config over its node type's.

    Dicts are merged deeply, the override winning; anything else in the
    override, a list too, replaces what was there.
    """
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            merged[key] = mergeConfig(current, value)
        else:
            merged[key] = value
    return merged


def parseGraph(
    template: dict, templateId: str
) -> tuple[list[dict], dict[str, list[str]]]:
    """The nodes the template's graph runs, in the template's order, and
    the ids of the nodes each of them - and the input node - goes on to,
    in the edges' order, a repeated edge once: all checked first, so
    nothing is reserved for a graph that cannot be built.

    Raises:
        AgentNotRunnableError: If two nodes share an id, or one has an id
            the graph keeps; or the edges are missing or malformed, name
            an id no node of the template has, lead from "input" to no
            node or to several, never reach "output", or leave a node
            with no edge out.
    """
    nodesById: dict[str, dict] = {}
    for node in template.get("nodes", []):
        nodeId = node["id"]
        if nodeId in KEPT_NODE_IDS:
            raise AgentNotRunnableError(
                f"Template {templateId!r} has a node of id {nodeId!r}, "
                "which the graph keeps for itself."
            )
        if nodeId in nodesById:
            raise AgentNotRunnableError(
                f"Template {templateId!r} has two nodes of id {nodeId!r}."
            )
        nodesById[nodeId] = node

    edges = template.get("edges")
    if not edges:
        raise AgentNotRunnableError(
            f"Template {templateId!r} has no edges to build a graph of."
        )
    nextNodeIds: dict[str, list[str]] = {}
    for index, edge in enumerate(edges):
        where = f"Edge {index} of template {templateId!r}"
        if not (
            isinstance(edge, dict)
            and isinstance(edge.get("from"), str)
            and isinstance(edge.get("to"), str)
        ):
            raise AgentNotRunnableError(
                f"{where} needs a from and a to, each one node's id."
            )
        source, target = edge["from"], edge["to"]
        if source == OUTPUT_NODE_ID:
            raise AgentNotRunnableError(
                f"{where} leaves from {OUTPUT_NODE_ID!r}, where a run goes "
                "out."
            )
        if target == INPUT_NODE_ID:
            raise AgentNotRunnableError(
                f"{where} leads to {INPUT_NODE_ID!r}, where a run comes in."
            )
        for nodeId in (source, target):
            if nodeId not in nodesById and nodeId not in SPECIAL_NODE_IDS:
                raise AgentNotRunnableError(
                    f"{where} names {nodeId!r}, which is no node's id."
                )
        targets = nextNodeIds.setdefault(source, [])
        if target not in targets:
            targets.append(target)

    firstNodeIds = nextNodeIds.get(INPUT_NODE_ID, [])
    if not firstNodeIds:
        raise AgentNotRunnableError(
            f"Template {templateId!r} has no edge from {INPUT_NODE_ID!r}, "
            "so its graph has no way in."
        )
    if len(firstNodeIds) > 1:
        raise AgentNotRunnableError(
            f"Template {templateId!r} has edges from {INPUT_NODE_ID!r} to "
            f"{len(firstNodeIds)} nodes; a run comes in at one, as no "
            "model there picks."
        )
    reached = {target for targets in nextNodeIds.values() for target in targets}
    if OUTPUT_NODE_ID not in reached:
        raise AgentNotRunnableError(
            f"Template {templateId!r} has no edge to {OUTPUT_NODE_ID!r}, "
            "so its graph has no way out."
        )
    graphNodes = [
        node
        for nodeId, node in nodesById.items()
        if nodeId in nextNodeIds or nodeId in reached
    ]
    for node in graphNodes:
        if node["id"] not in nextNodeIds:
            raise AgentNotRunnableError(
                f"Node {node['id']!r} of template {templateId!r} has no "
                f"edge out; every node leads on, at last to "
                f"{OUTPUT_NODE_ID!r}."
            )
    return graphNodes, nextNodeIds


def availableTriggersOf(template: dict) -> set[str]:
    """The ids of the template's triggers that can start an agent: those
    enabled, on one of its nodes."""
    nodeIds = {node["id"] for node in template.get("nodes", [])}
    return {
        trigger["id"]
        for trigger in template.get("triggers", [])
        if trigger.get("enabled") and trigger.get("node_id") in nodeIds
    }


def aboutOf(node: NodeBase) -> str:
    """What a node picking where a run goes next reads of `node`: its
    name, and what it does."""
    about = node.about()["function"]
    return f"{about['name']} - {about['description']}"


def checkPicks(node: NodeBase, targets: list[str]) -> None:
    """Refuse a node of `targets` - the ids of the nodes its edges lead to
    - that cannot pick one as it says.

    Raises:
        AgentNotRunnableError: If its pick limits are not counts of 1 or
            more, by node id, name a node it has no edge to, or limit
            every node it goes on to, so a run could find no way on; or
            it has several edges out but is no chat model node, which
            picks, or it searches a rag, which a node that picks does
            not - but a guard, whose class `searchesWhilePicking`.
    """
    name = node.name
    limits = node.config.get(PICK_LIMITS, {})
    if not isinstance(limits, dict) or not all(
        isinstance(limit, int) and not isinstance(limit, bool) and limit > 0
        for limit in limits.values()
    ):
        raise AgentNotRunnableError(
            f"Node {name!r} has pick limits that are not counts of 1 or "
            "more, by node id."
        )
    for nodeId in limits:
        if nodeId not in targets:
            raise AgentNotRunnableError(
                f"Node {name!r} limits its picks of {nodeId!r}, to which it "
                "has no edge."
            )
    if limits and set(targets) <= set(limits):
        raise AgentNotRunnableError(
            f"Node {name!r} limits its picks of every node it goes on to; "
            "one needs none, so a run always has a way on."
        )
    if len(targets) > 1 and not isinstance(node, ChatModelNode):
        raise AgentNotRunnableError(
            f"Node {name!r} has several edges out, but no model to pick "
            "one with."
        )
    if (
        len(targets) > 1
        and node.config.get("rag")
        and not node.searchesWhilePicking
    ):
        raise AgentNotRunnableError(
            f"Node {name!r} has several edges out, so it picks where a run "
            "goes, and a node that picks does not search a rag."
        )


def buildGraph(
    nodes: list[NodeBase], nextNodeIds: dict[str, list[str]]
) -> CompiledStateGraph:
    """The compiled LangGraph graph of `nodes`, each under its id, between
    the input node and the output node: each goes on to the node its one
    edge leads to, or picks one of several - by `nextNodeIds`, as
    `parseGraph` checked them - and no run goes past MAX_RUN_STEPS
    steps.

    Raises:
        AgentNotRunnableError: If a node cannot pick as `checkPicks` says.
    """
    for node in nodes:
        checkPicks(node, nextNodeIds[node.nodeId])
    aboutById = {node.nodeId: aboutOf(node) for node in nodes}
    aboutById[OUTPUT_NODE_ID] = OUTPUT_ABOUT
    graph = StateGraph(AgentState)
    graph.add_edge(START, INPUT_NODE_ID)
    for node in [InputNode(), *nodes]:
        targets = nextNodeIds[node.nodeId]
        if len(targets) == 1:
            ranNode = AfterReplyNode(node) if node.runsAfterReply else node
            graph.add_node(node.nodeId, ranNode)
            graph.add_edge(node.nodeId, targets[0])
        else:
            choices = {target: aboutById[target] for target in targets}
            graph.add_node(
                node.nodeId,
                ChoosingNode(node, choices),
                destinations=tuple(targets),
            )
    graph.add_node(OUTPUT_NODE_ID, OutputNode())
    graph.add_edge(OUTPUT_NODE_ID, END)
    return graph.compile().with_config(recursion_limit=MAX_RUN_STEPS)


class Agent:
    """A project's agent: built once into a graph of the nodes it
    reserved, run on queries, and destroyed with its nodes."""

    def __init__(
        self,
        agentId: str,
        name: str,
        projectId: str,
        templateId: str,
        nodeManager: NodeManager,
        ragId: str | None = None,
        description: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """An agent not built yet: who it is, the node manager it reserves
        its nodes from, the rag its nodes search, if it has one, and
        what it is for and like: its description and metadata."""
        self.agentId = agentId
        self.name = name
        self.projectId = projectId
        self.templateId = templateId
        self.nodeManager = nodeManager
        self.ragId = ragId
        self.description = description
        self.metadata = dict(metadata or {})
        # Its reserved nodes, and the graph built of them.
        self.nodes: list[NodeBase] = []
        self.graph: CompiledStateGraph | None = None
        # The ids of the triggers that can start it.
        self.triggerIds: set[str] = set()
        # The runs going on, kept until they end, as asyncio keeps only a
        # weak hold on a task.
        self.runs: set[asyncio.Task] = set()

    def __repr__(self) -> str:
        """Who the agent is, and its nodes by name - nothing of their
        credentials."""
        return (
            f"Agent(agentId={self.agentId!r}, name={self.name!r}, "
            f"projectId={self.projectId!r}, "
            f"templateId={self.templateId!r}, "
            f"nodes={[node.name for node in self.nodes]!r})"
        )

    async def build(self) -> None:
        """Fetch the template and parse its graph, fetch the types of the
        nodes it runs, reserve those nodes, and build the graph of them.

        Raises:
            AgentNotRunnableError: If the template, its graph or a node of
                it cannot run; no node is left reserved.
        """
        template = await agentStore.fetchTemplate(self.templateId)
        self._checkTemplate(template)
        graphNodes, nextNodeIds = parseGraph(template, self.templateId)
        nodeTypes = await asyncio.gather(
            *(
                agentStore.fetchNodeType(node["node_type_id"])
                for node in graphNodes
            )
        )
        typedNodes = list(zip(graphNodes, nodeTypes, strict=True))
        for node, nodeType in typedNodes:
            self._checkNodeType(node, nodeType)
        configs = [
            self._configOf(node, nodeType)
            for node, nodeType in typedNodes
        ]
        try:
            for (node, nodeType), config in zip(
                typedNodes, configs, strict=True
            ):
                self.nodes.append(self._reserve(node, nodeType, config))
            self.graph = buildGraph(self.nodes, nextNodeIds)
        except Exception:
            self._returnNodes()
            raise
        self.triggerIds = availableTriggersOf(template)
        LOGGER.info("Built %s.", self)

    def detailsOf(self) -> dict[str, str]:
        """What its nodes' prompts may name of the agent, by key: each
        text of its metadata - `persona`, `businessName` and the rest -
        then its own `name` and `description`."""
        details = {
            key: value
            for key, value in self.metadata.items()
            if isinstance(value, str)
        }
        details["name"] = self.name
        if self.description:
            details["description"] = self.description
        return details

    def isTriggerAvailable(self, triggerId: str) -> bool:
        """Whether the trigger `triggerId` can start the agent: its
        template has it, enabled, on one of its nodes."""
        return triggerId in self.triggerIds

    async def query(self, message: str, convoId: str | None = None) -> Reply:
        """The agent's answer to `message`, in the conversation `convoId`
        if one is named, and that id: given as soon as the run has it.

        The run starts at once, in a task of its own, and the
        conversation's chat history - a copy, empty the first time, or
        with no conversation - loads into it meanwhile: a node waits for
        it only as it reads it, so a guard's search of the rag goes on
        while it loads. The run replies at its first after-reply node, or
        its output node: the rest of it - keeping the history, summarising
        it - and saving the history it ends with, if a conversation is
        named, happen after the answer is handed back. A failure then is
        logged, and nothing is saved.

        Raises:
            AgentNotRunnableError: If the agent is not built;
                or the run goes on past MAX_RUN_STEPS steps without
                reaching the output node, as a loop whose node never picks
                the way out does.
        """
        if self.graph is None:
            raise AgentNotRunnableError(
                f"Agent {self.agentId!r} is not built."
            )
        reply = asyncio.get_running_loop().create_future()
        run = asyncio.create_task(self._runAndSave(message, convoId, reply))
        self.runs.add(run)
        run.add_done_callback(self.runs.discard)
        # Shielded: a query cancelled while it waits leaves the run to go
        # on, and save.
        answer = await asyncio.shield(reply)
        return Reply(answer, convoId)

    async def _runAndSave(
        self,
        message: str,
        convoId: str | None,
        reply: asyncio.Future,
    ) -> None:
        """Run the graph on `message`, its chat history loading meanwhile,
        giving `reply` its answer as the run replies, then save the chat
        history it ends with, in the conversation `convoId` if one is
        named.

        A failure before the reply is the reply's; one after it is logged,
        and saves nothing. A save Firestore refuses is logged too: the
        cache has the history already.
        """
        RUN_REPLY.set(reply)
        chatHistory: dict = {}
        loading = None
        if convoId is not None:
            loading = asyncio.create_task(
                self._loadChatHistory(convoId, chatHistory)
            )
        RUN_CHAT_HISTORY.set(loading)
        try:
            state = await self._run(message, chatHistory)
            if loading is not None:
                # Loaded before it is saved, even if no node read it.
                await loading
        except Exception as error:
            if not reply.done():
                reply.set_exception(error)
            else:
                LOGGER.exception(
                    "Agent %s replied in conversation %s, then failed; its "
                    "chat history is not saved.",
                    self.agentId,
                    convoId,
                )
            return
        finally:
            if loading is not None and not loading.done():
                loading.cancel()
            if not reply.done():
                reply.cancel()
        if convoId is None:
            return
        try:
            await agentStore.saveChatHistory(
                self.agentId, self.projectId, convoId, state["chatHistory"]
            )
        except Exception:
            LOGGER.exception(
                "Agent %s's chat history of conversation %s is cached, but "
                "Firestore did not save it; its next save writes it whole.",
                self.agentId,
                convoId,
            )

    async def settle(self) -> None:
        """Wait until every run going on has ended: replied, its chat
        history saved or its failure logged."""
        await asyncio.gather(*self.runs)

    def __del__(self) -> None:
        """Destroy the agent with what it holds, as nothing holds it any
        more: each node goes back to the node manager, which wipes its
        credential and config, and the graph and triggers go too."""
        self._returnNodes()
        self.triggerIds.clear()
        LOGGER.debug("Destroyed agent %s.", self.agentId)

    async def _run(self, message: str, chatHistory: dict) -> dict:
        """The state a run of the graph on `message` ends with, coming in
        with `chatHistory`.

        Raises:
            AgentNotRunnableError: If the run goes on past MAX_RUN_STEPS
                steps without reaching the output node.
        """
        try:
            return await self.graph.ainvoke(
                {
                    "inputMessage": message,
                    "chatHistory": chatHistory,
                    "nodeDetails": {},
                },
                config={"callbacks": [USAGE_LOG]},
            )
        except GraphRecursionError as error:
            raise AgentNotRunnableError(
                f"A run of agent {self.agentId!r} went past {MAX_RUN_STEPS} "
                f"steps without reaching {OUTPUT_NODE_ID!r}."
            ) from error

    async def _loadChatHistory(self, convoId: str, chatHistory: dict) -> None:
        """Load a copy of the conversation's chat history into
        `chatHistory`, in place - the run's, for its nodes to change as
        they will, while the cached one stays as it is until the run is
        saved; nothing, for a new conversation."""
        stored = await agentStore.fetchChatHistory(self.agentId, convoId)
        if stored is not None:
            chatHistory.update(copy.deepcopy(stored))

    def _returnNodes(self) -> None:
        """Hand every node back to the node manager, which wipes its
        credential and config; the graph of them goes with them."""
        for node in self.nodes:
            self.nodeManager.releaseNode(node)
        self.nodes.clear()
        self.graph = None

    def _checkTemplate(self, template: dict | None) -> None:
        """Refuse a template that is missing, another project's, or
        inactive.

        A premade platform template has no project, so it is refused too:
        an agent runs a copy of it in its own project.
        """
        if template is None:
            raise AgentNotRunnableError(
                f"Template {self.templateId!r} is gone."
            )
        if template.get("project_id") != self.projectId:
            raise AgentNotRunnableError(
                f"Template {self.templateId!r} is not of project "
                f"{self.projectId!r}."
            )
        if not template.get("is_active"):
            raise AgentNotRunnableError(
                f"Template {self.templateId!r} is inactive."
            )

    def _checkNodeType(self, node: dict, nodeType: dict | None) -> None:
        """Refuse a node whose type is missing, another project's, or
        inactive."""
        name = node["name"]
        if nodeType is None:
            raise AgentNotRunnableError(
                f"Node {name!r} has no node type {node['node_type_id']!r}."
            )
        if nodeType.get("project_id") not in (None, self.projectId):
            raise AgentNotRunnableError(
                f"Node {name!r} has another project's node type."
            )
        if not nodeType.get("is_active"):
            raise AgentNotRunnableError(
                f"Node {name!r} has an inactive node type."
            )

    def _configOf(self, node: dict, nodeType: dict) -> dict:
        """The config the node runs with: its own, merged over its type's;
        the agent's details, in AGENT_DETAILS; and, if it has
        `rag_search`, the agent's rag, in `rag`: as no config but the
        agent's names either.

        Raises:
            AgentNotRunnableError: If the node searches a rag, and the
                agent has none.
        """
        config = mergeConfig(nodeType.get("config", {}), node.get("config", {}))
        config.pop("rag", None)
        config[AGENT_DETAILS] = self.detailsOf()
        if config.get("rag_search"):
            if self.ragId is None:
                raise AgentNotRunnableError(
                    f"Node {node['name']!r} searches a rag, and agent "
                    f"{self.agentId!r} has none."
                )
            config["rag"] = {
                "project_id": self.projectId,
                "rag_id": self.ragId,
            }
        return config

    def _reserve(self, node: dict, nodeType: dict, config: dict) -> NodeBase:
        """The node, reserved from the node manager with `config`; a chat
        model takes no credential."""
        try:
            return self.nodeManager.reserveNode(
                nodeType["handler"],
                nodeId=node["id"],
                name=node["name"],
                config=config,
            )
        except (NodeManagerError, KeyError, ValueError) as error:
            raise AgentNotRunnableError(
                f"Node {node['name']!r} of agent {self.agentId!r} cannot be "
                f"reserved: {error}"
            ) from error


async def loadAgent(
    nodeManager: NodeManager, projectId: str, agentId: str
) -> Agent:
    """The agent `agentId` of `projectId`, read from its record and built
    with nodes from `nodeManager`.

    Raises:
        AgentNotFoundError: If the project has no such agent.
        AgentNotRunnableError: If the agent, or its template, cannot be
            run as it is.
    """
    record = await agentStore.fetchAgent(agentId)
    if (
        record is None
        or record.get("project_id") != projectId
        or record.get("deleted_at") is not None
    ):
        raise AgentNotFoundError(
            f"Project {projectId!r} has no agent {agentId!r}."
        )
    status = record.get("status")
    if status != AGENT_STATUS_ACTIVE:
        raise AgentNotRunnableError(
            f"Agent {agentId!r} is {status}; only an active agent runs."
        )
    metadata = record.get("metadata")
    agent = Agent(
        agentId,
        record["name"],
        projectId,
        record["template_id"],
        nodeManager,
        record.get("rag_id") if record.get("rag_enabled") else None,
        record.get("description"),
        metadata if isinstance(metadata, dict) else None,
    )
    await agent.build()
    return agent
