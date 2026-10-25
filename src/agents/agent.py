"""An agent: who it is, the nodes it reserved, and the LangGraph graph its
template wires them into, which answers its queries.

    agent = await loadAgent(nodeManager, projectId, agentId)
    agent.isTriggerAvailable(triggerId)
    answer = await agent.query("Do you deliver to Pune?")
    agent.release()

An `Agent` holds who it is - agentId, name, projectId and templateId,
from its record in `agents` - and the node manager it reserves its nodes
from. `build` readies it to run, once:

1. It fetches its template from `agent_templates`, and parses the graph
   in it - below - checking all of it before anything is reserved.
2. It fetches, from `node_types`, the type of each node the graph runs:
   the handler that runs the node, and the default config the node's own
   is merged over.
3. It reserves those nodes from the node manager, each configured as it
   will run. A chat model takes no credential: it runs on the platform's
   key, which the node manager's chat model nodes hold.
4. It builds the LangGraph graph of them, wired as the template says,
   and keeps it compiled, in `graph`.

It keeps its template's triggers too: `isTriggerAvailable` says whether
one can start the agent - the template has it, enabled, on one of its
nodes.

`query` runs the graph on a message, and answers with what the run goes
out with. Queries at once share the graph, and the nodes in it. The
nodes stay reserved while the agent is kept: `release` hands them back
to the node manager - at once, or once its last running query is
answered - which wipes their credentials and config.

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
which takes the last message of the conversation as the answer.

A run goes on one node at a time. A node with one edge out goes on to
the node it leads to; a node with several picks one of the nodes they
lead to, run as a `ChoosingNode`. Its model answers as structured
output, in a fixed shape: its
answer, and the id of the node it picks, which can be none but theirs.
It reads each of them by its name and what it does - its config's
`description` - and the output node as the end of the run, its own
answer going out as the agent's. So an if-else is a node with an edge
each way, and a loop an edge back:

    "edges": [
        {"from": "input", "to": "<draft>"},
        {"from": "<draft>", "to": "<review>"},
        {"from": "<review>", "to": "<draft>"},
        {"from": "<review>", "to": "output"}
    ]

        review picks: draft, to write it again, or output.

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
  does not list.
"""

import asyncio
import logging

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents import agentStore
from agents.nodeManager import NodeManager, NodeManagerError
from agents.nodes import (
    INPUT_NODE_ID,
    OUTPUT_ABOUT,
    OUTPUT_NODE_ID,
    SPECIAL_NODE_IDS,
    ChoosingNode,
    InputNode,
    NodeBase,
    OutputNode,
)

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


class AgentState(MessagesState):
    """A run's state: the message it comes in with, the conversation, and
    the answer it goes out with."""

    inputMessage: str
    answer: str


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


def buildGraph(
    nodes: list[NodeBase], nextNodeIds: dict[str, list[str]]
) -> CompiledStateGraph:
    """The compiled LangGraph graph of `nodes`, each under its id, between
    the input node and the output node: each goes on to the node its one
    edge leads to, or picks one of several - by `nextNodeIds`, as
    `parseGraph` checked them - and no run goes past MAX_RUN_STEPS
    steps."""
    aboutById = {node.nodeId: aboutOf(node) for node in nodes}
    aboutById[OUTPUT_NODE_ID] = OUTPUT_ABOUT
    graph = StateGraph(AgentState)
    graph.add_edge(START, INPUT_NODE_ID)
    for node in [InputNode(), *nodes]:
        targets = nextNodeIds[node.nodeId]
        if len(targets) == 1:
            graph.add_node(node.nodeId, node)
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
    reserved, then run on queries."""

    def __init__(
        self,
        agentId: str,
        name: str,
        projectId: str,
        templateId: str,
        nodeManager: NodeManager,
    ) -> None:
        """An agent not built yet: who it is, and the node manager it
        reserves its nodes from."""
        self.agentId = agentId
        self.name = name
        self.projectId = projectId
        self.templateId = templateId
        self.nodeManager = nodeManager
        # Its reserved nodes, and the graph built of them.
        self.nodes: list[NodeBase] = []
        self.graph: CompiledStateGraph | None = None
        # The ids of the triggers that can start it.
        self.triggerIds: set[str] = set()
        # Queries running now: a released agent hands its nodes back once
        # none is.
        self.runningQueries = 0
        self.isReleased = False

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
        try:
            for node, nodeType in typedNodes:
                self.nodes.append(self._reserve(node, nodeType))
            self.graph = buildGraph(self.nodes, nextNodeIds)
        except Exception:
            self._returnNodes()
            raise
        self.triggerIds = availableTriggersOf(template)
        LOGGER.info("Built %s.", self)

    def isTriggerAvailable(self, triggerId: str) -> bool:
        """Whether the trigger `triggerId` can start the agent: its
        template has it, enabled, on one of its nodes."""
        return triggerId in self.triggerIds

    async def query(self, message: str) -> str:
        """The agent's answer to `message`: the message comes in at the
        input node, and the answer is what goes out at the output node.

        Raises:
            AgentNotRunnableError: If the run goes on past MAX_RUN_STEPS
                steps without reaching the output node, as a loop whose
                node never picks the way out does.
        """
        self.runningQueries += 1
        try:
            state = await self.graph.ainvoke({"inputMessage": message})
        except GraphRecursionError as error:
            raise AgentNotRunnableError(
                f"A run of agent {self.agentId!r} went past {MAX_RUN_STEPS} "
                f"steps without reaching {OUTPUT_NODE_ID!r}."
            ) from error
        finally:
            self.runningQueries -= 1
            if self.isReleased and self.runningQueries == 0:
                self._returnNodes()
        return state["answer"]

    def release(self) -> None:
        """Let the agent go: its nodes go back to the node manager at once,
        or, while queries run, once the last of them is answered."""
        self.isReleased = True
        if self.runningQueries == 0:
            self._returnNodes()

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

    def _reserve(self, node: dict, nodeType: dict) -> NodeBase:
        """The node, reserved from the node manager, its config merged
        over its type's; a chat model takes no credential."""
        try:
            return self.nodeManager.reserveNode(
                nodeType["handler"],
                nodeId=node["id"],
                name=node["name"],
                config=mergeConfig(
                    nodeType.get("config", {}), node.get("config", {})
                ),
            )
        except (NodeManagerError, KeyError) as error:
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
    agent = Agent(
        agentId, record["name"], projectId, record["template_id"], nodeManager
    )
    await agent.build()
    return agent
