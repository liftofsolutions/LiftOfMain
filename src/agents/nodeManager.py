"""The nodes this process reserves for agents, and the credentials they carry.

    nodeManager = getNodeManager()
    reply = nodeManager.reserveNode(
        "openai_chat",
        nodeId=templateNode["id"],
        name=templateNode["name"],
        config=replyConfig,
    )
    try:
        graph = StateGraph(MessagesState)
        graph.add_node(reply.name, reply)
        graph.add_edge(START, reply.name)
        state = await graph.compile().ainvoke(state)
    finally:
        nodeManager.releaseNode(reply)

`reserveNode` hands out a node of the kind the handler names that is the
template node `nodeId`, named `name`, configured with the borrower's
credential and config. A free node that is that template node is
reserved if there is one; when none is, a new one is created as that
template node, for good, saved in the manager's `nodes`, and reserved.
`releaseNode` deconfigures it, deleting the credential and config, and
keeps it free for the template node's next reservation. A node is a
LangGraph node, so an agent's graph is built for one run from the nodes
it reserves - see `agents.nodes.nodeBase`. The kinds of node are
`agents.nodes.NODE_CLASSES`.

A reservation is checked before a node is taken. The template node needs
an id, and its name must be a slug - letters, digits, `_` and `-`,
starting with a letter or `_`, at most 64 - as a LangGraph node name and
a model's tool name have to be. A credential must be a `Credential`, so
it is never shown; a plain dict is refused. A node that requires a
credential is refused without one, and a node that takes none is refused
with one. Any of these, or a handler no node class has, raises
`NodeManagerError`. An empty credential counts as none. A node that
fails to configure - a chat model node given a model not listed, say -
raises its own error, and is deconfigured and kept free.

So the nodes grow with how many agents need them at once; only the free
ones are held to a limit. Each handler keeps up to `maxFreeNodes` free
nodes - MAX_FREE_NODES unless the manager is built with another limit -
whichever template nodes they are. A node released when its handler has
that many already is kept, and the one free longest is destroyed
instead, leaving `nodes` for good: the free nodes are those of the
template nodes reserved most lately. Creating and destroying a node are
logged at INFO, reserving and releasing one at DEBUG.

A reserved node is its borrower's alone until it is released, and
nothing one borrower brought is ever seen by the next. A node released
twice, or not reserved here, is refused with False: it can never be
reserved by two borrowers at once. Drop a node once it is released - it
may already be reserved again, with another borrower's credential.

Every call takes constant time and never waits, apart from a node's own
`configure`, `deconfigure` and `destroy`. One manager per process -
`getNodeManager`. Its calls run on the event loop, as `JobManager`'s do,
so it needs no lock.
"""

import logging
import re
from collections import defaultdict

from agents.nodes import NODE_CLASSES, Credential, NodeBase

LOGGER = logging.getLogger(__name__)

# Free nodes kept per handler; past this, the one free longest is
# destroyed.
MAX_FREE_NODES = 5

# A node's name, as a LangGraph node name and every provider's tool name
# take it: Gemini's must start with a letter or `_`.
SLUG_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}")


class NodeManagerError(RuntimeError):
    """A node could not be reserved as it was asked for."""


def freeKeyOf(handler: str, nodeId: str, name: str) -> tuple[str, str, str]:
    """What free nodes are kept under: the template node they are.

    The handler and name as well as the id: a template node renamed, or
    moved onto another node type, keeps its id.
    """
    return (handler, nodeId, name)


class NodeManager:
    """Reserves nodes, and keeps the free ones for their template nodes."""

    def __init__(
        self,
        nodeClasses: dict[str, type[NodeBase]] = NODE_CLASSES,
        maxFreeNodes: int = MAX_FREE_NODES,
    ) -> None:
        """A manager of the node classes in `nodeClasses`, by handler, that
        keeps up to `maxFreeNodes` free nodes of each; none built yet."""
        self.nodeClasses = nodeClasses
        self.maxFreeNodes = maxFreeNodes
        # Every node created and not destroyed: reserved or free.
        self.nodes: set[NodeBase] = set()
        # Free nodes, under `freeKeyOf` the template node they are.
        self.freeNodes: dict[tuple[str, str, str], list[NodeBase]] = {}
        # The free nodes of each handler, the one free longest first.
        self.freeNodesByHandler: defaultdict[str, dict[NodeBase, None]] = (
            defaultdict(dict)
        )
        # Reserved and not released yet.
        self.reservedNodes: set[NodeBase] = set()

    def reserveNode(
        self,
        handler: str,
        *,
        nodeId: str,
        name: str,
        credential: Credential | None = None,
        config: dict | None = None,
    ) -> NodeBase:
        """The template node `nodeId`, named `name`, as a node of
        `handler`, configured with `credential` and `config`.

        A free node that is that template node if there is one, else a
        new one, created and saved in `nodes`.

        Raises:
            NodeManagerError: If no node class has `handler`, there is no
                `nodeId`, `name` is not a slug, or the credential is not a
                `Credential` or does not suit the node's
                `requiresCredential`.
            Exception: Whatever the node's `configure` raises, once the
                node is deconfigured and kept free.
        """
        nodeClass = self.nodeClasses.get(handler)
        if nodeClass is None:
            raise NodeManagerError(f"No node has the handler {handler!r}.")
        if not nodeId:
            raise NodeManagerError(
                f"A {handler} node needs its template node's id."
            )
        if not SLUG_PATTERN.fullmatch(name or ""):
            raise NodeManagerError(
                f"Node name {name!r} is not a slug: letters, digits, _ and "
                "-, starting with a letter or _, at most 64."
            )
        if credential is not None and not isinstance(credential, Credential):
            raise NodeManagerError(
                "A credential must be a Credential, which is never shown."
            )
        if nodeClass.requiresCredential and not credential:
            raise NodeManagerError(f"A {handler} node needs a credential.")
        if credential and not nodeClass.requiresCredential:
            raise NodeManagerError(f"A {handler} node takes no credential.")

        node = self._takeFreeNode(freeKeyOf(handler, nodeId, name))
        if node is None:
            node = self._createNode(nodeClass, nodeId, name)
        try:
            node.configure(credential, config)
        except Exception:
            node.deconfigure()
            self._keepFree(node)
            raise
        self.reservedNodes.add(node)
        LOGGER.debug("Reserved %s.", node)
        return node

    def releaseNode(self, node: NodeBase) -> bool:
        """Deconfigure `node`, deleting its credential and config, and
        keep it free for its template node's next reservation.

        False, and nothing changed, if `node` is not reserved here: it was
        released already, or never reserved from this manager.
        """
        if node not in self.reservedNodes:
            LOGGER.error("%s is not reserved here; not released.", node)
            return False
        LOGGER.debug("Releasing %s.", node)
        self.reservedNodes.discard(node)
        node.deconfigure()
        self._keepFree(node)
        return True

    def _createNode(
        self, nodeClass: type[NodeBase], nodeId: str, name: str
    ) -> NodeBase:
        """A new node of `nodeClass`, the template node `nodeId`, named
        `name`, saved in `nodes`."""
        node = nodeClass(nodeId, name)
        self.nodes.add(node)
        LOGGER.info("Created a new node: %s.", node)
        return node

    def _takeFreeNode(self, freeKey: tuple[str, str, str]) -> NodeBase | None:
        """A free node kept under `freeKey`, no longer free; None if there
        is none."""
        freeNodes = self.freeNodes.get(freeKey)
        if not freeNodes:
            return None
        node = freeNodes[-1]
        self._removeFreeNode(node)
        return node

    def _keepFree(self, node: NodeBase) -> None:
        """Keep `node` free; then, if its handler has more than
        `maxFreeNodes` free, destroy the one free longest, which leaves
        `nodes`."""
        handlerNodes = self.freeNodesByHandler[node.handler]
        handlerNodes[node] = None
        freeKey = freeKeyOf(node.handler, node.nodeId, node.name)
        self.freeNodes.setdefault(freeKey, []).append(node)
        if len(handlerNodes) > self.maxFreeNodes:
            oldest = next(iter(handlerNodes))
            self._removeFreeNode(oldest)
            self.nodes.discard(oldest)
            oldest.destroy()
            LOGGER.info("Destroyed %s.", oldest)

    def _removeFreeNode(self, node: NodeBase) -> None:
        """Take the free `node` out of the free nodes."""
        freeKey = freeKeyOf(node.handler, node.nodeId, node.name)
        freeNodes = self.freeNodes[freeKey]
        freeNodes.remove(node)
        if not freeNodes:
            del self.freeNodes[freeKey]
        del self.freeNodesByHandler[node.handler][node]


_NODE_MANAGER: NodeManager | None = None


def getNodeManager() -> NodeManager:
    """The process wide node manager, built once and shared."""
    global _NODE_MANAGER
    if _NODE_MANAGER is None:
        _NODE_MANAGER = NodeManager()
    return _NODE_MANAGER
