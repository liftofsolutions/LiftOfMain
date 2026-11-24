"""The nodes this process hands agents, and the credentials they carry.

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

`reserveNode` builds a node on demand: a new node of the kind the
handler names, as the template node `nodeId`, named `name`, configured
with the borrower's credential and config. `releaseNode` deconfigures
it, deleting the credential and config; the borrower then drops it. The
manager keeps no node, free or reserved: each reservation is a node of
its own, so nothing one borrower brought is ever seen by the next. A
node is a LangGraph node, so an agent's graph is built for one run from
the nodes it reserves - see `agents.nodes.nodeBase`. The kinds of node
are `agents.nodes.NODE_CLASSES`.

A reservation is checked before a node is built. The template node needs
an id, and its name must be a slug - letters, digits, `_` and `-`,
starting with a letter or `_`, at most 64 - as a LangGraph node name and
a model's tool name have to be. A credential must be a `Credential`, so
it is never shown; a plain dict is refused. A node that requires a
credential is refused without one, and a node that takes none is refused
with one. Any of these, or a handler no node class has, raises
`NodeManagerError`. An empty credential counts as none. A node that
fails to configure - a chat model node given a model not listed, say -
raises its own error, and is deconfigured.

Reserving and releasing a node are logged at DEBUG. Every call takes
constant time and never waits, apart from a node's own `configure` and
`deconfigure`. One manager per process - `getNodeManager`.
"""

import logging
import re

from agents.nodes import NODE_CLASSES, Credential, NodeBase

LOGGER = logging.getLogger(__name__)

# A node's name, as a LangGraph node name and every provider's tool name
# take it: Gemini's must start with a letter or `_`.
SLUG_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}")


class NodeManagerError(RuntimeError):
    """A node could not be reserved as it was asked for."""


class NodeManager:
    """Builds nodes on demand, and wipes them as they are released."""

    def __init__(
        self, nodeClasses: dict[str, type[NodeBase]] = NODE_CLASSES
    ) -> None:
        """A manager of the node classes in `nodeClasses`, by handler."""
        self.nodeClasses = nodeClasses

    def reserveNode(
        self,
        handler: str,
        *,
        nodeId: str,
        name: str,
        credential: Credential | None = None,
        config: dict | None = None,
    ) -> NodeBase:
        """A new node of `handler`, the template node `nodeId`, named
        `name`, configured with `credential` and `config`.

        Raises:
            NodeManagerError: If no node class has `handler`, there is no
                `nodeId`, `name` is not a slug, or the credential is not a
                `Credential` or does not suit the node's
                `requiresCredential`.
            Exception: Whatever the node's `configure` raises, once the
                node is deconfigured.
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

        node = nodeClass(nodeId, name)
        try:
            node.configure(credential, config)
        except Exception:
            node.deconfigure()
            raise
        LOGGER.debug("Reserved %s.", node)
        return node

    def releaseNode(self, node: NodeBase) -> None:
        """Deconfigure `node`, deleting its credential and config; drop it
        then."""
        LOGGER.debug("Releasing %s.", node)
        node.deconfigure()


_NODE_MANAGER: NodeManager | None = None


def getNodeManager() -> NodeManager:
    """The process wide node manager, built once and shared."""
    global _NODE_MANAGER
    if _NODE_MANAGER is None:
        _NODE_MANAGER = NodeManager()
    return _NODE_MANAGER
