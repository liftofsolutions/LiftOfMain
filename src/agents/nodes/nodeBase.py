"""The base every node is built on: what an agent reserves, LangGraph runs.

A node is a LangGraph node. `graph.add_node(node.nodeId, node)` adds it
to a graph, which awaits it with the graph's state and merges the dict
it returns into that state.

    class WhatsAppNode(NodeBase):
        handler = "whatsapp"
        requiresCredential = True

        async def __call__(self, state: dict) -> dict:
            ...  # send the last message, with self._credential.reveal()
            return {"sent": [...]}

Each kind of node is a subclass that sets `handler`, as `node_types`
records name it in `db/agent-schema`, and writes `__call__`: its work on
the graph's state. `requiresCredential` says what an agent must bring to
reserve one:

* True - a credential, always. Asked for without one, it is refused.
* False, the default - none. Asked for with one, it is refused, so a
  credential never reaches a node that does not use it.

A node is built as one node of an agent's template, for good: its
`nodeId` and `name` are that node's `id` and `name` in `agent_templates`,
and neither can change. The name is a slug, such as `reply` or
`web_search`: the tool name a model knows it by. A graph holds the node
under its id, as two nodes of a template may share a name.

What changes is what a node holds for one reservation. The `NodeManager`
reserves nodes and releases them: reserving one calls `configure`, which
gives it its own copies of the borrower's credential and config, and
releasing it calls `deconfigure`, which deletes both before anyone else
can reserve it. A subclass that builds something from them - a client, a
model - builds it in `configure`, after the base's, and deletes it in
`deconfigure`, before the base's. Anything a node keeps from one
reservation to the next must never be built from a credential, and is
freed in `destroy`, called when the manager does not keep the node.

A credential is a `Credential`, which is never shown, and the node keeps
it protected in `_credential`: its own code reads a value with
`self._credential.reveal(key)`, at the moment a call needs it. Nothing a
node shows - its repr, its logs, its `about` - says anything of it.

Each node also says what it does, for a model to read: its class's
`description`, or the `description` in its config, when a project gives
its own node one. `about()` puts the node in LangChain's tool format, so
binding it to a chat model - `llm.bind_tools([node.about()])` - is how
the model learns the node is there, what it is called and what it does,
and can ask for it with a tool call. A node picking where a run goes
next reads the nodes it may pick the same way: by name, and what they
do.

So an agent's graph is built of the nodes it reserves, and kept with
them: they stay reserved to that agent, and no one else, until it
releases them.
"""

import copy
from abc import ABC, abstractmethod
from typing import ClassVar

from agents.nodes.credential import Credential


class NodeBase(ABC):
    """One node of an agent's template, reserved by one agent at a time."""

    handler: ClassVar[str] = ""
    requiresCredential: ClassVar[bool] = False
    # What the node does, as a model reads it to know when to use it.
    description: ClassVar[str] = ""
    # Whether it runs after the run replies, its work not delaying the
    # answer: see `agents.nodes.specialNodes`.
    runsAfterReply: ClassVar[bool] = False

    def __init__(self, nodeId: str, name: str) -> None:
        """The template node `nodeId`, named `name`, for good; with no
        credential or config until it is configured."""
        self._nodeId = nodeId
        self._name = name
        self._credential: Credential | None = None
        self.config: dict = {}

    @property
    def nodeId(self) -> str:
        """The template node's id. It never changes."""
        return self._nodeId

    @property
    def name(self) -> str:
        """The template node's name, a slug. It never changes."""
        return self._name

    def configure(
        self,
        credential: Credential | None = None,
        config: dict | None = None,
    ) -> None:
        """Take its own copies of `credential` and `config`, for one
        reservation.

        Copies, so neither the borrower nor the node can change the
        other's. An empty credential is kept as none.
        """
        self._credential = credential.copy() if credential else None
        self.config = copy.deepcopy(config or {})

    def deconfigure(self) -> None:
        """Delete the credential and config; the node is idle again.

        Both are cleared, not only dropped, so anything still holding them
        holds nothing. They are the node's own copies, so the borrower's
        are left as they were. Its id and name stay.
        """
        if self._credential is not None:
            self._credential.clear()
        self._credential = None
        self.config.clear()

    def destroy(self) -> None:
        """Free what the node keeps from one reservation to the next.

        Called on a released node the manager does not keep. A node keeps
        nothing by default, so there is nothing to free.
        """

    def about(self) -> dict:
        """The node, described for a model, in LangChain's tool format.

        `bind_tools` on a LangChain chat model sends it to the model, which
        then knows the node by its `name`, and what it does: its config's
        `description`, if it has one, else its class's. It takes no
        arguments from the model.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    self.config.get("description") or self.description
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        }

    @abstractmethod
    async def __call__(self, state: dict) -> dict:
        """Do the node's work on the graph's `state`; the part that changes.

        LangGraph merges the returned dict into the state, by the state's
        own rules - `messages` are added to, for instance.
        """

    def __repr__(self) -> str:
        """Which node it is, and its config.

        Nothing of its credential, not even whether it holds one, so a
        node can be logged.
        """
        return (
            f"{type(self).__name__}(handler={self.handler!r}, "
            f"name={self.name!r}, nodeId={self.nodeId!r}, "
            f"config={self.config!r})"
        )
