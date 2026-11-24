"""The guard node: a customer agent's guardrails, and its first search of
the rag, at once.

    input -> guard -> support -> output
               |
               +-> output   (turned away)

A guard is an OpenAI chat node with an edge to the customer agent and
one to the output node, so a graph runs it as a `ChoosingNode`: its
model picks where the run goes - the guardrail - a message to pass on,
or one to turn away with the guard's own reply. While its model picks,
the guard searches the agent's rag for the message as it came in, so
the search takes no time of its own:

* turned away - the guard's reply goes out as the agent's answer, and
  the search is dropped.
* passed on - the search, and what it found, are added to `messages`
  before the guard's reply, as the customer agent's own first search:
  it answers from them, and may search again, as a chat model node with
  a rag does.

Its config is an `openai_chat` node's - `model`, `system_prompt`,
`reasoning_effort` - with `rag_search`, so the agent hands it its rag; a
guard with no rag is refused as it is configured.
"""

import asyncio

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from agents.nodes.chatModelNode import RAG_TOOL_NAME, OpenAIChatNode
from agents.nodes.credential import Credential
from agents.nodes.specialNodes import OUTPUT_NODE_ID

# The id of the guard's search of the message, as the customer agent
# reads it.
GUARD_SEARCH_ID = "guard-search"


class GuardNode(OpenAIChatNode):
    """Checks a message, and searches the rag for it, at once."""

    handler = "openai_guard"
    description = "Checks the message, and searches the rag for it."
    searchesWhilePicking = True

    def configure(
        self,
        credential: Credential | None = None,
        config: dict | None = None,
    ) -> None:
        """Take the config, and build the model.

        Raises:
            ValueError: If the config hands it no rag to search.
            KeyError: If the config names no model, or one
                `configLLM.json` does not list.
        """
        super().configure(credential, config)
        if not self.config.get("rag"):
            raise ValueError(
                f"Guard node {self.name!r} has no rag: set rag_search."
            )

    async def choose(
        self, state: dict, choices: dict[str, str]
    ) -> tuple[list[BaseMessage], str]:
        """The messages the guard adds, and its pick of `choices`: its
        model's pick, and its search of the message, made at once. Passed
        on, the search and what it found, then the guard's reply; turned
        away, its reply alone."""
        search = {
            "name": RAG_TOOL_NAME,
            "args": {"query": state["inputMessage"]},
            "id": GUARD_SEARCH_ID,
        }
        (replied, nextNodeId), found = await asyncio.gather(
            super().choose(state, choices), self._search(search)
        )
        if nextNodeId == OUTPUT_NODE_ID:
            return replied, nextNodeId
        searched = [
            AIMessage("", tool_calls=[search]),
            ToolMessage(found, tool_call_id=GUARD_SEARCH_ID),
        ]
        return [*searched, *replied], nextNodeId
