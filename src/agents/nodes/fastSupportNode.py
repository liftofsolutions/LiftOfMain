"""TEMPORARY - an experiment in answering faster, not a node to keep yet:
a guard and a support node in one, whose guard and rag search run at
the same time.

    input -> fast_support -> history -> output

The node does the guard's work and support's. It sends the message to
its guard model - DeepSeek, thinking off - which passes it on or turns
it away, and meanwhile searches the agent's rag for the message, as it
came in. Turned away, the guard's message is the node's answer, and
nothing more is asked. Passed on, the search is added to `messages` as
support's own first search, and its support model answers from it - or
searches again, as a chat model node with a rag does: in most runs one
call to the support model, not two, after the guard and the search,
which take about as long as the slower of them.

Its config is an `openai_chat` node's - its model, `system_prompt`,
`max_tokens`, `rag_search` - with `guard`: `{"model", "system_prompt"}`,
the guard's DeepSeek model and prompt.
"""

import asyncio

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)

from agents.nodes.chatHistoryNode import historyMessagesOf, summaryOf
from agents.nodes.chatModelNode import (
    DEFAULT_ROLE,
    RAG_TOOL_NAME,
    SUMMARY_ABOUT,
    OpenAIChatNode,
    nextStepOf,
)
from agents.nodes.credential import Credential
from agents.nodes.specialNodes import OUTPUT_ABOUT, OUTPUT_NODE_ID
from LLM import ChatModel, buildChatModel

# The guard's provider, and its thinking off, as its picks need.
GUARD_PROVIDER = "deepseek"
GUARD_SETTINGS = {"extra_body": {"thinking": {"type": "disabled"}}}
# What the guard picks between: support, or out, with its own message.
SUPPORT_CHOICE = "support"
GUARD_CHOICES = {
    SUPPORT_CHOICE: "support - Answers the customer.",
    OUTPUT_NODE_ID: OUTPUT_ABOUT,
}
# The id of the search of the message, made as the guard reads it.
EARLY_SEARCH_ID = "early-search"


class FastSupportNode(OpenAIChatNode):
    """A guard and support in one: the guard's pick and the rag search at
    once, then support's answer."""

    handler = "fast_support_chat"
    description = "Guards the message, and answers it from the rag."

    def __init__(self, nodeId: str, name: str) -> None:
        """The template node `nodeId`, named `name`, with no models until
        it is configured."""
        super().__init__(nodeId, name)
        self.guardModel: ChatModel | None = None

    def configure(
        self,
        credential: Credential | None = None,
        config: dict | None = None,
    ) -> None:
        """Build the support model, then the guard's.

        Raises:
            KeyError: If the config names no guard, or a model
                `configLLM.json` does not list.
        """
        super().configure(credential, config)
        self.guardModel = buildChatModel(
            DEFAULT_ROLE,
            GUARD_PROVIDER,
            self.config["guard"]["model"],
            GUARD_SETTINGS,
        )

    def deconfigure(self) -> None:
        """Delete the guard's model, then the rest."""
        self.guardModel = None
        super().deconfigure()

    async def __call__(self, state: dict) -> dict:
        """The guard's message, if it turns the message away; else the
        search of the message - made as the guard read it - and support's
        answer, added to `messages`, the answer kept as the run's."""
        search = {
            "name": RAG_TOOL_NAME,
            "args": {"query": state["inputMessage"]},
            "id": EARLY_SEARCH_ID,
        }
        (guardMessage, nextNodeId), found = await asyncio.gather(
            self._guard(state), self._search(search)
        )
        if nextNodeId != SUPPORT_CHOICE:
            return {"messages": [guardMessage], "answer": guardMessage.text}
        searched = [
            AIMessage("", tool_calls=[search]),
            ToolMessage(found, tool_call_id=EARLY_SEARCH_ID),
        ]
        answered = await self._answerWithSearches(
            [*self._messagesFor(state), *searched]
        )
        added = [*searched, *answered]
        return {"messages": added, "answer": added[-1].text}

    async def _guard(self, state: dict) -> tuple[AIMessage, str]:
        """The guard's message, and its pick of GUARD_CHOICES, sent what a
        chat model node is - its own prompt first."""
        nextStep = await self.guardModel.llm.with_structured_output(
            nextStepOf(GUARD_CHOICES)
        ).ainvoke(self._guardMessagesFor(state))
        return AIMessage(nextStep.message), nextStep.next

    def _guardMessagesFor(self, state: dict) -> list[BaseMessage]:
        """The guard's prompt, the summary of the chat history, its
        messages, then every message in `state`."""
        chatHistory = state.get("chatHistory", {})
        systemPrompt = self.config["guard"]["system_prompt"]
        summary = summaryOf(chatHistory)
        if summary:
            systemPrompt = f"{systemPrompt}\n\n{SUMMARY_ABOUT}\n{summary}"
        return [
            SystemMessage(systemPrompt),
            *historyMessagesOf(chatHistory),
            *state["messages"],
        ]
