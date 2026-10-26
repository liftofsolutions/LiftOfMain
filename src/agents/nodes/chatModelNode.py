"""Chat model nodes: answer the conversation in a graph's `messages`.

    reply = getNodeManager().reserveNode(
        "openai_chat",
        nodeId="473b3fd6-5886-40fc-bd5a-a5d414591bf2",
        name="reply",
        config={"model": "gpt-5.5", "system_prompt": "You are Acme's bot."},
    )
    graph.add_node(reply.nodeId, reply)

One node per provider in `LLM/configLLM.json`: `openai_chat`,
`gemini_chat`, `deepseek_chat` and `xai_chat`. They take no credential:
each calls its provider with the platform's key, as `LLM` reads it from
the server's configuration.

Configuring a node builds its chat model from the config it is reserved
with; deconfiguring it deletes the model along with the config. The
config picks:

* `model` - one of its provider's models in `configLLM.json`. Any other
  raises `KeyError` as the node is reserved, so no node type can run a
  model not listed there.
* `system_prompt` - optional; the `chatAgent` prompt in
  `systemPrompt.json` when it is not set.
* `temperature`, `max_tokens` - optional; they replace the model's own
  settings from `configLLM.json`. OpenAI's gpt-5 models take only their
  default temperature, so LangChain drops any other. A reasoning model
  counts its reasoning in `max_tokens`.
* how much the model reasons before it answers - optional, its
  provider's own: `reasoning_effort` on OpenAI and Gemini - "minimal",
  "low", "medium" or "high"; `thinking` on DeepSeek - false
  to answer without thinking, which its forced tool calls, as a node
  picking where a run goes makes them, need.
* `rag_search` - optional; true lets the model search the agent's
  knowledge base, the rag its record names, which the agent hands the
  node as it reserves it, in `rag`: `{"project_id", "rag_id"}`.
* `pick_limits` - optional, for a node with several edges out: the most
  times a run's model picks each node, by id. `{"<draft>": 3}` sends a
  run back to draft three times at most.

Nothing else in the config reaches the model's settings: a `base_url`,
say, would send the platform's key somewhere else. A call sends the
system prompt, then the conversation's chat history - its summary after
the prompt, its messages next, as `chatHistoryNode` keeps them - then
every message in the state; it adds the model's answer to `messages`,
and keeps it as the run's `answer`, until a node answers after it.

A node with `rag` binds RAG_TOOL: the model may search the rag, a query
at a time, and answers from what it found. Each search, and its result,
is added to `messages` before the answer, so the nodes after it read
what the answer was made of. A model still searching after MAX_SEARCHES
rounds is asked to answer with what it has.

`choose` sends the same, for a node with several edges out, and has the
model answer as structured output, in the shape `nextStepOf` gives: its
answer, and the id of the node the conversation goes on to, out of the
choices it is given - so it can pick none but those. A graph runs such a
node as a `ChoosingNode`, which adds the answer to `messages` - not as
the run's answer: a picking node steers the run, as a reviewer saying
what to change does - and sends the run on to the node picked. It counts
its picks in its `nodeDetails`, `{"picks": {nodeId: count}}`, and offers
its model no node it has picked as often as its `pick_limits` say; with
one node left to pick, the model is not asked, and the run goes on
there - a reviewer that has sent a reply back as often as it may passes
the next one unread.
"""

import math
from typing import ClassVar, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import Command
from pydantic import BaseModel, Field, create_model

from agents.nodes.chatHistoryNode import historyMessagesOf, summaryOf
from agents.nodes.credential import Credential
from agents.nodes.nodeBase import NodeBase
from documentProcessing.service import searchRag
from LLM import ChatModel, buildChatModel

# The role whose system prompt is used when a node's config has none.
DEFAULT_ROLE = "chatAgent"

# The settings a node's config may change; the model's other settings are
# its own, from configLLM.json.
CONFIG_SETTINGS = ("temperature", "max_tokens")

# What a model picking where the conversation goes next reads of the two
# parts of its answer.
MESSAGE_ABOUT = "Your answer, added to the conversation."
NEXT_ABOUT = "Where the conversation goes next: the id of one of these."

# What a model reads of the summary of the conversation before its
# messages, after its system prompt.
SUMMARY_ABOUT = "The conversation before the messages below, summarised:"

# The config key of the most times a node picks each node it may.
PICK_LIMITS = "pick_limits"

# The tool a node with a rag searches it with, in LangChain's tool
# format; and the rounds of searches a model may make before it answers.
RAG_TOOL_NAME = "search_knowledge_base"
RAG_TOOL = {
    "type": "function",
    "function": {
        "name": RAG_TOOL_NAME,
        "description": (
            "Search the knowledge base: the parts of its documents that "
            "best answer the query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for, in a few words.",
                }
            },
            "required": ["query"],
        },
    },
}
MAX_SEARCHES = 3
NOTHING_FOUND = "Nothing in the knowledge base answers it."


def nextStepOf(choices: dict[str, str]) -> type[BaseModel]:
    """The shape a model answers in to pick one of `choices` - their ids,
    each with what the model reads of it: its answer, and the id it
    picks, which can be none but theirs."""
    choicesAbout = [
        f"{choiceId}: {about}" for choiceId, about in choices.items()
    ]
    return create_model(
        "NextStep",
        __doc__="Your answer, and where the conversation goes next.",
        message=(str, Field(description=MESSAGE_ABOUT)),
        next=(
            Literal[tuple(choices)],
            Field(description="\n".join([NEXT_ABOUT, *choicesAbout])),
        ),
    )


class ChatModelNode(NodeBase):
    """Answers the conversation with one of its provider's chat models."""

    # The provider, as configLLM.json and LangChain name it.
    provider: ClassVar[str] = ""
    # The config key of how much its models reason, as LangChain names
    # the setting; none, for a provider whose nodes do not set it.
    reasoningSetting: ClassVar[str | None] = None

    def __init__(self, nodeId: str, name: str) -> None:
        """The template node `nodeId`, named `name`, with no chat model
        until it is configured."""
        super().__init__(nodeId, name)
        self.chatModel: ChatModel | None = None

    def configure(
        self,
        credential: Credential | None = None,
        config: dict | None = None,
    ) -> None:
        """Take the config, then build the chat model it picks.

        Raises:
            KeyError: If the config names no model, or one `configLLM.json`
                does not list for the provider.
        """
        super().configure(credential, config)
        overrides = {
            setting: self.config[setting]
            for setting in CONFIG_SETTINGS
            if setting in self.config
        }
        overrides.update(self.reasoningOverrides())
        self.chatModel = buildChatModel(
            DEFAULT_ROLE, self.provider, self.config["model"], overrides
        )

    def reasoningOverrides(self) -> dict:
        """The model's settings for how much it reasons, as the config
        says: its `reasoningSetting`, if the config has it."""
        if self.reasoningSetting not in self.config:
            return {}
        return {self.reasoningSetting: self.config[self.reasoningSetting]}

    def deconfigure(self) -> None:
        """Delete the chat model, then the credential and config."""
        self.chatModel = None
        super().deconfigure()

    async def __call__(self, state: dict) -> dict:
        """Add the model's answer to the conversation in `messages` - its
        searches before it, if it has a rag - and keep it as the run's
        answer."""
        messages = self._messagesFor(state)
        if self.config.get("rag"):
            added = await self._answerWithSearches(messages)
        else:
            added = [await self.chatModel.llm.ainvoke(messages)]
        return {"messages": added, "answer": added[-1].text}

    async def _answerWithSearches(
        self, messages: list[BaseMessage]
    ) -> list[BaseMessage]:
        """The model's searches of the rag, each with what it found, then
        its answer: the messages it adds to `messages`."""
        added: list[BaseMessage] = []
        searcher = self.chatModel.llm.bind_tools([RAG_TOOL])
        for _ in range(MAX_SEARCHES):
            answer = await searcher.ainvoke([*messages, *added])
            added.append(answer)
            if not answer.tool_calls:
                return added
            for toolCall in answer.tool_calls:
                found = await self._search(toolCall)
                added.append(ToolMessage(found, tool_call_id=toolCall["id"]))
        answerer = self.chatModel.llm.bind_tools(
            [RAG_TOOL], tool_choice="none"
        )
        added.append(await answerer.ainvoke([*messages, *added]))
        return added

    async def _search(self, toolCall: dict) -> str:
        """What the rag holds for the query of `toolCall`, as the model
        reads it: the text of each part found, closest first."""
        query = toolCall["args"].get("query")
        if toolCall["name"] != RAG_TOOL_NAME or not isinstance(query, str):
            return f"No search but {RAG_TOOL_NAME}, given a query."
        rag = self.config["rag"]
        found = await searchRag(rag["project_id"], rag["rag_id"], query)
        if "error" in found:
            return found["error"]
        texts = [match["text"] for match in found["matches"]]
        return "\n\n".join(texts) or NOTHING_FOUND

    async def choose(
        self, state: dict, choices: dict[str, str]
    ) -> tuple[AIMessage, str]:
        """The model's answer to the conversation, and the id of the one of
        `choices` it picks for the conversation to go on to.

        `choices` holds the ids of the nodes it may pick, each with what
        the model reads of it.
        """
        nextStep = await self.chatModel.llm.with_structured_output(
            nextStepOf(choices)
        ).ainvoke(self._messagesFor(state))
        return AIMessage(nextStep.message), nextStep.next

    def _messagesFor(self, state: dict) -> list[BaseMessage]:
        """What the model is sent: the system prompt, the summary of the
        conversation's chat history, its messages, then every message in
        `state`."""
        chatHistory = state.get("chatHistory", {})
        systemPrompt = (
            self.config.get("system_prompt") or self.chatModel.systemPrompt
        )
        summary = summaryOf(chatHistory)
        if summary:
            systemPrompt = f"{systemPrompt}\n\n{SUMMARY_ABOUT}\n{summary}"
        return [
            SystemMessage(systemPrompt),
            *historyMessagesOf(chatHistory),
            *state["messages"],
        ]


class OpenAIChatNode(ChatModelNode):
    """A chat model node on OpenAI."""

    handler = "openai_chat"
    provider = "openai"
    reasoningSetting = "reasoning_effort"
    description = "Answers the conversation with an OpenAI chat model."


class GeminiChatNode(ChatModelNode):
    """A chat model node on Google's Gemini."""

    handler = "gemini_chat"
    provider = "google_genai"
    reasoningSetting = "reasoning_effort"
    description = "Answers the conversation with a Google Gemini chat model."


class DeepSeekChatNode(ChatModelNode):
    """A chat model node on DeepSeek."""

    handler = "deepseek_chat"
    provider = "deepseek"
    description = "Answers the conversation with a DeepSeek chat model."

    def reasoningOverrides(self) -> dict:
        """DeepSeek's thinking, on or off as the config's `thinking` says:
        sent in the request's body, as DeepSeek's API takes it."""
        if "thinking" not in self.config:
            return {}
        thinking = "enabled" if self.config["thinking"] else "disabled"
        return {"extra_body": {"thinking": {"type": thinking}}}


class XAIChatNode(ChatModelNode):
    """A chat model node on xAI."""

    handler = "xai_chat"
    provider = "xai"
    description = "Answers the conversation with an xAI Grok chat model."


class ChoosingNode:
    """A chat model node with several edges out, run so that it picks
    which of the nodes they lead to the run goes on to - each as often as
    its `pick_limits` let it.

        graph.add_node(
            node.nodeId,
            ChoosingNode(node, choices),
            destinations=tuple(choices),
        )
    """

    def __init__(self, node: ChatModelNode, choices: dict[str, str]) -> None:
        """`node`, picking out of `choices`: the ids of the nodes its edges
        lead to, each with what its model reads of that node."""
        self.node = node
        self.nodeId = node.nodeId
        self.choices = choices

    async def __call__(self, state: dict) -> Command:
        """The node's answer added to the conversation, the pick counted,
        and the run sent on to the node it picked: one of those it has
        picked fewer times than its limits say. With one of them left,
        the node is not asked: the run goes on there."""
        picks = state["nodeDetails"].get(self.nodeId, {}).get("picks", {})
        limits = self.node.config.get(PICK_LIMITS, {})
        choices = {
            choiceId: about
            for choiceId, about in self.choices.items()
            if picks.get(choiceId, 0) < limits.get(choiceId, math.inf)
        }
        update = {}
        if len(choices) == 1:
            [nextNodeId] = choices
        else:
            answer, nextNodeId = await self.node.choose(state, choices)
            update["messages"] = [answer]
        picks = {**picks, nextNodeId: picks.get(nextNodeId, 0) + 1}
        update["nodeDetails"] = {self.nodeId: {"picks": picks}}
        return Command(goto=nextNodeId, update=update)
