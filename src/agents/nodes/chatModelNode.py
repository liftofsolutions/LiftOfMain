"""Chat model nodes: answer the conversation in a graph's `messages`.

    reply = getNodeManager().reserveNode(
        "openai_chat",
        nodeId="473b3fd6-5886-40fc-bd5a-a5d414591bf2",
        name="reply",
        config={"model": "gpt-5.5", "system_prompt": "You are Acme's bot."},
    )
    graph.add_node(reply.name, reply)

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
  default temperature, so LangChain drops any other.

Nothing else in the config reaches the model's settings: a `base_url`,
say, would send the platform's key somewhere else. A call sends the
system prompt, then every message in the state, and adds the model's
answer to `messages`.
"""

from typing import ClassVar

from langchain_core.messages import SystemMessage

from agents.nodes.credential import Credential
from agents.nodes.nodeBase import NodeBase
from LLM import ChatModel, buildChatModel

# The role whose system prompt is used when a node's config has none.
DEFAULT_ROLE = "chatAgent"

# The settings a node's config may change; the model's other settings are
# its own, from configLLM.json.
CONFIG_SETTINGS = ("temperature", "max_tokens")


class ChatModelNode(NodeBase):
    """Answers the conversation with one of its provider's chat models."""

    # The provider, as configLLM.json and LangChain name it.
    provider: ClassVar[str] = ""

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
        self.chatModel = buildChatModel(
            DEFAULT_ROLE, self.provider, self.config["model"], overrides
        )

    def deconfigure(self) -> None:
        """Delete the chat model, then the credential and config."""
        self.chatModel = None
        super().deconfigure()

    async def __call__(self, state: dict) -> dict:
        """Add the model's answer to the conversation in `messages`."""
        systemPrompt = (
            self.config.get("system_prompt") or self.chatModel.systemPrompt
        )
        answer = await self.chatModel.llm.ainvoke(
            [SystemMessage(systemPrompt), *state["messages"]]
        )
        return {"messages": [answer]}


class OpenAIChatNode(ChatModelNode):
    """A chat model node on OpenAI."""

    handler = "openai_chat"
    provider = "openai"
    description = "Answers the conversation with an OpenAI chat model."


class GeminiChatNode(ChatModelNode):
    """A chat model node on Google's Gemini."""

    handler = "gemini_chat"
    provider = "google_genai"
    description = "Answers the conversation with a Google Gemini chat model."


class DeepSeekChatNode(ChatModelNode):
    """A chat model node on DeepSeek."""

    handler = "deepseek_chat"
    provider = "deepseek"
    description = "Answers the conversation with a DeepSeek chat model."


class XAIChatNode(ChatModelNode):
    """A chat model node on xAI."""

    handler = "xai_chat"
    provider = "xai"
    description = "Answers the conversation with an xAI Grok chat model."
