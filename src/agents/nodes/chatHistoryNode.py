"""The chat history node: keeps a conversation's turns in its chat
history, and summarises them once they grow long.

    {
        "node_type_id": "<chat_history>",
        "name": "history",
        "config": {
            "provider": "openai",
            "model": "gpt-5.4-mini",
            "summarise_after_tokens": 10000
        }
    }

The history it keeps, in the state's `chatHistory`:

    {
        "summary": "The customer asked whether Acme delivers to Pune.",
        "messages": [
            {"role": "human", "text": "How long does it take?"},
            {"role": "ai", "text": "Two to three business days."}
        ]
    }

Each run it is in adds the turn to `messages`: the message the run came
in with, and the run's answer, as `answerOf` gives it. It then counts
the tokens of the whole history, its summary and every message; past
`summarise_after_tokens` - SUMMARISE_AFTER_TOKENS, unless the config says
- its model, on the `summarizer` prompt, summarises all of it but the
last message, the old summary with it, and the history keeps that
summary, and the last message alone.

Its model is any one `configLLM.json` lists, on the platform's key: the
config's `provider` and `model`. It takes no credential.

It runs after the run replies (`runsAfterReply`): the asker has the
answer before the turn is kept, summarised or saved.

Every chat model node reads the history this node keeps: its summary,
`summaryOf`, after its system prompt, and its messages,
`historyMessagesOf`, before the run's own.
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agents.nodes.credential import Credential
from agents.nodes.nodeBase import NodeBase
from agents.nodes.specialNodes import answerOf
from documentProcessing.documentAnalyzer import countTokens
from LLM import ChatModel, buildChatModel

# The role whose system prompt summarises the history.
SUMMARY_ROLE = "summarizer"

# The tokens a history may hold before it is summarised, unless a node's
# config says another number.
SUMMARISE_AFTER_TOKENS = 10_000

# Who says a message of the history: the customer, or the agent.
HUMAN_ROLE = "human"
AI_ROLE = "ai"
MESSAGE_CLASSES = {HUMAN_ROLE: HumanMessage, AI_ROLE: AIMessage}


def summaryOf(chatHistory: dict) -> str | None:
    """The summary of the conversation before its messages, if any."""
    return chatHistory.get("summary")


def historyMessagesOf(chatHistory: dict) -> list[BaseMessage]:
    """The messages of the history, as a chat model reads them."""
    return [
        MESSAGE_CLASSES[message["role"]](message["text"])
        for message in chatHistory.get("messages", [])
    ]


def transcriptOf(summary: str | None, messages: list[dict]) -> str:
    """The history as text: its summary, then each message, by who said
    it."""
    lines = [f"{message['role']}: {message['text']}" for message in messages]
    if summary:
        lines.insert(0, f"Summary so far: {summary}")
    return "\n".join(lines)


class ChatHistoryNode(NodeBase):
    """Keeps the conversation's turns, summarised once they grow long."""

    handler = "chat_history"
    description = "Keeps the conversation, summarised once it grows long."
    runsAfterReply = True

    def __init__(self, nodeId: str, name: str) -> None:
        """The template node `nodeId`, named `name`, with no model until
        it is configured."""
        super().__init__(nodeId, name)
        self.chatModel: ChatModel | None = None

    def configure(
        self,
        credential: Credential | None = None,
        config: dict | None = None,
    ) -> None:
        """Take the config, then build the model that summarises.

        Raises:
            KeyError: If the config names no provider or model, or one
                `configLLM.json` does not list.
            ValueError: If `summarise_after_tokens` is not a count of 1
                or more.
        """
        super().configure(credential, config)
        tokens = self.summariseAfterTokens()
        isCount = isinstance(tokens, int) and not isinstance(tokens, bool)
        if not isCount or tokens < 1:
            raise ValueError(
                f"summarise_after_tokens is {tokens!r}, not a count of 1 or "
                "more."
            )
        self.chatModel = buildChatModel(
            SUMMARY_ROLE, self.config["provider"], self.config["model"]
        )

    def deconfigure(self) -> None:
        """Delete the model, then the credential and config."""
        self.chatModel = None
        super().deconfigure()

    def summariseAfterTokens(self) -> int:
        """The tokens the history may hold before it is summarised."""
        return self.config.get(
            "summarise_after_tokens", SUMMARISE_AFTER_TOKENS
        )

    async def __call__(self, state: dict) -> dict:
        """The history with the run's turn added, summarised if it is now
        past its tokens."""
        chatHistory = state["chatHistory"]
        summary = summaryOf(chatHistory)
        messages = [
            *chatHistory.get("messages", []),
            {"role": HUMAN_ROLE, "text": state["inputMessage"]},
            {"role": AI_ROLE, "text": answerOf(state)},
        ]
        tokens = countTokens(transcriptOf(summary, messages))
        if tokens > self.summariseAfterTokens():
            summarised = await self.chatModel.generateResponse(
                transcriptOf(summary, messages[:-1]), ""
            )
            summary, messages = summarised.text, messages[-1:]
        return {
            "chatHistory": {
                **chatHistory,
                "summary": summary,
                "messages": messages,
            }
        }
