"""Btest for `agents.nodes.chatModelNode`, the chat model nodes.

No provider is called. `init_chat_model`, where `LLM` hands over to the
provider's SDK, is replaced by `RecordingModel`, which answers every call
with ANSWER and keeps what it was built with and what it was sent; and
the platform's keys are test ones. Everything up to that is real: the
node manager, `buildChatModel`, `configLLM.json` and `systemPrompt.json`.
Every node is reserved as the example template's reply node.

1. Each provider's handler reserves its own node, which takes no
   credential and says what it does.
2. Reserving a node builds its model, once, from the config: its model,
   and its temperature and max_tokens over the settings in
   `configLLM.json`. Nothing else in the config reaches the model's
   settings, and a model not listed there is refused as the node is
   reserved, leaving the node deconfigured and free.
3. A call sends the system prompt, then the conversation, and adds the
   answer to `messages`; with no prompt in the config, the `chatAgent`
   prompt is sent.
4. Choosing sends the same, and has the model answer in a shape: its
   answer, and the id of one of the choices - each read with what it is
   - and no other.
5. The node runs as the node of a LangGraph graph, under its name, and
   releasing it deletes its model, credential and config, and keeps its
   id and name.

    uv run pytest Btest/agents -q
"""

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import ValidationError

from agents.nodeManager import NodeManager
from agents.nodes import (
    ChatModelNode,
    DeepSeekChatNode,
    GeminiChatNode,
    OpenAIChatNode,
    XAIChatNode,
)
from agents.nodes.chatModelNode import NEXT_ABOUT, nextStepOf
from config import ServerConfig
from LLM import agentChatModel

# The example template's reply node.
REPLY_NODE_ID = "473b3fd6-5886-40fc-bd5a-a5d414591bf2"

QUESTION = "Do you deliver to Pune?"
ANSWER = "Yes, we deliver to Pune in 2-3 business days."
SYSTEM_PROMPT = "You are Acme's sales assistant. Answer briefly."
PLATFORM_KEYS = {"openai": "platform-openai-key"}

HANDLER_CASES = [
    # handler, its node class, its provider, one of its models
    ("openai_chat", OpenAIChatNode, "openai", "gpt-5.5"),
    ("gemini_chat", GeminiChatNode, "google_genai", "gemini-3.5-flash"),
    ("deepseek_chat", DeepSeekChatNode, "deepseek", "deepseek-v4-flash"),
    ("xai_chat", XAIChatNode, "xai", "grok-4.5"),
]

# Where a node may send the conversation next: each id, with what the
# model reads of it.
CHOICES = {
    "e3877987-c576-4a8e-8763-a8a498d2f775": "handoff - Hands it to a person.",
    "output": "the end of the run.",
}

# What a released reply node is left holding: its id and name, alone.
RELEASED_NODE = {
    "_nodeId": REPLY_NODE_ID,
    "_name": "reply",
    "_credential": None,
    "config": {},
    "chatModel": None,
}


class RecordingModel:
    """Stands in for a provider's chat model.

    Answers every call with ANSWER, and keeps the model and settings it
    was built with, and every list of messages it was sent. Asked to
    answer in a shape, it keeps the shape, and picks `pick` - a test sets
    it.
    """

    def __init__(self, model: str, settings: dict) -> None:
        """A model built as `init_chat_model(model, **settings)` was."""
        self.model = model
        self.settings = settings
        self.received: list[list] = []
        self.shapes: list[type] = []
        self.pick: str | None = None

    async def ainvoke(self, messages: list) -> AIMessage:
        """Keep `messages`, and answer with ANSWER."""
        self.received.append(list(messages))
        return AIMessage(ANSWER)

    def with_structured_output(self, shape: type) -> "ShapedModel":
        """The model, answering in `shape`."""
        self.shapes.append(shape)
        return ShapedModel(self, shape)


class ShapedModel:
    """A `RecordingModel` answering in a shape: ANSWER, and its pick."""

    def __init__(self, model: RecordingModel, shape: type) -> None:
        """`model`, answering in `shape`."""
        self.model = model
        self.shape = shape

    async def ainvoke(self, messages: list) -> object:
        """Keep `messages`, and answer in the shape, with the pick."""
        self.model.received.append(list(messages))
        return self.shape(message=ANSWER, next=self.model.pick)


@pytest.fixture
def builtModels(monkeypatch) -> list[RecordingModel]:
    """Every model the nodes build, in order, in place of a real one."""
    models = []

    def buildRecordingModel(model: str, **settings) -> RecordingModel:
        recordingModel = RecordingModel(model, settings)
        models.append(recordingModel)
        return recordingModel

    serverConfig = ServerConfig(llmApiKeys=PLATFORM_KEYS)
    monkeypatch.setattr(
        agentChatModel, "getServerConfig", lambda: serverConfig
    )
    monkeypatch.setattr(
        agentChatModel, "init_chat_model", buildRecordingModel
    )
    return models


def buildReplyConfig(**overrides) -> dict:
    """The reply node's config, merged as `agents.example.json` merges it.

    The node type's model and settings, under the node's own temperature
    and system prompt.
    """
    config = {
        "model": "gpt-5.5",
        "temperature": 0.3,
        "max_tokens": 1024,
        "system_prompt": SYSTEM_PROMPT,
    }
    config.update(overrides)
    return config


def buildState(**overrides) -> dict:
    """A graph state holding the customer's one question."""
    state = {"messages": [HumanMessage(QUESTION)]}
    state.update(overrides)
    return state


def reserveReply(
    nodeManager: NodeManager, handler: str = "openai_chat", **overrides
) -> ChatModelNode:
    """The example template's reply node, reserved from `nodeManager` on
    `handler`, its config built with `overrides`."""
    return nodeManager.reserveNode(
        handler,
        nodeId=REPLY_NODE_ID,
        name="reply",
        config=buildReplyConfig(**overrides),
    )


class TestHandlers:
    """Each provider in `configLLM.json` has its own node."""

    @pytest.mark.parametrize(
        "handler, nodeClass, provider, model", HANDLER_CASES
    )
    def testHandlerReservesItsProvidersNode(
        self, builtModels, handler, nodeClass, provider, model
    ):
        node = reserveReply(NodeManager(), handler, model=model)

        assert type(node) is nodeClass
        assert node._credential is None
        [builtModel] = builtModels
        assert builtModel.model == model
        assert builtModel.settings["model_provider"] == provider

    def testNodeSaysWhatItDoes(self, builtModels):
        node = reserveReply(NodeManager())

        about = node.about()["function"]
        assert about["name"] == "reply"
        assert about["description"] == (
            "Answers the conversation with an OpenAI chat model."
        )


class TestModelSettings:
    """Which model is built as a node is reserved, and with what."""

    def testModelIsBuiltFromTheConfigWhenReserved(self, builtModels):
        reserveReply(NodeManager())

        [model] = builtModels
        assert model.model == "gpt-5.5"
        assert model.settings == {
            "model_provider": "openai",
            "max_tokens": 1024,
            "timeout": 60,
            "temperature": 0.3,
            "api_key": "platform-openai-key",
        }

    def testModelIsBuiltOncePerReservation(self, builtModels):
        node = reserveReply(NodeManager())

        asyncio.run(node(buildState()))
        asyncio.run(node(buildState()))

        [model] = builtModels
        assert len(model.received) == 2

    def testNothingElseInTheConfigReachesTheModelsSettings(
        self, builtModels
    ):
        reserveReply(
            NodeManager(),
            base_url="https://attacker.example/v1",
            api_key="attacker-key",
            timeout=600,
            description="Answers Acme's customers.",
        )

        [model] = builtModels
        assert "base_url" not in model.settings
        assert "description" not in model.settings
        assert model.settings["api_key"] == "platform-openai-key"
        assert model.settings["timeout"] == 60

    @pytest.mark.parametrize(
        ("handler", "model", "config", "reasoning"),
        [
            (
                "openai_chat",
                "gpt-5.4-nano",
                {"reasoning_effort": "minimal"},
                {"reasoning_effort": "minimal"},
            ),
            (
                "gemini_chat",
                "gemini-3.1-flash-lite",
                {"reasoning_effort": "minimal"},
                {"reasoning_effort": "minimal"},
            ),
            (
                "deepseek_chat",
                "deepseek-v4-flash",
                {"thinking": False},
                {"extra_body": {"thinking": {"type": "disabled"}}},
            ),
            (
                "deepseek_chat",
                "deepseek-v4-flash",
                {"thinking": True},
                {"extra_body": {"thinking": {"type": "enabled"}}},
            ),
            ("openai_chat", "gpt-5.4-nano", {}, {}),
            ("deepseek_chat", "deepseek-v4-flash", {}, {}),
            # Each provider's own setting only, and never a body of the
            # config's own.
            (
                "openai_chat",
                "gpt-5.4-nano",
                {"thinking": False, "extra_body": {"a": 1}},
                {},
            ),
            (
                "deepseek_chat",
                "deepseek-v4-flash",
                {"reasoning_effort": "low"},
                {},
            ),
            ("xai_chat", "grok-4.5", {"reasoning_effort": "low"}, {}),
        ],
        ids=[
            "openaiEffort",
            "geminiEffort",
            "deepseekThinkingOff",
            "deepseekThinkingOn",
            "openaiUnset",
            "deepseekUnset",
            "openaiGivenOthers",
            "deepseekGivenAnEffort",
            "xaiGivenAnEffort",
        ],
    )
    def testReasoningIsSetAsTheProviderTakesIt(
        self, builtModels, handler, model, config, reasoning
    ):
        reserveReply(NodeManager(), handler, model=model, **config)

        [built] = builtModels
        settings = {
            key: built.settings[key]
            for key in ("reasoning_effort", "extra_body", "thinking")
            if key in built.settings
        }
        assert settings == reasoning

    def testModelNotInConfigLLMIsRefusedWhenReserved(self, builtModels):
        nodeManager = NodeManager()

        with pytest.raises(KeyError):
            reserveReply(nodeManager, model="gpt-unlisted")

        assert builtModels == []
        assert nodeManager.reservedNodes == set()
        [node] = list(nodeManager.freeNodesByHandler["openai_chat"])
        assert vars(node) == RELEASED_NODE
        assert nodeManager.nodes == {node}


class TestCall:
    """What one call sends the model, and what it adds to the state."""

    def testAnswerIsAddedToTheMessagesAndKeptAsTheRunsAnswer(
        self, builtModels
    ):
        node = reserveReply(NodeManager())

        update = asyncio.run(node(buildState()))

        assert update == {"messages": [AIMessage(ANSWER)], "answer": ANSWER}

    def testSystemPromptThenTheConversationAreSent(self, builtModels):
        node = reserveReply(NodeManager())
        conversation = [
            HumanMessage("Hi"),
            AIMessage("Hello! How can I help?"),
            HumanMessage(QUESTION),
        ]

        asyncio.run(node({"messages": conversation}))

        [model] = builtModels
        assert model.received == [[SystemMessage(SYSTEM_PROMPT), *conversation]]

    def testChatAgentPromptIsSentWhenTheConfigHasNone(self, builtModels):
        node = reserveReply(NodeManager(), system_prompt=None)

        asyncio.run(node(buildState()))

        [model] = builtModels
        systemMessage = model.received[0][0]
        assert isinstance(systemMessage, SystemMessage)
        assert systemMessage.content.startswith(
            "You are a helpful assistant that answers from the context."
        )


class TestChoice:
    """A node picking where the conversation goes next."""

    def testChoiceIsTheModelsAnswerAndItsPick(self, builtModels):
        node = reserveReply(NodeManager())
        [model] = builtModels
        model.pick = "output"

        answer, picked = asyncio.run(node.choose(buildState(), CHOICES))

        assert (answer, picked) == (AIMessage(ANSWER), "output")
        assert model.received == [
            [SystemMessage(SYSTEM_PROMPT), HumanMessage(QUESTION)]
        ]

    def testModelIsAskedForItsAnswerAndOneOfTheChoices(self, builtModels):
        node = reserveReply(NodeManager())
        [model] = builtModels
        model.pick = "output"

        asyncio.run(node.choose(buildState(), CHOICES))

        [shape] = model.shapes
        schema = shape.model_json_schema()
        assert schema["required"] == ["message", "next"]
        assert schema["properties"]["message"]["type"] == "string"
        nextStep = schema["properties"]["next"]
        assert nextStep["enum"] == list(CHOICES)
        assert nextStep["description"].splitlines() == [
            NEXT_ABOUT,
            "e3877987-c576-4a8e-8763-a8a498d2f775: handoff - Hands it to a "
            "person.",
            "output: the end of the run.",
        ]

    def testPickOutsideTheChoicesIsRefused(self):
        nextStep = nextStepOf(CHOICES)

        with pytest.raises(ValidationError):
            nextStep(message=ANSWER, next="elsewhere")


class TestInLangGraph:
    """The node runs as a LangGraph node, and keeps nothing released."""

    def testRunsAsTheNodeOfAGraphUnderItsName(self, builtModels):
        nodeManager = NodeManager()
        reply = reserveReply(nodeManager)
        graph = StateGraph(MessagesState)
        graph.add_node(reply.name, reply)
        graph.add_edge(START, reply.name)
        graph.add_edge(reply.name, END)

        try:
            state = asyncio.run(graph.compile().ainvoke(buildState()))
        finally:
            nodeManager.releaseNode(reply)

        assert [message.content for message in state["messages"]] == [
            QUESTION,
            ANSWER,
        ]

    def testReleaseDeletesTheModelCredentialAndConfig(self, builtModels):
        nodeManager = NodeManager()
        reply = reserveReply(nodeManager)
        asyncio.run(reply(buildState()))

        nodeManager.releaseNode(reply)

        assert vars(reply) == RELEASED_NODE
