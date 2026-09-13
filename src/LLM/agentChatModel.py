"""One chat model per role, built from `configLLM.json` and `systemPrompt.json`.

    chatModel = buildChatModel("chatAgent", "openai", "gpt-5.5")
    reply = await chatModel.generateResponse(humanMessage, context)
"""

import json
from dataclasses import dataclass
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import loadEnvironment

LLM_DIR = Path(__file__).resolve().parent
DEFAULT_KEY = "default"


def loadJson(fileName: str) -> dict:
    """Read a JSON file that sits next to this module."""
    return json.loads((LLM_DIR / fileName).read_text(encoding="utf-8"))


CONFIG_LLM = loadJson("configLLM.json")
SYSTEM_PROMPTS = loadJson("systemPrompt.json")


def getSystemPrompt(role: str, provider: str, model: str) -> str:
    """The most specific prompt: model, then provider, then role default.

    Prompts are stored as lists of lines, joined here with newlines.
    """
    rolePrompts = SYSTEM_PROMPTS[role]
    providerPrompts = rolePrompts.get(provider, {})
    promptLines = (
        providerPrompts.get(model)
        or providerPrompts.get(DEFAULT_KEY)
        or rolePrompts[DEFAULT_KEY][DEFAULT_KEY]
    )
    return "\n".join(promptLines)


@dataclass(frozen=True, slots=True)
class ChatModel:
    """A role's LLM: its provider, model, settings and system prompt."""

    role: str
    provider: str
    model: str
    settings: dict
    systemPrompt: str
    llm: BaseChatModel

    async def generateResponse(
        self, humanMessage: str, context: str
    ) -> AIMessage:
        """Answer the human message from the context, under the prompt."""
        messages = [
            SystemMessage(self.systemPrompt),
            HumanMessage(f"Context:\n{context}\n\nMessage:\n{humanMessage}"),
        ]
        return await self.llm.ainvoke(messages)


def buildChatModel(role: str, provider: str, model: str) -> ChatModel:
    """The chat model for a role on the given provider and model.

    Raises:
        KeyError: If the provider or model is not in `configLLM.json`, or
            the role is not in `systemPrompt.json`.
    """
    # Provider packages read their API key from the environment on build.
    loadEnvironment()
    settings = CONFIG_LLM[provider][model]
    return ChatModel(
        role=role,
        provider=provider,
        model=model,
        settings=settings,
        systemPrompt=getSystemPrompt(role, provider, model),
        llm=init_chat_model(model, model_provider=provider, **settings),
    )
