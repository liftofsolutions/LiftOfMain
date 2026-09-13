"""Btest for the LLM component: a real sample response for every
role x provider combination.

Unlike the other Btests this one calls the providers for real, so it
needs `OPENAI_API_KEY`, `DEEPSEEK_API_KEY` and `GEMINI_API_KEY` in
`.env`. Only the cheapest model of each provider is used, crossed with
every role in `systemPrompt.json`. Each response is logged whole:

    uv run pytest Btest/LLM -q --log-cli-level=INFO
"""

import asyncio
import logging

import pytest
from langchain_core.messages import AIMessage

from LLM import buildChatModel

LOGGER = logging.getLogger(__name__)

# The cheapest model of each provider in `configLLM.json`.
MODELS = [
    ("openai", "gpt-5.4-nano"),
    ("deepseek", "deepseek-v4-flash"),
    ("google_genai", "gemini-3.1-flash-lite"),
]

SUMMARIZER_SOURCE = (
    "The Zorvane Bridge in Kelport is 412 metres long. It opened "
    "in 1998 and carries both road and rail traffic. It replaced "
    "a ferry crossing that had run since 1911 and was, at the "
    "time of opening, the longest bridge in the region."
)

# One case per role: a request that role can act on, plus a check that
# the response has the shape its system prompt asks for.
ROLE_CASES = {
    "chatAgent": {
        "humanMessage": "How long is the Zorvane Bridge?",
        "context": "The Zorvane Bridge in Kelport is 412 metres long.",
        "check": lambda text: "412" in text,
    },
    "reviewer": {
        "humanMessage": "The Zorvane Bridge is 500 metres long.",
        "context": "The Zorvane Bridge in Kelport is 412 metres long.",
        "check": lambda text: "approved" in text.lower(),
    },
    "chunker": {
        "humanMessage": SUMMARIZER_SOURCE,
        "context": "Source: Kelport infrastructure register.",
        "check": lambda text: "[" in text and "]" in text,
    },
    "summarizer": {
        "humanMessage": SUMMARIZER_SOURCE,
        "context": "Source: Kelport infrastructure register.",
        "check": lambda text: len(text) < len(SUMMARIZER_SOURCE),
    },
    "ragResponseEvaluator": {
        "humanMessage": (
            "Question: How long is the Zorvane Bridge?\n"
            "Answer: The Zorvane Bridge is 412 metres long."
        ),
        "context": "The Zorvane Bridge in Kelport is 412 metres long.",
        "check": lambda text: "faithfulness" in text.lower(),
    },
}

CASES = [
    (role, provider, model)
    for role in ROLE_CASES
    for provider, model in MODELS
]


class TestGenerateResponse:
    """A real sample response for every role x provider combination."""

    @pytest.mark.parametrize(
        ("role", "provider", "model"),
        CASES,
        ids=[f"{role}-{model}" for role, provider, model in CASES],
    )
    def testGeneratesExpectedShape(self, role, provider, model):
        case = ROLE_CASES[role]
        chatModel = buildChatModel(role, provider, model)

        response = asyncio.run(
            chatModel.generateResponse(case["humanMessage"], case["context"])
        )

        LOGGER.info(
            "%s %s %s response:\n%s",
            role,
            provider,
            model,
            response.model_dump_json(indent=2),
        )
        assert isinstance(response, AIMessage)
        assert case["check"](response.text), response.text
