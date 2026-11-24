"""The tokens each chat model call used, logged as it ends.

    from LLM import USAGE_LOG

    await graph.ainvoke(state, config={"callbacks": [USAGE_LOG]})

`UsageLog` is a LangChain callback: handed to a run, it hears every chat
model call made inside it - however deep, as LangChain passes a run's
callbacks on to what it calls - and logs one line for each as it ends:

    Model usage: {"provider": "openai", "model": "gpt-6-luna",
                  "input": 1200, "cachedInput": 1024, "output": 180,
                  "reasoning": 64, "seconds": 1.32}

`provider` and `model` are as the call was made with, from LangChain's
`ls_provider` and `ls_model_name`, so they match `configLLM.json`.
`input` counts every token sent, `cachedInput` the part of them the
provider served from its cache; `output` every token made, `reasoning`
the part of them spent reasoning; `seconds`, the call's, from its start
to its answer. A call whose provider reports no usage logs none. Nothing
of what was sent or answered is logged.
"""

import json
import logging
import time
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

LOGGER = logging.getLogger(__name__)

# What starts each line, for a reader of the log to find them by.
USAGE_PREFIX = "Model usage: "


class UsageLog(AsyncCallbackHandler):
    """Logs the tokens of each chat model call of the runs it is handed
    to."""

    def __init__(self) -> None:
        """Hearing no call yet."""
        # Each call under way, by its run id: its provider and model, and
        # when it started, by `time.perf_counter`.
        self.models: dict[UUID, tuple[str | None, str | None, float]] = {}

    async def on_chat_model_start(
        self,
        serialized: dict,
        messages: list,
        *,
        run_id: UUID,
        metadata: dict | None = None,
        **kwargs,
    ) -> None:
        """Keep the provider and model the call is made with, and when it
        started."""
        metadata = metadata or {}
        self.models[run_id] = (
            metadata.get("ls_provider"),
            metadata.get("ls_model_name"),
            time.perf_counter(),
        )

    async def on_llm_end(
        self, response: LLMResult, *, run_id: UUID, **kwargs
    ) -> None:
        """Log the tokens the call used, if its provider said, and how
        long it took."""
        call = self.models.pop(run_id, None)
        if call is None:
            return
        provider, model, started = call
        seconds = time.perf_counter() - started
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None)
                if usage:
                    LOGGER.info(
                        "%s%s",
                        USAGE_PREFIX,
                        json.dumps(
                            usageOf(provider, model, usage)
                            | {"seconds": round(seconds, 3)}
                        ),
                    )

    async def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs
    ) -> None:
        """Forget the call: it used nothing worth logging."""
        self.models.pop(run_id, None)


def usageOf(provider: str | None, model: str | None, usage: dict) -> dict:
    """The line logged of a call: who made it, and its tokens."""
    inputDetails = usage.get("input_token_details") or {}
    outputDetails = usage.get("output_token_details") or {}
    return {
        "provider": provider,
        "model": model,
        "input": usage.get("input_tokens", 0),
        "cachedInput": inputDetails.get("cache_read", 0),
        "output": usage.get("output_tokens", 0),
        "reasoning": outputDetails.get("reasoning", 0),
    }


# The one log every agent run is handed.
USAGE_LOG = UsageLog()
