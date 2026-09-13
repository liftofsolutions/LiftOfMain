# LLM

`src/LLM/` gives each role its own chat model: role, provider, model,
settings and system prompt, in one object. Like `config/` it is shared logic,
not a route component, so it has no `routes.py` or `schemas.py`.

| File | Holds |
| --- | --- |
| `configLLM.json` | provider -> model -> settings, nothing else |
| `systemPrompt.json` | role -> provider -> model -> prompt lines |
| `agentChatModel.py` | `ChatModel` and `buildChatModel` |

## Usage

```python
from LLM import buildChatModel

chatModel = buildChatModel("chatAgent", "openai", "gpt-5.5")
reply = await chatModel.generateResponse(humanMessage, context)
```

`ChatModel` holds `role`, `provider`, `model`, `settings`, `systemPrompt` and
`llm`, the LangChain chat model that LangGraph nodes use directly.
`generateResponse` sends the system prompt, then the context and human
message, and returns the model's `AIMessage`.

## configLLM.json

```json
{
  "openai": {
    "gpt-5.5": {"max_tokens": 4096, "timeout": 60}
  }
}
```

Provider keys are LangChain's `model_provider` names: `openai`, `deepseek`,
`google_genai`, `xai`. Settings go unchanged into `init_chat_model`, so they
keep LangChain's parameter names (`max_tokens`).

## systemPrompt.json

```json
{
  "reviewer": {
    "default": {
      "default": ["line one", "line two"]
    }
  }
}
```

A prompt is a list of lines joined with newlines, so no JSON line passes 80
characters. The most specific prompt wins:

1. `role -> provider -> model`
2. `role -> provider -> default`
3. `role -> default -> default`

Today each role has only the shared default. To give a provider or a model
its own prompt, add it beside `default`; no code changes.

Roles: `reviewer`, `chunker`, `summarizer`, `ragResponseEvaluator`,
`chatAgent`.

## API keys

Loaded from `.env` by `config.loadEnvironment`, then read by each provider
package under its own name: `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`,
`GEMINI_API_KEY`, `XAI_API_KEY`.
