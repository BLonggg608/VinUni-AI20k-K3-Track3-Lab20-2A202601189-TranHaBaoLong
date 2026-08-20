"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from multi_agent_research_lab.core.config import get_settings

# Token pricing per 1M tokens (USD, gpt-4o-mini)
_TOKEN_PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client using OpenAI."""

    def __init__(self, model: str | None = None, temperature: float = 0.0):
        settings = get_settings()
        self.model = model or settings.openai_model
        self.temperature = temperature

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Call OpenAI Chat Completions API with retry and token logging."""
        import os

        api_key = os.environ.get("OPENAI_API_KEY") or get_settings().openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment or .env")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package not installed. Run: pip install '.[llm]'"
            ) from exc

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )

        choice = response.choices[0]
        usage = response.usage

        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None

        cost = None
        if input_tokens and output_tokens:
            prices = _TOKEN_PRICES.get(self.model, {"input": 0.0, "output": 0.0})
            cost = (input_tokens / 1_000_000) * prices["input"] + (
                output_tokens / 1_000_000
            ) * prices["output"]

        return LLMResponse(
            content=choice.message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
