from __future__ import annotations

from typing import Any

from anthropic import Anthropic

from ai_chat_client.config import Settings


class AnthropicRunner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = Anthropic(api_key=settings.anthropic_api_key)

    def create_message(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ):
        kwargs: dict[str, Any] = {
            "model": self.settings.llm_model,
            "max_tokens": self.settings.max_tokens,
            "temperature": self.settings.temperature,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        return self.client.messages.create(**kwargs)

