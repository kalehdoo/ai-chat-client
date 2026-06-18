from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# Provider SDK Imports
from anthropic import Anthropic
from google import genai
from google.genai import types
from openai import OpenAI

from ai_chat_client.config import Settings


class BaseRunner(ABC):
    """Abstract base class that all model runners must implement."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @abstractmethod
    def create_message(
        self,
        *,
        system: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> UniversalResponse:
        """Sends a message payload to the provider and returns a normalized response."""
        pass


class UniversalUsage:
    """Standardized token usage container across providers."""

    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class UniversalContent:
    """Standardized content block. Supports both text and tool_use block types."""

    def __init__(
        self,
        type: str = "text",
        text: str = "",
        name: str = "",
        input: Any = None,
        id: str = "",
    ):
        self.type = type
        self.text = text
        self.name = name
        self.input = input if input is not None else {}
        self.id = id


class UniversalResponse:
    """Normalized response payload wrapper that exposes a consistent interface."""

    def __init__(
        self,
        text: str,
        input_tokens: int,
        output_tokens: int,
        raw_response: Any,
        content: Optional[List["UniversalContent"]] = None,
    ):
        self.content = (
            content
            if content is not None
            else ([UniversalContent(type="text", text=text)] if text else [])
        )
        self.usage = UniversalUsage(input_tokens, output_tokens)
        self.raw_response = raw_response


# =====================================================================
# 1. ANTHROPIC RUNNER
# =====================================================================
class AnthropicRunner(BaseRunner):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.client = Anthropic(api_key=settings.anthropic_api_key)

    def create_message(
        self,
        *,
        system: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> UniversalResponse:
        kwargs: Dict[str, Any] = {
            "model": self.settings.llm_model,
            "max_tokens": self.settings.max_tokens,
            "temperature": self.settings.temperature,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(**kwargs)

        content_blocks: List[UniversalContent] = []
        for part in response.content:
            part_type = getattr(part, "type", "text")
            if part_type == "tool_use":
                content_blocks.append(
                    UniversalContent(
                        type="tool_use",
                        name=getattr(part, "name", ""),
                        input=getattr(part, "input", {}),
                        id=getattr(part, "id", ""),
                    )
                )
            else:
                content_blocks.append(
                    UniversalContent(
                        type="text",
                        text=getattr(part, "text", ""),
                    )
                )

        text_content = "".join(b.text for b in content_blocks if b.type == "text")

        return UniversalResponse(
            text=text_content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw_response=response,
            content=content_blocks,
        )


# =====================================================================
# 2. OPENAI RUNNER
# =====================================================================
class OpenAIRunner(BaseRunner):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.client = OpenAI(api_key=api_key)

    def create_message(
        self,
        *,
        system: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> UniversalResponse:
        # Translate messaging history to OpenAI format (inject system instructions)
        openai_messages = [{"role": "system", "content": system}]
        for msg in messages:
            openai_messages.append({"role": msg["role"], "content": msg["content"]})

        kwargs: Dict[str, Any] = {
            "model": self.settings.llm_model,
            "max_completion_tokens": self.settings.max_tokens,
            "temperature": self.settings.temperature,
            "messages": openai_messages,
        }

        # Translate tools to OpenAI's JSON Schema function format
        if tools:
            openai_tools = []
            for tool in tools:
                openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": tool["input_schema"],
                        },
                    }
                )
            kwargs["tools"] = openai_tools

        response = self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        text_content = choice.message.content or ""

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        return UniversalResponse(
            text=text_content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_response=response,
        )


# =====================================================================
# 3. GEMINI RUNNER (Using New google-genai SDK)
# =====================================================================
class GeminiRunner(BaseRunner):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        # Initializes using GEMINI_API_KEY from environment automatically
        self.client = genai.Client()

    def create_message(
        self,
        *,
        system: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> UniversalResponse:
        # 1. Translate message history to Gemini's Content type list
        gemini_contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_contents.append(
                types.Content(
                    role=role, parts=[types.Part.from_text(text=msg["content"])]
                )
            )

        # 2. Build configuration with System Instructions
        config_kwargs: Dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": self.settings.max_tokens,
            "temperature": self.settings.temperature,
        }

        # 3. Translate tool schemas to Gemini's native Tool declarations
        if tools:
            gemini_tools = []
            for tool in tools:
                # Convert standard json-schema parameters to Gemini Schema format
                schema_props = {}
                for prop_name, prop_val in (
                    tool["input_schema"].get("properties", {}).items()
                ):
                    schema_props[prop_name] = types.Schema(
                        type=getattr(
                            types.Type,
                            prop_val.get("type", "string").upper(),
                            types.Type.STRING,
                        ),
                        description=prop_val.get("description", ""),
                    )

                gemini_tools.append(
                    types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(
                                name=tool["name"],
                                description=tool["description"],
                                parameters=types.Schema(
                                    type=types.Type.OBJECT,
                                    properties=schema_props,
                                    required=tool["input_schema"].get("required", []),
                                ),
                            )
                        ]
                    )
                )
            config_kwargs["tools"] = gemini_tools

        config = types.GenerateContentConfig(**config_kwargs)

        response = self.client.models.generate_content(
            model=self.settings.llm_model, contents=gemini_contents, config=config
        )

        input_tokens = (
            response.usage_metadata.prompt_token_count or 0
            if response.usage_metadata
            else 0
        )
        output_tokens = (
            response.usage_metadata.candidates_token_count or 0
            if response.usage_metadata
            else 0
        )

        return UniversalResponse(
            text=response.text or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_response=response,
        )


# =====================================================================
# FACTORY GENERATOR
# =====================================================================
def get_runner(settings: Settings) -> BaseRunner:
    """Returns the corresponding client runner based on settings provider."""
    provider = settings.llm_provider.lower().strip()
    if provider == "anthropic":
        return AnthropicRunner(settings)
    elif provider == "openai":
        return OpenAIRunner(settings)
    elif provider == "gemini":
        return GeminiRunner(settings)
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{settings.llm_provider}'. "
            "Supported providers are: 'anthropic', 'openai', 'gemini'."
        )
