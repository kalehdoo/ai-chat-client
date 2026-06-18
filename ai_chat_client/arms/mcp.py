from __future__ import annotations

import json
import time
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from ai_chat_client.config import Settings
from ai_chat_client.llm.runners import BaseRunner
from ai_chat_client.prompts import MCP_SYSTEM_PROMPT
from ai_chat_client.results import ExperimentResult
from ai_chat_client.test_bank import TestCase


async def run_mcp_case(
    *,
    settings: Settings,
    llm: BaseRunner,
    run_id: str,
    test_case: TestCase,
) -> ExperimentResult:
    headers = {"Authorization": settings.mcp_auth_header}
    started = time.perf_counter()
    tool_trace: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    final_text = ""

    try:
        async with streamablehttp_client(settings.mcp_server_url, headers=headers) as (
            read,
            write,
            _get_session_id,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                tools = [_anthropic_tool(tool) for tool in listed.tools]
                messages: list[dict[str, Any]] = [
                    {"role": "user", "content": test_case.user_prompt}
                ]

                for _round in range(settings.max_tool_rounds):
                    response = llm.create_message(
                        system=MCP_SYSTEM_PROMPT,
                        messages=messages,
                        tools=tools,
                    )
                    input_tokens += response.usage.input_tokens
                    output_tokens += response.usage.output_tokens
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [
                                _content_block_to_dict(block)
                                for block in response.content
                            ],
                        }
                    )

                    tool_uses = [
                        block for block in response.content if block.type == "tool_use"
                    ]
                    text_blocks = [
                        block.text for block in response.content if block.type == "text"
                    ]
                    if text_blocks:
                        final_text = "\n".join(text_blocks).strip()

                    if not tool_uses:
                        break

                    tool_results = []
                    for tool_use in tool_uses:
                        tool_started = time.perf_counter()
                        result = await session.call_tool(tool_use.name, tool_use.input)
                        tool_latency_ms = (time.perf_counter() - tool_started) * 1000
                        result_text = _tool_result_text(result)
                        tool_trace.append(
                            {
                                "name": tool_use.name,
                                "input": tool_use.input,
                                "latency_ms": tool_latency_ms,
                                "is_error": _is_tool_error(result),
                                "output": result_text,
                            }
                        )
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": result_text,
                                "is_error": _is_tool_error(result),
                            }
                        )

                    messages.append({"role": "user", "content": tool_results})

        latency_ms = (time.perf_counter() - started) * 1000
        return ExperimentResult(
            run_id=run_id,
            question_id=test_case.question_id,
            complexity_tier=test_case.complexity_tier,
            experimental_arm=settings.experimental_arm,
            provider=settings.llm_provider,
            model=settings.llm_model,
            temperature=settings.temperature,
            user_prompt=test_case.user_prompt,
            latency_ms_total=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            generated_sql=_last_query_sql(tool_trace),
            final_answer=final_text,
            execution_success=bool(final_text)
            and not any(t["is_error"] for t in tool_trace),
            tool_calls_count=len(tool_trace),
            tool_trace=tool_trace,
            raw_output=final_text,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        error_type, error_message = _exception_summary(exc)
        return ExperimentResult(
            run_id=run_id,
            question_id=test_case.question_id,
            complexity_tier=test_case.complexity_tier,
            experimental_arm=settings.experimental_arm,
            provider=settings.llm_provider,
            model=settings.llm_model,
            temperature=settings.temperature,
            user_prompt=test_case.user_prompt,
            latency_ms_total=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            execution_success=False,
            error_type=error_type,
            error_message=error_message,
            tool_calls_count=len(tool_trace),
            tool_trace=tool_trace,
        )


def _anthropic_tool(tool: Any) -> dict[str, Any]:
    input_schema = getattr(tool, "inputSchema", None) or getattr(
        tool, "input_schema", None
    )
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": input_schema,
    }


def _content_block_to_dict(block: Any) -> dict[str, Any]:
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    if isinstance(block, dict):
        return block
    block_type = getattr(block, "type", "text")
    if block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": getattr(block, "id", ""),
            "name": getattr(block, "name", ""),
            "input": getattr(block, "input", {}),
        }
    return {"type": block_type, "text": getattr(block, "text", str(block))}


def _tool_result_text(result: Any) -> str:
    parts = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        else:
            parts.append(str(block))
    return "\n".join(parts)


def _last_query_sql(tool_trace: list[dict[str, Any]]) -> str:
    for call in reversed(tool_trace):
        if call["name"] == "query":
            sql = call.get("input", {}).get("sql", "")
            return sql if isinstance(sql, str) else json.dumps(sql)
    return ""


def _is_tool_error(result: Any) -> bool:
    return bool(getattr(result, "isError", False) or getattr(result, "is_error", False))


def _exception_summary(exc: BaseException) -> tuple[str, str]:
    flattened = _flatten_exceptions(exc)
    if not flattened:
        return type(exc).__name__, str(exc)
    error_type = " | ".join(type(item).__name__ for item in flattened)
    error_message = " | ".join(str(item) for item in flattened if str(item))
    return error_type, error_message or str(exc)


def _flatten_exceptions(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        flattened: list[BaseException] = []
        for child in exc.exceptions:
            flattened.extend(_flatten_exceptions(child))
        return flattened
    return [exc]
