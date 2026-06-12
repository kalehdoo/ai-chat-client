from __future__ import annotations

import re
import time

from ai_chat_client.config import Settings
from ai_chat_client.llm.anthropic_client import AnthropicRunner
from ai_chat_client.prompts import BASELINE_SYSTEM_PROMPT
from ai_chat_client.results import ExperimentResult
from ai_chat_client.test_bank import TestCase


async def run_baseline_case(
    *,
    settings: Settings,
    llm: AnthropicRunner,
    run_id: str,
    test_case: TestCase,
    database_schema: str | None = None,
) -> ExperimentResult:
    schema = database_schema if database_schema is not None else settings.baseline_schema()
    prompt = (
        f"SQL Dialect: {settings.baseline_sql_dialect}\n\n"
        f"Database Schema or Reference SQL:\n{schema}\n\n"
        f"Question: {test_case.user_prompt}"
    )
    started = time.perf_counter()
    try:
        response = llm.create_message(
            system=BASELINE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - started) * 1000
        output = response.content[0].text if response.content else ""
        sql = extract_sql(output)
        success = bool(sql) and is_read_only_sql(sql)
        error_message = "" if success else "Model did not return a read-only SQL query."
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
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            generated_sql=sql,
            final_answer=output,
            execution_success=success,
            error_message=error_message,
            raw_output=output,
        )
    except Exception as exc:
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
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            execution_success=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def extract_sql(text: str) -> str:
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    sql = fenced.group(1) if fenced else text
    return sql.strip().rstrip(";") + (";" if sql.strip() else "")


def is_read_only_sql(sql: str) -> bool:
    normalized = re.sub(r"\s+", " ", sql.strip().lower())
    if not normalized:
        return False
    blocked = (
        "insert ",
        "update ",
        "delete ",
        "drop ",
        "alter ",
        "create ",
        "truncate ",
        "merge ",
        "grant ",
        "revoke ",
    )
    return normalized.startswith(("select ", "with ")) and not any(word in normalized for word in blocked)
