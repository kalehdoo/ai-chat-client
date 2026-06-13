from __future__ import annotations

import os
import re
import time
from pathlib import Path

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
    # 1. Fetch the raw configuration value
    schema_source = database_schema if database_schema is not None else settings.baseline_schema()
    
    # 2. Dynamic Interception: Check if the string points to an actual file
    if schema_source and os.path.exists(schema_source):
        try:
            with open(schema_source, "r", encoding="utf-8") as f:
                schema = f.read()
            print(f"Successfully read schema content from path: {schema_source}")
        except Exception as e:
            schema = f"-- Error reading schema file: {str(e)}"
    else:
        # Fall back to treating it as a raw DDL string if it's not a path
        schema = schema_source if schema_source else ""
    
    # 3. Dynamic Semantic Layer Context Injection
    semantic_context = ""
    # Safely retrieve values from settings (falling back directly to env if class is not updated yet)
    semantic_default = getattr(settings, "semantic_default", os.getenv("SEMANTIC_DEFAULT", "off")).lower().strip()
    semantic_dir_str = getattr(settings, "semantic_dir", os.getenv("SEMANTIC_DIR", "./semantic"))
    semantic_dir = Path(semantic_dir_str)

    if semantic_default == "on":
        if semantic_dir.exists() and semantic_dir.is_dir():
            semantic_blocks = []
            # Read files: glossary, schemas, and main yml files
            files_to_read = ["glossary.yml", "schemas.yml", "main.yml"]
            for filename in files_to_read:
                filepath = semantic_dir / filename
                if filepath.exists():
                    try:
                        content = filepath.read_text(encoding="utf-8")
                        semantic_blocks.append(
                            f"--- SEMANTIC METADATA FILE: {filename} ---\n{content}"
                        )
                    except Exception as e:
                        semantic_blocks.append(
                            f"--- SEMANTIC METADATA FILE: {filename} ---\nError reading file: {str(e)}"
                        )
            if semantic_blocks:
                semantic_context = "\n\n=== ADDITIONAL SEMANTIC METADATA & BUSINESS GLOSSARIES ===\n" + "\n\n".join(semantic_blocks)
                print(f"[SEMANTIC ON] Injected {len(semantic_blocks)} semantic context files into Baseline prompt.")
        else:
            print(f"[SEMANTIC ON] Configured directory '{semantic_dir}' was not found. Skipping semantic context.")

    # 4. Construct your prompt string with the true file text
    prompt = (
        f"System prompt: {BASELINE_SYSTEM_PROMPT}\n\n"
        f"SQL Dialect: {settings.baseline_sql_dialect}\n\n"
        f"Database Schema or Reference SQL:\n{schema}\n\n"
        f"Semantic Metadata: {semantic_context}\n\n"
        f"Question: {test_case.user_prompt}"
    )
    started = time.perf_counter()
    try:

        # Create a comprehensive system payload containing your core instructions and the schema text
        combined_system_block = (
            f"System prompt: {BASELINE_SYSTEM_PROMPT}\n\n"
            f"SQL Dialect: {settings.baseline_sql_dialect}\n\n"
            f"Database Schema or Reference SQL:\n{schema}\n\n"
            f"Semantic Metadata: {semantic_context}"
        )
        response = llm.create_message(
            system=combined_system_block,  # Just pass the raw string directly
            messages=[{"role": "user", "content": f"User Request: {test_case.user_prompt}"}],
        )
        # response = llm.create_message(
        #     system=BASELINE_SYSTEM_PROMPT,
        #     messages=[{"role": "user", "content": prompt}],
        # )
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
            full_prompt=prompt,
            latency_ms_total=latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            cache_read=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
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
            full_prompt=prompt,
            latency_ms_total=latency_ms,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cache_read=0,
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
