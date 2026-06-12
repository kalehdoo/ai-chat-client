from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ai_chat_client.config import Settings


@dataclass
class ExperimentResult:
    run_id: str
    question_id: str
    complexity_tier: str
    experimental_arm: str
    provider: str
    model: str
    temperature: float
    user_prompt: str
    latency_ms_total: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    full_prompt: str = ""
    generated_sql: str = ""
    final_answer: str = ""
    execution_success: bool = False
    error_type: str = ""
    error_message: str = ""
    tool_calls_count: int = 0
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    raw_output: str = ""
    cache_read: int = 0


def create_run_dir(settings: Settings) -> tuple[str, Path]:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_id = f"{timestamp}_{settings.experimental_arm.lower()}"
    run_dir = settings.results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def write_run_config(settings: Settings, run_id: str, run_dir: Path) -> None:
    safe_config = {
        "run_id": run_id,
        "experimental_arm": settings.experimental_arm,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "test_bank_path": str(settings.test_bank_path),
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "mcp_server_url": settings.mcp_server_url,
        "max_tool_rounds": settings.max_tool_rounds,
        "baseline_sql_dialect": settings.baseline_sql_dialect,
    }
    (run_dir / "run_config.json").write_text(json.dumps(safe_config, indent=2), encoding="utf-8")


def append_jsonl(path: Path, result: ExperimentResult) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(result), default=str) + "\n")


def write_csv(path: Path, results: list[ExperimentResult]) -> None:
    flattened = []
    for result in results:
        row = asdict(result)
        row["tool_trace"] = json.dumps(row["tool_trace"])
        flattened.append(row)
    pd.DataFrame(flattened).to_csv(path, index=False)


def write_json(path: Path, results: list[ExperimentResult]) -> None:
    path.write_text(
        json.dumps([asdict(result) for result in results], indent=2, default=str),
        encoding="utf-8",
    )


def write_tool_traces(path: Path, results: list[ExperimentResult]) -> None:
    traces = {
        result.question_id: {
            "user_prompt": result.user_prompt,
            "tool_calls_count": result.tool_calls_count,
            "tool_trace": result.tool_trace,
        }
        for result in results
    }
    path.write_text(json.dumps(traces, indent=2, default=str), encoding="utf-8")


def write_summary(path: Path, results: list[ExperimentResult]) -> None:
    total = len(results)
    successes = sum(1 for result in results if result.execution_success)
    summary = {
        "total_questions": total,
        "execution_successes": successes,
        "execution_success_rate": successes / total if total else 0,
        "avg_latency_ms_total": sum(result.latency_ms_total for result in results) / total
        if total
        else 0,
        "avg_input_tokens": sum(result.input_tokens for result in results) / total if total else 0,
        "avg_output_tokens": sum(result.output_tokens for result in results) / total if total else 0,
        "avg_tool_calls": sum(result.tool_calls_count for result in results) / total if total else 0,
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
