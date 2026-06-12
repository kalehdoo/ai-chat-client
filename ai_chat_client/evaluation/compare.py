from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def compare_runs(baseline_path: Path, mcp_path: Path, output_path: Path | None = None) -> pd.DataFrame:
    baseline = _read_jsonl(baseline_path).add_prefix("baseline_")
    mcp = _read_jsonl(mcp_path).add_prefix("mcp_")
    merged = baseline.merge(
        mcp,
        left_on="baseline_question_id",
        right_on="mcp_question_id",
        how="outer",
    )
    merged["latency_delta_ms"] = merged["mcp_latency_ms_total"] - merged["baseline_latency_ms_total"]
    merged["token_delta"] = merged["mcp_total_tokens"] - merged["baseline_total_tokens"]
    merged["both_success"] = merged["baseline_execution_success"] & merged["mcp_execution_success"]

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)
        write_comparison_json(output_path.with_suffix(".json"), merged)
    return merged


def write_comparison_json(path: Path, df: pd.DataFrame) -> None:
    payload = {
        "summary": summarize_comparison(df),
        "rows": _json_records(df),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def summarize_comparison(df: pd.DataFrame) -> dict:
    total = len(df)
    return {
        "total_questions": total,
        "baseline_success_rate": _mean_bool(df, "baseline_execution_success"),
        "mcp_success_rate": _mean_bool(df, "mcp_execution_success"),
        "both_success_rate": _mean_bool(df, "both_success"),
        "avg_latency_delta_ms": _mean_number(df, "latency_delta_ms"),
        "avg_token_delta": _mean_number(df, "token_delta"),
        "avg_mcp_tool_calls": _mean_number(df, "mcp_tool_calls_count"),
    }


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _mean_bool(df: pd.DataFrame, column: str) -> float:
    if column not in df or df.empty:
        return 0
    return float(df[column].fillna(False).mean())


def _mean_number(df: pd.DataFrame, column: str) -> float:
    if column not in df or df.empty:
        return 0
    return float(pd.to_numeric(df[column], errors="coerce").mean())


def _json_records(df: pd.DataFrame) -> list[dict]:
    normalized = df.where(pd.notna(df), None)
    return normalized.to_dict(orient="records")
