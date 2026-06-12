from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from ai_chat_client.arms.baseline import run_baseline_case
from ai_chat_client.arms.mcp import _exception_summary
from ai_chat_client.arms.mcp import run_mcp_case
from ai_chat_client.config import Settings
from ai_chat_client.evaluation.compare import compare_runs, summarize_comparison
from ai_chat_client.llm.anthropic_client import AnthropicRunner
from ai_chat_client.results import (
    append_jsonl,
    create_run_dir,
    write_csv,
    write_json,
    write_run_config,
    write_summary,
    write_tool_traces,
)
from ai_chat_client.test_bank import load_test_bank


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and MCP LLM experiment arms.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("run", help="Run the configured experiment arm.")
    subparsers.add_parser("doctor", help="Check the configured MCP connection.")

    compare = subparsers.add_parser("compare", help="Compare two JSONL result files.")
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--mcp", required=True, type=Path)
    compare.add_argument("--output", type=Path)

    args = parser.parse_args()
    command = args.command or "run"
    if command == "run":
        asyncio.run(run())
    elif command == "doctor":
        asyncio.run(doctor())
    elif command == "compare":
        df = compare_runs(args.baseline, args.mcp, args.output)
        print(json.dumps(summarize_comparison(df), indent=2))
        if args.output:
            print(f"Comparison written to {args.output}")
            print(f"Comparison JSON written to {args.output.with_suffix('.json')}")


async def run() -> None:
    settings = Settings.from_env()
    run_id, run_dir = create_run_dir(settings)
    write_run_config(settings, run_id, run_dir)

    llm = AnthropicRunner(settings)
    test_cases = load_test_bank(settings.test_bank_path)
    jsonl_path = run_dir / "results.jsonl"
    results = []
    baseline_schema = settings.baseline_schema() if settings.experimental_arm == "BASELINE" else None

    print(f"Starting {settings.experimental_arm} run with {len(test_cases)} questions.")
    for test_case in test_cases:
        print(f"Running {test_case.question_id} ({test_case.complexity_tier})...")
        if settings.experimental_arm == "BASELINE":
            result = await run_baseline_case(
                settings=settings,
                llm=llm,
                run_id=run_id,
                test_case=test_case,
                database_schema=baseline_schema,
            )
        else:
            result = await run_mcp_case(
                settings=settings,
                llm=llm,
                run_id=run_id,
                test_case=test_case,
            )

        results.append(result)
        append_jsonl(jsonl_path, result)

    write_csv(run_dir / "results.csv", results)
    write_json(run_dir / "results.json", results)
    write_tool_traces(run_dir / "tool_traces.json", results)
    write_summary(run_dir / "summary.json", results)
    print(f"Run complete. Results written to {run_dir}")


async def doctor() -> None:
    settings = Settings.from_env(require_llm_key=False)
    headers = {"Authorization": settings.mcp_auth_header} if settings.mcp_auth_header else {}
    try:
        async with streamablehttp_client(settings.mcp_server_url, headers=headers) as (
            read,
            write,
            get_session_id,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                print(f"MCP connection ok: {settings.mcp_server_url}")
                print(f"Session: {get_session_id()}")
                print(f"Tools: {', '.join(tool.name for tool in tools.tools)}")
    except Exception as exc:
        error_type, error_message = _exception_summary(exc)
        print(f"MCP connection failed: {error_type}: {error_message}")


if __name__ == "__main__":
    main()
