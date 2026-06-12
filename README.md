# ai-chat-client

CLI harness for comparing direct text-to-SQL prompts against the same questions answered through a Warehouse MCP server.

## What It Does

- Reads questions from `test_bank.csv`
- Runs either `EXPERIMENTAL_ARM=BASELINE` or `EXPERIMENTAL_ARM=MCP`
- Calls Anthropic with deterministic settings by default
- Records latency, token usage, generated SQL, final answers, errors, and MCP tool traces
- Writes each run to a timestamped folder under `results/`
- Compares baseline and MCP result files

## Configure

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Minimum baseline configuration:

```env
EXPERIMENTAL_ARM=BASELINE
ANTHROPIC_API_KEY=...
TEST_BANK_PATH=test_bank.csv
```

Minimum MCP configuration:

```env
EXPERIMENTAL_ARM=MCP
ANTHROPIC_API_KEY=...
MCP_SERVER_URL=http://localhost:3001/mcp
MCP_AUTH_HEADER=Bearer admin-secret-key
```

For baseline mode, provide the database schema with either:

```env
BASELINE_SCHEMA_PATH=./schema.sql
BASELINE_SQL_DIALECT=duckdb
```

or:

```env
RAW_DDL_SCHEMA="CREATE TABLE ..."
BASELINE_SQL_DIALECT=duckdb
```

The schema file can contain `CREATE TABLE` statements or simple reference `SELECT` statements with table and column names. Baseline mode does not connect to DuckDB, MotherDuck, or any other database. It only gives this schema context to the LLM and asks for SQL compatible with `BASELINE_SQL_DIALECT`.

Schema precedence in baseline mode is:

1. `BASELINE_SCHEMA_PATH`
2. `RAW_DDL_SCHEMA`
3. built-in demo schema

## Test Bank

Required columns:

```csv
question_id,complexity_tier,user_prompt
```

Optional scoring columns:

```csv
expected_sql,expected_answer_notes
```

## Run

Before running MCP mode, check that the MCP server is reachable:

```bash
python main.py doctor
```

```bash
python main.py run
```

or:

```bash
python run_harness.py run
```

Each run writes:

```text
results/<timestamp>_<arm>/
  run_config.json
  results.jsonl
  results.json
  results.csv
  tool_traces.json
  summary.json
```

`results.jsonl` is the source of truth. `results.json` and `tool_traces.json` are easy to paste into JSON viewers. The CSV is for quick spreadsheet analysis and stores nested fields as CSV-escaped JSON strings.

## Compare Runs

```bash
python main.py compare \
  --baseline results/<baseline_run>/results.jsonl \
  --mcp results/<mcp_run>/results.jsonl \
  --output results/comparison.csv
```

The comparison writes both `results/comparison.csv` and `results/comparison.json`. The JSON includes a `summary` object plus row-level comparison records.

## Notes

MCP mode uses the server's Streamable HTTP transport at `/mcp`. The server should already be running, for example at:

```text
http://localhost:3001/mcp
```
