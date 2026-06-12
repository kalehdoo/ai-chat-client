from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


VALID_ARMS = {"BASELINE", "MCP"}


@dataclass(frozen=True)
class Settings:
    experimental_arm: str
    llm_provider: str
    llm_model: str
    anthropic_api_key: str
    test_bank_path: Path
    results_dir: Path
    temperature: float
    max_tokens: int
    mcp_server_url: str
    mcp_auth_header: str
    baseline_schema_path: Path | None
    raw_ddl_schema: str
    baseline_sql_dialect: str
    max_tool_rounds: int

    @classmethod
    def from_env(cls, *, require_llm_key: bool = True) -> "Settings":
        load_dotenv(Path(".env"))

        arm = os.getenv("EXPERIMENTAL_ARM", "BASELINE").strip().upper()
        provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()

        settings = cls(
            experimental_arm=arm,
            llm_provider=provider,
            llm_model=os.getenv("LLM_MODEL", "claude-sonnet-4-5").strip(),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
            test_bank_path=Path(os.getenv("TEST_BANK_PATH", "test_bank.csv")),
            results_dir=Path(os.getenv("RESULTS_DIR", "results")),
            temperature=float(os.getenv("TEMPERATURE", "0")),
            max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
            mcp_server_url=os.getenv("MCP_SERVER_URL", "http://localhost:3001/mcp").strip(),
            mcp_auth_header=os.getenv("MCP_AUTH_HEADER", "").replace('"', "").replace("'", "").strip(),
            baseline_schema_path=_optional_path(os.getenv("BASELINE_SCHEMA_PATH")),
            raw_ddl_schema=os.getenv("RAW_DDL_SCHEMA", "").strip(),
            baseline_sql_dialect=os.getenv("BASELINE_SQL_DIALECT", "duckdb").strip().lower(),
            max_tool_rounds=int(os.getenv("MAX_TOOL_ROUNDS", "8")),
        )
        settings.validate(require_llm_key=require_llm_key)
        return settings

    def validate(self, *, require_llm_key: bool = True) -> None:
        if self.experimental_arm not in VALID_ARMS:
            raise ValueError(f"EXPERIMENTAL_ARM must be one of {sorted(VALID_ARMS)}.")
        if self.llm_provider != "anthropic":
            raise ValueError("Only LLM_PROVIDER=anthropic is implemented right now.")
        if require_llm_key and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required.")
        if not self.test_bank_path.exists():
            raise ValueError(f"TEST_BANK_PATH does not exist: {self.test_bank_path}")
        if self.experimental_arm == "MCP" and not self.mcp_auth_header:
            raise ValueError("MCP_AUTH_HEADER is required for EXPERIMENTAL_ARM=MCP.")
        if self.max_tool_rounds < 1:
            raise ValueError("MAX_TOOL_ROUNDS must be at least 1.")

    def baseline_schema(self) -> str:
        if self.baseline_schema_path:
            return self.baseline_schema_path.read_text(encoding="utf-8")
        if self.raw_ddl_schema:
            return self.raw_ddl_schema
        return DEFAULT_RAW_DDL_SCHEMA


def _optional_path(value: str | None) -> Path | None:
    if not value or not value.strip():
        return None
    return Path(value.strip())


DEFAULT_RAW_DDL_SCHEMA = """
CREATE TABLE customers(
  customer_id VARCHAR,
  customer_unique_id VARCHAR,
  customer_zip_code_prefix VARCHAR,
  customer_city VARCHAR,
  customer_state VARCHAR
);

CREATE TABLE geolocation(
  geolocation_zip_code_prefix VARCHAR,
  geolocation_lat DOUBLE,
  geolocation_lng DOUBLE,
  geolocation_city VARCHAR,
  geolocation_state VARCHAR
);

CREATE TABLE order_items(
  order_id VARCHAR,
  order_item_id BIGINT,
  product_id VARCHAR,
  seller_id VARCHAR,
  shipping_limit_date TIMESTAMP,
  price DOUBLE,
  freight_value DOUBLE
);

CREATE TABLE order_payments(
  order_id VARCHAR,
  payment_sequential BIGINT,
  payment_type VARCHAR,
  payment_installments BIGINT,
  payment_value DOUBLE
);

CREATE TABLE order_reviews(
  review_id VARCHAR,
  order_id VARCHAR,
  review_score BIGINT,
  review_comment_title VARCHAR,
  review_comment_message VARCHAR,
  review_creation_date TIMESTAMP,
  review_answer_timestamp TIMESTAMP
);

CREATE TABLE orders(
  order_id VARCHAR,
  customer_id VARCHAR,
  order_status VARCHAR,
  order_purchase_timestamp TIMESTAMP,
  order_approved_at TIMESTAMP,
  order_delivered_carrier_date TIMESTAMP,
  order_delivered_customer_date TIMESTAMP,
  order_estimated_delivery_date TIMESTAMP
);

CREATE TABLE product_category_name_translation(
  product_category_name VARCHAR,
  product_category_name_english VARCHAR
);

CREATE TABLE products(
  product_id VARCHAR,
  product_category_name VARCHAR,
  product_name_lenght BIGINT,
  product_description_lenght BIGINT,
  product_photos_qty BIGINT,
  product_weight_g BIGINT,
  product_length_cm BIGINT,
  product_height_cm BIGINT,
  product_width_cm BIGINT
);

CREATE TABLE sellers(
  seller_id VARCHAR,
  seller_zip_code_prefix VARCHAR,
  seller_city VARCHAR,
  seller_state VARCHAR
);
""".strip()
