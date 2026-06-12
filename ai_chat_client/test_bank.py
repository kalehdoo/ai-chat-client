from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TestCase:
    question_id: str
    complexity_tier: str
    user_prompt: str
    expected_sql: str = ""
    expected_answer_notes: str = ""


def load_test_bank(path: Path) -> list[TestCase]:
    df = pd.read_csv(path).fillna("")
    required = {"question_id", "complexity_tier", "user_prompt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Test bank is missing columns: {sorted(missing)}")

    cases = []
    for row in df.to_dict(orient="records"):
        cases.append(
            TestCase(
                question_id=str(row["question_id"]),
                complexity_tier=str(row["complexity_tier"]),
                user_prompt=str(row["user_prompt"]),
                expected_sql=str(row.get("expected_sql", "")),
                expected_answer_notes=str(row.get("expected_answer_notes", "")),
            )
        )
    return cases

