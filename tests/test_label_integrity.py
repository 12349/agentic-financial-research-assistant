"""
tests/test_label_integrity.py

CI regression tests to guard against label leakage.
Fails if any gold label dataset matches planner.rule_based.plan() output
at a suspicious rate (> 65%), indicating that the dataset was derived from
the system under test rather than independent annotation.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from planner.rule_based import plan

# If an evaluation set matches the rule-based planner above this threshold,
# it indicates probable circular derivation/leakage.
SUSPICIOUS_AGREEMENT_THRESHOLD = 0.65


def _check_leakage(file_path: Path, max_agreement: float = SUSPICIOUS_AGREEMENT_THRESHOLD):
    if not file_path.exists():
        pytest.skip(f"File not found: {file_path}")

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        pytest.skip(f"Empty dataset: {file_path}")

    matches = 0
    for r in records:
        q = r["query"]
        gold = sorted(r.get("gold_tools", []))
        predicted = sorted([call["tool"] for call in plan(q)])
        if gold == predicted:
            matches += 1

    agreement_rate = matches / len(records)
    print(f"\n[{file_path.name}] Agreement with rule planner: {matches}/{len(records)} ({agreement_rate:.1%})")

    assert agreement_rate < max_agreement, (
        f"LABEL LEAKAGE DETECTED in {file_path}! "
        f"Gold labels match rule planner output at {agreement_rate:.1%} "
        f"(threshold: {max_agreement:.1%}). "
        f"Gold labels must be independently annotated, never derived from rule_based.plan()."
    )


def test_test_split_label_integrity():
    """Ensure test split was independently annotated."""
    _check_leakage(Path("eval/queries_test.jsonl"))


def test_dev_split_label_integrity():
    """Ensure dev split was independently annotated."""
    _check_leakage(Path("eval/queries_dev.jsonl"))


def test_train_split_label_integrity():
    """Ensure train split was independently annotated."""
    _check_leakage(Path("eval/queries_train.jsonl"))


def test_external_queries_label_integrity():
    """Ensure external_queries dataset does not leak rule planner output."""
    _check_leakage(Path("eval/external_queries/external_queries.jsonl"))
