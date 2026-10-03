#!/usr/bin/env python3
"""
experiments/abstention_gate.py

Phase 2.9 Item 7: Relevance gate developed on DEV-only data.

Gate design: ticker/entity match + fixture coverage check.
  - Developed purely on DEV split (n=45, n_no_data=3)
  - Evaluated descriptively on TEST no_data queries (n=7)

FINDINGS (see results/abstention_gate_report.json):
  - A simple ticker/company-name entity gate achieves 2/7 (0.286) abstention
    accuracy on the 7 test no_data queries, BELOW S1 baseline of 3/7 (0.429).
  - Root cause: most no_data queries DO mention in-fixture company names
    (XOM, NVDA, JPM, etc.) but ask for data types not in fixtures
    (short interest, options flow, CDS spreads, dividends, institutional holders).
  - A meaningful gate requires fixture-coverage awareness (which data types
    are available per ticker), not just entity matching.
  - Because the gate performs below S1 baseline, it is NOT applied to S1/S7.
  - This analysis is reported descriptively. n=7 is too small for significance.

Outputs:
  - results/abstention_gate_report.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

KNOWN_TICKERS = {"AAPL", "TSLA", "NVDA", "JPM", "MSFT", "AMZN", "GOOGL", "META", "XOM", "AMD"}
COMPANY_NAMES = {
    "apple", "tesla", "nvidia", "jpmorgan", "jp morgan",
    "microsoft", "amazon", "google", "meta", "exxon", "amd",
    "alphabet", "coinbase",
}

# Fixture data types available (developed from data/*.json inspection on DEV)
FIXTURE_DATA_TYPES = {
    "AAPL": {"earnings", "guidance", "ratings"},
    "TSLA": {"earnings", "guidance", "ratings"},
    "NVDA": {"earnings", "guidance", "ratings"},
    "JPM":  {"earnings", "guidance", "ratings"},
    "MSFT": {"earnings", "guidance", "ratings"},
    "AMZN": {"earnings", "guidance", "ratings"},
    "GOOGL": {"earnings", "guidance", "ratings"},
    "META": {"earnings", "guidance", "ratings"},
    "XOM":  {"earnings", "guidance", "ratings"},
    "AMD":  {"earnings", "guidance", "ratings"},
    # news is available for AAPL, TSLA, NVDA, JPM only
}
TICKERS_WITH_NEWS = {"AAPL", "TSLA", "NVDA", "JPM"}

OUT_FILE = ROOT / "results" / "abstention_gate_report.json"


def entity_gate(query_text: str) -> bool:
    """Returns True if query mentions any known ticker or company name."""
    tickers = set(re.findall(r"\b[A-Z]{2,5}\b", query_text))
    ticker_match = bool(tickers & KNOWN_TICKERS)
    lower = query_text.lower()
    company_match = any(c in lower for c in COMPANY_NAMES)
    return ticker_match or company_match


def evaluate_on_no_data(query_file: Path) -> list[dict]:
    queries = [json.loads(l) for l in open(query_file)]
    no_data = [q for q in queries if q["category"] == "no_data"]
    results = []
    for q in no_data:
        gate = entity_gate(q["query"])
        predict_abstain = not gate
        # For no_data queries, correct answer is abstain
        correct = predict_abstain
        results.append({
            "qid": q["id"],
            "query": q["query"],
            "category": q["category"],
            "entity_gate_result": gate,
            "predicted_action": "abstain" if predict_abstain else "answer",
            "correct_abstention": correct,
        })
    return results


def main():
    dev_results = evaluate_on_no_data(ROOT / "eval" / "queries_dev.jsonl")
    test_results = evaluate_on_no_data(ROOT / "eval" / "queries_test.jsonl")

    dev_acc = sum(r["correct_abstention"] for r in dev_results) / max(len(dev_results), 1)
    test_acc = sum(r["correct_abstention"] for r in test_results) / max(len(test_results), 1)

    report = {
        "gate_description": (
            "Entity-match gate: returns True (in-scope) if query contains a known ticker "
            "symbol or company name. Developed on DEV no_data queries (n=3)."
        ),
        "gate_limitation": (
            "Most no_data queries DO mention in-fixture entities but ask for data types not "
            "in fixtures (short interest, options flow, CDS spreads, dividends, institutional "
            "holders). Entity matching alone cannot detect fixture coverage gaps."
        ),
        "dev_calibration": {
            "n": len(dev_results),
            "abstention_accuracy": round(dev_acc, 4),
            "details": dev_results,
        },
        "test_evaluation_descriptive": {
            "n": len(test_results),
            "abstention_accuracy": round(test_acc, 4),
            "baseline_s1": 0.4286,
            "baseline_s6": 0.1429,
            "baseline_s7": 0.0,
            "gate_vs_s1_delta": round(test_acc - 0.4286, 4),
            "details": test_results,
        },
        "conclusion": (
            f"Entity gate abstention accuracy = {test_acc:.3f} ({int(test_acc*len(test_results))}/{len(test_results)}), "
            "below S1 baseline 0.429. Gate NOT applied to S1/S7. "
            "A meaningful gate requires fixture-coverage awareness beyond entity matching. "
            "Reported descriptively only; n=7 is insufficient for statistical claims."
        ),
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Abstention gate report saved to {OUT_FILE}")
    print(f"  DEV  abstention accuracy: {dev_acc:.3f} (n={len(dev_results)})")
    print(f"  TEST abstention accuracy: {test_acc:.3f} (n={len(test_results)}) | S1 baseline: 0.429")
    print(f"  Conclusion: {report['conclusion']}")


if __name__ == "__main__":
    main()
