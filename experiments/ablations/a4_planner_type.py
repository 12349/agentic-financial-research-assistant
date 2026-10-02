"""
experiments/ablations/a4_planner_type.py

Ablation 4: Planner Type (Rule-based vs. LLM Planner).
Compares tool-selection precision, recall, F1, exact match, and latency
between S1 (rule-based) and S2 (qwen2.5:7b-instruct) on dev split (n=45).
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.config import RESULTS_DIR


def run_a4() -> dict:
    s1_dev_path = Path(RESULTS_DIR) / "s1_rule_planner" / "summary_dev.json"
    s2_dev_path = Path(RESULTS_DIR) / "s2_llm_planner" / "summary_dev.json"

    s1_data = json.loads(s1_dev_path.read_text()) if s1_dev_path.exists() else {}
    s2_data = json.loads(s2_dev_path.read_text()) if s2_dev_path.exists() else {}

    s1_tools = s1_data.get("tool_selection", {})
    s2_tools = s2_data.get("tool_selection", {})

    summary = {
        "ablation": "A4_planner_type",
        "split": "dev",
        "rule_planner_s1": {
            "f1": s1_tools.get("f1_mean"),
            "precision": s1_tools.get("precision_mean"),
            "recall": s1_tools.get("recall_mean"),
            "exact_match": s1_tools.get("exact_match_rate"),
            "abstention_acc": s1_tools.get("abstention_acc"),
            "tool_calls_per_q": s1_tools.get("tool_calls_per_q"),
            "mean_latency_ms": s1_data.get("mean_latency_ms"),
        },
        "llm_planner_s2": {
            "f1": s2_tools.get("f1_mean"),
            "precision": s2_tools.get("precision_mean"),
            "recall": s2_tools.get("recall_mean"),
            "exact_match": s2_tools.get("exact_match_rate"),
            "abstention_acc": s2_tools.get("abstention_acc"),
            "tool_calls_per_q": s2_tools.get("tool_calls_per_q"),
            "mean_latency_ms": s2_data.get("mean_latency_ms"),
        },
    }

    out_file = Path(RESULTS_DIR) / "ablations" / "a4_planner_type.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"A4 results saved to: {out_file}")
    return summary


if __name__ == "__main__":
    run_a4()
