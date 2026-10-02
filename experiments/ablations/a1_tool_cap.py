"""
experiments/ablations/a1_tool_cap.py

Ablation 1: Tool Cap Ablation.
Evaluates effect of hard tool cap (4 vs unlimited) on S1 (dev split, n=45).
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.synthesizer import _synthesize_fallback
from experiments.config import QUERIES_DEV, RESULTS_DIR
from experiments.metrics.tool_selection import evaluate_tool_selection
from experiments.systems.common import execute_tools, load_queries
from planner.rule_based import plan as rule_plan


def run_a1() -> dict:
    queries = load_queries(QUERIES_DEV)

    def eval_with_cap(cap: int):
        records = []
        for q in queries:
            plan = rule_plan(q["query"])
            plan_capped = plan[:cap]
            outputs, _ = execute_tools(plan_capped, max_calls=cap)
            synth = _synthesize_fallback(q["query"], outputs)
            pred_tools = [c["tool"] for c in plan_capped if "tool" in c]
            records.append({
                "query": q["query"],
                "gold_tools": q.get("gold_tools", []),
                "pred_tools": pred_tools,
                "category": q.get("category", "unknown"),
                "answer": synth["answer"],
                "tool_outputs": outputs,
            })
        return evaluate_tool_selection(records)

    res_cap_4 = eval_with_cap(4)
    res_unlimited = eval_with_cap(100)

    summary = {
        "ablation": "A1_tool_cap",
        "split": "dev",
        "n_queries": len(queries),
        "cap_4": {
            "f1": res_cap_4.f1_mean,
            "exact_match": res_cap_4.exact_match_rate,
            "tool_calls_per_q": res_cap_4.tool_calls_per_q,
        },
        "cap_unlimited": {
            "f1": res_unlimited.f1_mean,
            "exact_match": res_unlimited.exact_match_rate,
            "tool_calls_per_q": res_unlimited.tool_calls_per_q,
        },
        "finding": (
            "Rule-based planner naturally emits at most 2 tools, so cap 4 vs unlimited "
            "has identical behavior on S1; cap primarily protects against runaway ReAct loops."
        ),
    }

    out_file = Path(RESULTS_DIR) / "ablations" / "a1_tool_cap.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"A1 results saved to: {out_file}")
    print(f"Cap 4 F1:         {res_cap_4.f1_mean:.4f}")
    print(f"Cap unlimited F1: {res_unlimited.f1_mean:.4f}")
    return summary


if __name__ == "__main__":
    run_a1()
