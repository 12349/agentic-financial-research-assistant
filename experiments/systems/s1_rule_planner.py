"""
experiments/systems/s1_rule_planner.py

System 1: Rule-Based Planner + Deterministic Formatter Synthesizer.
No LLM required — fast, offline, reproducible baseline.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from agent.synthesizer import _synthesize_fallback
from experiments.config import QUERIES_DEV, QUERIES_TEST
from experiments.metrics.synthesis_quality import evaluate_synthesis_quality
from experiments.metrics.tool_selection import evaluate_tool_selection
from experiments.systems.common import execute_tools, load_queries, save_system_results
from planner.rule_based import plan as rule_plan


def run_s1(split_path: str, split_name: str = "test") -> dict:
    """Run S1 evaluation on a dataset split."""
    queries = load_queries(split_path)
    records = []
    latencies = []

    print(f"--- Running S1 (rule-planner) on {split_name} (n={len(queries)}) ---")

    for i, q in enumerate(queries):
        query_text = q["query"]
        qid = q.get("id", f"q_{i:03d}")
        gold_tools = q.get("gold_tools", [])
        category = q.get("category", "unknown")

        t0 = time.perf_counter()
        plan = rule_plan(query_text)
        tool_outputs, exec_ms = execute_tools(plan)
        synth = _synthesize_fallback(query_text, tool_outputs)
        t_total_ms = (time.perf_counter() - t0) * 1000

        latencies.append(t_total_ms)

        pred_tools = [c["tool"] for c in plan if "tool" in c]

        records.append({
            "id": qid,
            "query_id": qid,
            "query": query_text,
            "category": category,
            "gold_tools": gold_tools,
            "pred_tools": pred_tools,
            "plan": plan,
            "tool_outputs": tool_outputs,
            "answer": synth["answer"],
            "sources": synth["sources"],
            "latency_ms": round(t_total_ms, 2),
            "tool_exec_ms": exec_ms,
        })

    tool_eval = evaluate_tool_selection(records)
    synth_eval = evaluate_synthesis_quality(records)

    extra_summary = {
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "max_latency_ms": round(max(latencies), 2) if latencies else 0.0,
        "min_latency_ms": round(min(latencies), 2) if latencies else 0.0,
    }

    summary_file = save_system_results(
        system_id="s1_rule_planner",
        split_name=split_name,
        records=records,
        tool_eval=tool_eval,
        synth_eval=synth_eval,
        extra_summary=extra_summary,
    )

    print(f"Results saved to: {summary_file}")
    print(f"Tool Selection F1: {tool_eval.f1_mean:.4f} {tool_eval.f1_ci}")
    print(f"Exact Match Rate:  {tool_eval.exact_match_rate:.4f} {tool_eval.exact_ci}")
    print(f"Abstention Acc:    {tool_eval.abstention_acc}")
    print(f"Citation Validity: {synth_eval.citation_valid_mean:.4f}")
    print(f"Faithfulness:      {synth_eval.faithfulness_mean:.4f}")
    print(f"Mean Latency:      {extra_summary['mean_latency_ms']} ms")

    return {
        "records": records,
        "tool_eval": tool_eval,
        "synth_eval": synth_eval,
        "summary_file": str(summary_file),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S1 rule planner evaluation")
    parser.add_argument("--split", choices=["test", "dev"], default="dev",
                        help="Split to evaluate: 'dev' (default) or 'test'")
    args = parser.parse_args()

    split_path = QUERIES_TEST if args.split == "test" else QUERIES_DEV
    run_s1(split_path, split_name=args.split)
