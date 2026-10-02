"""
experiments/systems/s4_call_all.py

System 4: Call-All-Tools Baseline.
Always calls all 4 tools (max recall baseline).
Planner: fixed plan of 4 tools.
Synthesizer: deterministic formatter.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from agent.synthesizer import _synthesize_fallback
from experiments.config import KNOWN_TOOLS, QUERIES_DEV, QUERIES_TEST
from experiments.metrics.synthesis_quality import evaluate_synthesis_quality
from experiments.metrics.tool_selection import evaluate_tool_selection
from experiments.systems.common import execute_tools, load_queries, save_system_results
from planner.rule_based import _KNOWN_TICKERS, _TICKER_RE


def _extract_ticker(query: str) -> str:
    """Extract ticker from query or default to NVDA."""
    words = _TICKER_RE.findall(query)
    for w in words:
        if w.upper() in _KNOWN_TICKERS:
            return w.upper()
    return "NVDA"


def call_all_plan(query: str) -> list[dict]:
    """Emit calls for all 4 tools."""
    ticker = _extract_ticker(query)
    return [
        {"tool": "search_news", "args": {"query": query, "ticker": ticker}},
        {"tool": "get_ratings", "args": {"ticker": ticker}},
        {"tool": "get_guidance", "args": {"ticker": ticker}},
        {"tool": "get_earnings", "args": {"ticker": ticker}},
    ]


def run_s4(split_path: str, split_name: str = "test") -> dict:
    """Run S4 evaluation on a dataset split."""
    queries = load_queries(split_path)
    records = []
    latencies = []

    print(f"--- Running S4 (call-all) on {split_name} (n={len(queries)}) ---")

    for i, q in enumerate(queries):
        query_text = q["query"]
        qid = q.get("id", f"q_{i:03d}")
        gold_tools = q.get("gold_tools", [])
        category = q.get("category", "unknown")

        t0 = time.perf_counter()
        plan = call_all_plan(query_text)
        tool_outputs, exec_ms = execute_tools(plan)
        synth = _synthesize_fallback(query_text, tool_outputs)
        t_total_ms = (time.perf_counter() - t0) * 1000

        latencies.append(t_total_ms)

        pred_tools = [c["tool"] for c in plan]

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
        system_id="s4_call_all",
        split_name=split_name,
        records=records,
        tool_eval=tool_eval,
        synth_eval=synth_eval,
        extra_summary=extra_summary,
    )

    print(f"Results saved to: {summary_file}")
    print(f"Tool Selection F1: {tool_eval.f1_mean:.4f} {tool_eval.f1_ci}")
    print(f"Recall:            {tool_eval.recall_mean:.4f} {tool_eval.recall_ci}")
    print(f"Precision:         {tool_eval.precision_mean:.4f} {tool_eval.precision_ci}")
    print(f"Exact Match Rate:  {tool_eval.exact_match_rate:.4f} {tool_eval.exact_ci}")
    print(f"Tool Calls / Q:    {tool_eval.tool_calls_per_q:.2f}")

    return {
        "records": records,
        "tool_eval": tool_eval,
        "synth_eval": synth_eval,
        "summary_file": str(summary_file),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S4 call-all baseline")
    parser.add_argument("--split", choices=["test", "dev"], default="dev",
                        help="Split to evaluate: 'dev' (default) or 'test'")
    args = parser.parse_args()

    split_path = QUERIES_TEST if args.split == "test" else QUERIES_DEV
    run_s4(split_path, split_name=args.split)
