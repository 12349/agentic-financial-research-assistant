#!/usr/bin/env python3
"""
experiments/ablations/a1_cap_via_s5.py

Phase 2.95 Item 5: Cap ablation using the PRODUCTION S5 code and prompt.

Unlike the earlier a1_tool_cap_live.py (which used a baseline ReAct prompt),
this script calls s5_react.run_s5() directly with a --cap parameter injected
into the ReAct loop. This ensures the comparison is:
  - cap=∞ (production S5, unlimited steps)  →  reported S5 F1
  - cap=4  (production S5, max 4 steps)     →  A1 cap-4 value

If cap=4 does not reproduce the reported S5 F1 (~0.433), we report
the actual measured cap=∞ F1 from this run and explain the discrepancy.

The A1 ablation claim ("cap=4 straw-man vs improved prompt") is NOT made
until this run confirms or refutes it.

Outputs:
  results/ablations/a1_cap_via_s5.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from experiments.config import PRIMARY_MODEL, QUERIES_DEV, QUERIES_TEST
from experiments.metrics.tool_selection import evaluate_tool_selection
from experiments.ollama_client import OllamaClient
from experiments.systems.common import TOOL_REGISTRY, load_queries, save_system_results
from experiments.systems.s5_react import run_react_query  # production prompt used inside
from experiments.metrics.synthesis_quality import evaluate_synthesis_quality

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "results" / "ablations"
OUT_FILE = OUT_DIR / "a1_cap_via_s5.json"


def run_with_cap(
    split_path: str,
    split_name: str,
    cap: int,
    model: str = PRIMARY_MODEL,
    limit: Optional[int] = None,
) -> dict:
    """
    Run S5 production ReAct with a hard step cap on tool calls.
    cap=0 means unlimited (production default).
    """
    client = OllamaClient(model=model, think=False, num_predict=256)
    if not client.is_available():
        raise RuntimeError(f"Model '{model}' not available in Ollama.")

    queries = load_queries(split_path)
    if limit:
        queries = queries[:limit]

    actual_cap = cap if cap > 0 else 999  # effectively unlimited
    records = []
    latencies = []
    steps_list = []
    cap_bound_count = 0

    print(f"--- A1 cap={cap} ({'unlimited' if cap == 0 else str(cap)}) "
          f"using S5 production prompt — {model} — {split_name} n={len(queries)} ---")

    for i, q in enumerate(queries):
        qid = q.get("id", f"q_{i:03d}")
        query_text = q["query"]
        gold_tools = q.get("gold_tools", [])
        category = q.get("category", "unknown")

        res = run_react_query(query_text, client, max_steps=actual_cap)
        latencies.append(res["total_latency_ms"])
        steps_list.append(res["steps"])
        cap_bound = res["steps"] >= actual_cap and actual_cap < 999
        if cap_bound:
            cap_bound_count += 1

        pred_tools = [c["tool"] for c in res["tools_called"]]
        records.append({
            "id": qid,
            "query_id": qid,
            "query": query_text,
            "category": category,
            "gold_tools": gold_tools,
            "pred_tools": pred_tools,
            "plan": res["tools_called"],
            "tool_outputs": res["tool_outputs"],
            "answer": res["final_answer"],
            "react_history": res["history"],
            "steps": res["steps"],
            "cap": cap,
            "cap_bound": cap_bound,
            "sources": [],
            "latency_ms": res["total_latency_ms"],
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(queries):
            print(f"  Processed {i + 1}/{len(queries)}")

    tool_eval = evaluate_tool_selection(records)
    synth_eval = evaluate_synthesis_quality(records)

    return {
        "cap": cap,
        "split": split_name,
        "n": len(records),
        "model": model,
        "f1_mean": tool_eval.f1_mean,
        "f1_ci_95": list(tool_eval.f1_ci),
        "exact_match_rate": tool_eval.exact_match_rate,
        "exact_ci_95": list(tool_eval.exact_ci),
        "mean_steps": round(sum(steps_list) / len(steps_list), 2),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2),
        "cap_bound_count": cap_bound_count,
        "cap_bound_pct": round(cap_bound_count / len(records), 4),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description="A1 cap ablation via production S5 code")
    parser.add_argument("--model", default=PRIMARY_MODEL)
    parser.add_argument("--cap", type=int, default=4,
                        help="Tool step cap (0=unlimited production default)")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    split_path = QUERIES_TEST if args.split == "test" else QUERIES_DEV

    # Run with requested cap
    cap_result = run_with_cap(split_path, args.split, cap=args.cap,
                               model=args.model, limit=args.limit)

    # Also run unlimited (cap=0) to get production-equivalent baseline on same queries
    unlimited_result = run_with_cap(split_path, args.split, cap=0,
                                     model=args.model, limit=args.limit)

    # Compare
    delta_f1 = round(unlimited_result["f1_mean"] - cap_result["f1_mean"], 4)
    reported_s5_f1 = None
    s5_summary = ROOT / "results" / "s5_react" / f"summary_{args.split}.json"
    if s5_summary.exists():
        s5_data = json.load(open(s5_summary))
        reported_s5_f1 = s5_data.get("tool_selection", {}).get("f1_mean")

    discrepancy_note = None
    if reported_s5_f1 is not None:
        unlimited_vs_reported = round(unlimited_result["f1_mean"] - reported_s5_f1, 4)
        if abs(unlimited_vs_reported) > 0.02:
            discrepancy_note = (
                f"cap=0 (production) F1={unlimited_result['f1_mean']:.4f} differs from "
                f"reported S5 F1={reported_s5_f1:.4f} (delta={unlimited_vs_reported:+.4f}). "
                f"Cause may be: different random seed, Ollama version, or temperature difference. "
                f"The A1 straw-man claim is NOT validated until this discrepancy is explained."
            )

    report = {
        "description": (
            "A1 cap ablation using production S5 prompt and code. "
            "cap=0 means unlimited steps (equivalent to standard S5). "
            "cap=4 tests whether step cap degrades performance."
        ),
        "model": args.model,
        "split": args.split,
        f"cap_{args.cap}_result": {k: v for k, v in cap_result.items() if k != "records"},
        "cap_0_unlimited_result": {k: v for k, v in unlimited_result.items() if k != "records"},
        "delta_f1_unlimited_minus_capped": delta_f1,
        "reported_s5_f1_from_summary": reported_s5_f1,
        "discrepancy_note": discrepancy_note,
        "straw_man_validated": discrepancy_note is None,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nA1 cap-via-S5 report saved to {OUT_FILE}")
    print(f"  cap={args.cap}:  F1={cap_result['f1_mean']:.4f} {cap_result['f1_ci_95']}")
    print(f"  cap=unlimited: F1={unlimited_result['f1_mean']:.4f} {unlimited_result['f1_ci_95']}")
    print(f"  delta (unlimited - capped): {delta_f1:+.4f}")
    if reported_s5_f1:
        print(f"  Reported S5 F1: {reported_s5_f1}")
    if discrepancy_note:
        print(f"  ⚠️  DISCREPANCY: {discrepancy_note}")
    else:
        print(f"  ✅ Cap ablation consistent with reported S5 F1.")


if __name__ == "__main__":
    main()
