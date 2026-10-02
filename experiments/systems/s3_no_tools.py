"""
experiments/systems/s3_no_tools.py

System 3: No-Tools Parametric Baseline.
Sends user query directly to the LLM (qwen2.5:7b-instruct) without tools.
Measures parametric hallucination, unsupported claim rate, and abstention behavior.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

from experiments.config import PRIMARY_MODEL, QUERIES_DEV, QUERIES_TEST
from experiments.metrics.synthesis_quality import evaluate_synthesis_quality
from experiments.metrics.tool_selection import evaluate_tool_selection
from experiments.ollama_client import OllamaResponse, synthesizer_client
from experiments.systems.common import load_queries, save_system_results

_NO_TOOLS_SYSTEM = (
    "You are a financial assistant. Answer the user's question directly from your memory. "
    "Be concise and factual. If you do not know the exact figures, state so clearly."
)


def run_s3(split_path: str, split_name: str = "test", limit: Optional[int] = None) -> dict:
    """Run S3 evaluation on a dataset split."""
    client = synthesizer_client(model=PRIMARY_MODEL)
    if not client.is_available():
        raise RuntimeError(f"Model '{PRIMARY_MODEL}' not available in Ollama. Run: ollama pull {PRIMARY_MODEL}")

    queries = load_queries(split_path)
    if limit:
        queries = queries[:limit]

    records = []
    latencies = []

    print(f"--- Running S3 (no-tools baseline with {PRIMARY_MODEL}) on {split_name} (n={len(queries)}) ---")

    for i, q in enumerate(queries):
        query_text = q["query"]
        qid = q.get("id", f"q_{i:03d}")
        gold_tools = q.get("gold_tools", [])
        category = q.get("category", "unknown")

        prompt = f"{_NO_TOOLS_SYSTEM}\n\nQuestion: {query_text}\nAnswer:"

        t0 = time.perf_counter()
        resp = client.generate(prompt, num_predict=512)
        t_total_ms = (time.perf_counter() - t0) * 1000

        latencies.append(t_total_ms)

        pred_tools = []  # No tools called by definition

        records.append({
            "id": qid,
            "query_id": qid,
            "query": query_text,
            "category": category,
            "gold_tools": gold_tools,
            "pred_tools": pred_tools,
            "plan": [],
            "llm_raw_response": resp.response,
            "llm_elapsed_ms": resp.elapsed_ms,
            "llm_prompt_tokens": resp.prompt_tokens,
            "llm_gen_tokens": resp.gen_tokens,
            "tool_outputs": [],
            "answer": resp.response,
            "sources": [],
            "latency_ms": round(t_total_ms, 2),
            "tool_exec_ms": 0.0,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(queries):
            print(f"  Processed {i + 1}/{len(queries)} queries...")

    tool_eval = evaluate_tool_selection(records)
    synth_eval = evaluate_synthesis_quality(records)

    extra_summary = {
        "model": PRIMARY_MODEL,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
    }

    summary_file = save_system_results(
        system_id="s3_no_tools",
        split_name=split_name,
        records=records,
        tool_eval=tool_eval,
        synth_eval=synth_eval,
        extra_summary=extra_summary,
    )

    print(f"Results saved to: {summary_file}")
    print(f"Tool Selection F1: {tool_eval.f1_mean:.4f}")
    print(f"Abstention Acc:    {tool_eval.abstention_acc}")
    print(f"Unsupported Rate:  {synth_eval.unsupported_rate_mean:.4f}")

    return {
        "records": records,
        "tool_eval": tool_eval,
        "synth_eval": synth_eval,
        "summary_file": str(summary_file),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S3 no-tools baseline")
    parser.add_argument("--split", choices=["test", "dev"], default="dev",
                        help="Split to evaluate: 'dev' (default) or 'test'")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on number of queries")
    args = parser.parse_args()

    split_path = QUERIES_TEST if args.split == "test" else QUERIES_DEV
    run_s3(split_path, split_name=args.split, limit=args.limit)
