"""
experiments/systems/s2_llm_planner.py

System 2: LLM Planner (qwen2.5:7b-instruct) + Deterministic Formatter Synthesizer.
Isolates the planner's contribution under identical synthesis.
Inference via local Ollama REST API (think=False, temp=0, seed=42).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from agent.synthesizer import _synthesize_fallback
from experiments.config import PRIMARY_MODEL, QUERIES_DEV, QUERIES_TEST
from experiments.metrics.synthesis_quality import evaluate_synthesis_quality
from experiments.metrics.tool_selection import evaluate_tool_selection
from experiments.ollama_client import OllamaResponse, planner_client
from experiments.systems.common import execute_tools, load_queries, save_system_results
from planner.llm_planner import (
    _SYSTEM_PROMPT,
    _TOOL_DESCRIPTIONS,
    _parse_llm_response,
    _validate_plan,
)


def _build_planner_prompt(query: str) -> str:
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Available tools:\n{_TOOL_DESCRIPTIONS}\n\n"
        f"User query: {query}\n\n"
        "Output ONLY a raw JSON array of tool calls. Do not include markdown code fences or explanation."
    )


def plan_with_llm(query: str, client) -> tuple[list[dict], OllamaResponse, bool]:
    """
    Call Ollama with qwen2.5:7b-instruct to generate tool plan.
    Returns (validated_plan, ollama_response, parse_success).
    """
    prompt = _build_planner_prompt(query)
    resp = client.generate(prompt, num_predict=256)

    parsed = _parse_llm_response(resp.response)
    if parsed is not None:
        validated = _validate_plan(parsed)
        return validated, resp, True

    # If parsing failed completely, return empty list or fallback to empty
    return [], resp, False


def run_s2(split_path: str, split_name: str = "test", limit: Optional[int] = None) -> dict:
    """Run S2 evaluation on a dataset split."""
    client = planner_client(model=PRIMARY_MODEL)
    if not client.is_available():
        raise RuntimeError(f"Model '{PRIMARY_MODEL}' not available in Ollama. Run: ollama pull {PRIMARY_MODEL}")

    queries = load_queries(split_path)
    if limit:
        queries = queries[:limit]

    records = []
    latencies = []
    llm_latencies = []
    gen_tokens_list = []
    parse_successes = 0

    print(f"--- Running S2 (llm-planner with {PRIMARY_MODEL}) on {split_name} (n={len(queries)}) ---")

    for i, q in enumerate(queries):
        query_text = q["query"]
        qid = q.get("id", f"q_{i:03d}")
        gold_tools = q.get("gold_tools", [])
        category = q.get("category", "unknown")

        t0 = time.perf_counter()
        plan, resp, parsed_ok = plan_with_llm(query_text, client)
        if parsed_ok:
            parse_successes += 1

        tool_outputs, exec_ms = execute_tools(plan)
        synth = _synthesize_fallback(query_text, tool_outputs)
        t_total_ms = (time.perf_counter() - t0) * 1000

        latencies.append(t_total_ms)
        llm_latencies.append(resp.elapsed_ms)
        gen_tokens_list.append(resp.gen_tokens)

        pred_tools = [c["tool"] for c in plan if "tool" in c]

        records.append({
            "id": qid,
            "query_id": qid,
            "query": query_text,
            "category": category,
            "gold_tools": gold_tools,
            "pred_tools": pred_tools,
            "plan": plan,
            "llm_raw_response": resp.response,
            "llm_elapsed_ms": resp.elapsed_ms,
            "llm_prompt_tokens": resp.prompt_tokens,
            "llm_gen_tokens": resp.gen_tokens,
            "llm_tok_per_sec": resp.tok_per_sec,
            "parsed_ok": parsed_ok,
            "tool_outputs": tool_outputs,
            "answer": synth["answer"],
            "sources": synth["sources"],
            "latency_ms": round(t_total_ms, 2),
            "tool_exec_ms": exec_ms,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(queries):
            print(f"  Processed {i + 1}/{len(queries)} queries...")

    tool_eval = evaluate_tool_selection(records)
    synth_eval = evaluate_synthesis_quality(records)

    extra_summary = {
        "model": PRIMARY_MODEL,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "mean_llm_latency_ms": round(sum(llm_latencies) / len(llm_latencies), 2) if llm_latencies else 0.0,
        "mean_gen_tokens": round(sum(gen_tokens_list) / len(gen_tokens_list), 2) if gen_tokens_list else 0.0,
        "parse_success_rate": round(parse_successes / len(queries), 4) if queries else 0.0,
    }

    summary_file = save_system_results(
        system_id="s2_llm_planner",
        split_name=split_name,
        records=records,
        tool_eval=tool_eval,
        synth_eval=synth_eval,
        extra_summary=extra_summary,
    )

    print(f"Results saved to: {summary_file}")
    print(f"Tool Selection F1:  {tool_eval.f1_mean:.4f} {tool_eval.f1_ci}")
    print(f"Exact Match Rate:   {tool_eval.exact_match_rate:.4f} {tool_eval.exact_ci}")
    print(f"Abstention Acc:     {tool_eval.abstention_acc}")
    print(f"Parse Success Rate: {extra_summary['parse_success_rate']}")
    print(f"Mean LLM Latency:   {extra_summary['mean_llm_latency_ms']} ms")

    return {
        "records": records,
        "tool_eval": tool_eval,
        "synth_eval": synth_eval,
        "summary_file": str(summary_file),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S2 LLM planner evaluation")
    parser.add_argument("--split", choices=["test", "dev"], default="dev",
                        help="Split to evaluate: 'dev' (default) or 'test'")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on number of queries")
    args = parser.parse_args()

    split_path = QUERIES_TEST if args.split == "test" else QUERIES_DEV
    run_s2(split_path, split_name=args.split, limit=args.limit)
