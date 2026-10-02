"""
experiments/systems/s6_llm_synth.py

System 6: Rule Planner + LLM Synthesizer (qwen2.5:7b-instruct).
Uses identical tool plans to S1, but synthesizes the answer using an LLM.
Measures citation validity, numeric faithfulness, and unsupported-claim rate
to isolate the benefit of deterministic citation formatting vs. LLM generation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from agent.synthesizer import _SYNTHESIS_SYSTEM, _build_synthesis_message, _extract_sources
from experiments.config import PRIMARY_MODEL, QUERIES_DEV, QUERIES_TEST
from experiments.metrics.synthesis_quality import evaluate_synthesis_quality
from experiments.metrics.tool_selection import evaluate_tool_selection
from experiments.ollama_client import OllamaResponse, synthesizer_client
from experiments.systems.common import execute_tools, load_queries, save_system_results
from planner.rule_based import plan as rule_plan


def synthesize_with_llm(query: str, tool_outputs: list[dict], client) -> tuple[str, list[dict], OllamaResponse]:
    """Generate grounded answer via Ollama."""
    message = _build_synthesis_message(query, tool_outputs)
    prompt = f"{_SYNTHESIS_SYSTEM}\n\n{message}"
    resp = client.generate(prompt, num_predict=1024)
    sources = _extract_sources(tool_outputs)
    return resp.response, sources, resp


def run_s6(split_path: str, split_name: str = "test", limit: Optional[int] = None, model: str = PRIMARY_MODEL) -> dict:
    """Run S6 evaluation on a dataset split."""
    client = synthesizer_client(model=model)
    if not client.is_available():
        raise RuntimeError(f"Model '{model}' not available in Ollama. Run: ollama pull {model}")

    queries = load_queries(split_path)
    if limit:
        queries = queries[:limit]

    records = []
    latencies = []
    llm_latencies = []

    print(f"--- Running S6 (llm-synth with {model}) on {split_name} (n={len(queries)}) ---")

    for i, q in enumerate(queries):
        query_text = q["query"]
        qid = q.get("id", f"q_{i:03d}")
        gold_tools = q.get("gold_tools", [])
        category = q.get("category", "unknown")

        t0 = time.perf_counter()
        plan = rule_plan(query_text)
        tool_outputs, exec_ms = execute_tools(plan)
        answer_text, sources, resp = synthesize_with_llm(query_text, tool_outputs, client)
        t_total_ms = (time.perf_counter() - t0) * 1000

        latencies.append(t_total_ms)
        llm_latencies.append(resp.elapsed_ms)

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
            "answer": answer_text,
            "sources": sources,
            "llm_elapsed_ms": resp.elapsed_ms,
            "llm_prompt_tokens": resp.prompt_tokens,
            "llm_gen_tokens": resp.gen_tokens,
            "llm_tok_per_sec": resp.tok_per_sec,
            "latency_ms": round(t_total_ms, 2),
            "tool_exec_ms": exec_ms,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(queries):
            print(f"  Processed {i + 1}/{len(queries)} queries...")

    tool_eval = evaluate_tool_selection(records)
    synth_eval = evaluate_synthesis_quality(records)

    extra_summary = {
        "model": model,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "mean_llm_latency_ms": round(sum(llm_latencies) / len(llm_latencies), 2) if llm_latencies else 0.0,
    }

    system_id = "s6_llm_synth" if model == PRIMARY_MODEL else f"s6_{model.replace(':', '_')}"
    summary_file = save_system_results(
        system_id=system_id,
        split_name=split_name,
        records=records,
        tool_eval=tool_eval,
        synth_eval=synth_eval,
        extra_summary=extra_summary,
    )

    print(f"Results saved to: {summary_file}")
    print(f"Citation Validity:   {synth_eval.citation_valid_mean:.4f} {synth_eval.citation_ci}")
    print(f"Faithfulness:        {synth_eval.faithfulness_mean:.4f} {synth_eval.faithfulness_ci}")
    print(f"Unsupported Rate:    {synth_eval.unsupported_rate_mean:.4f} {synth_eval.unsupported_ci}")
    print(f"Mean LLM Latency:    {extra_summary['mean_llm_latency_ms']} ms")

    return {
        "records": records,
        "tool_eval": tool_eval,
        "synth_eval": synth_eval,
        "summary_file": str(summary_file),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S6 LLM synthesis evaluation")
    parser.add_argument("--split", choices=["test", "dev"], default="dev",
                        help="Split to evaluate: 'dev' (default) or 'test'")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on number of queries")
    parser.add_argument("--model", type=str, default=PRIMARY_MODEL,
                        help="Model name (default: PRIMARY_MODEL)")
    args = parser.parse_args()

    split_path = QUERIES_TEST if args.split == "test" else QUERIES_DEV
    run_s6(split_path, split_name=args.split, limit=args.limit, model=args.model)
