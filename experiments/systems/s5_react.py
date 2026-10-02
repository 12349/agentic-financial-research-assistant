"""
experiments/systems/s5_react.py

System 5: ReAct Agent (qwen2.5:7b-instruct).
Implements interleaved Thought -> Action -> Observation loop.
Capped at REACT_MAX_STEPS (4 steps).
Tools: search_news, get_ratings, get_guidance, get_earnings.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

from experiments.config import PRIMARY_MODEL, QUERIES_DEV, QUERIES_TEST, REACT_MAX_STEPS, REACT_STOP_TOKEN
from experiments.metrics.synthesis_quality import evaluate_synthesis_quality
from experiments.metrics.tool_selection import evaluate_tool_selection
from experiments.ollama_client import OllamaClient, OllamaResponse
from experiments.systems.common import TOOL_REGISTRY, load_queries, save_system_results

_REACT_SYSTEM_PROMPT = """You are a financial research assistant using a Thought-Action-Observation loop to answer investor questions.
You have access to the following 4 tools:
1. search_news(query: str, ticker: str | None) - searches news articles
2. get_ratings(ticker: str) - returns analyst ratings and price targets
3. get_guidance(ticker: str) - returns company guidance forecasts
4. get_earnings(ticker: str) - returns actual vs estimated quarterly earnings

Format:
Question: the input question
Thought: your reasoning about what information is needed
Action: <tool_name>[{"arg_name": "arg_value"}]
Observation: the result of the action
... (repeat Thought/Action/Observation up to 4 times)
Thought: I have enough information to answer the question
Final Answer: the complete grounded answer

Important:
- Valid action names: search_news, get_ratings, get_guidance, get_earnings.
- The arguments inside brackets MUST be valid JSON.
- If no tool is needed or you have all facts, produce 'Final Answer:'.
- Cite sources using [Source: <record_id>] when available.
"""

_ACTION_RE = re.compile(r"Action:\s*([a-zA-Z_]+)\s*\[(.*?)\]", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)


def parse_action(text: str) -> Optional[tuple[str, dict]]:
    """Parse tool name and arguments from an Action line."""
    m = _ACTION_RE.search(text)
    if not m:
        return None
    tool_name = m.group(1).strip()
    raw_args = m.group(2).strip()
    try:
        args = json.loads(raw_args)
        if isinstance(args, dict):
            return tool_name, args
    except Exception:
        # Fallback heuristic: ticker extract
        m_ticker = re.search(r"['\"]?([A-Z]{2,5})['\"]?", raw_args)
        if m_ticker:
            return tool_name, {"ticker": m_ticker.group(1)}
    return tool_name, {}


def run_react_query(query: str, client: OllamaClient) -> dict:
    """Run multi-step ReAct agent on a single query."""
    history = f"Question: {query}\n"
    tools_called = []
    tool_outputs = []
    total_llm_ms = 0
    total_gen_tokens = 0
    total_prompt_tokens = 0
    steps = 0
    final_answer = ""

    t0 = time.perf_counter()

    for step in range(REACT_MAX_STEPS):
        steps += 1
        prompt = f"{_REACT_SYSTEM_PROMPT}\n\n{history}Thought:"
        resp = client.generate(prompt, num_predict=256)
        total_llm_ms += resp.elapsed_ms
        total_gen_tokens += resp.gen_tokens
        total_prompt_tokens += resp.prompt_tokens

        step_text = f"Thought: {resp.response.strip()}"
        history += step_text + "\n"

        # Check for Final Answer
        fa_match = _FINAL_ANSWER_RE.search(step_text)
        if fa_match:
            final_answer = fa_match.group(1).strip()
            break

        # Check for Action
        action = parse_action(step_text)
        if not action:
            # No action found, break out
            final_answer = resp.response.strip()
            break

        tool_name, tool_args = action
        if tool_name not in TOOL_REGISTRY:
            obs = f"Error: Unknown tool '{tool_name}'."
            history += f"Observation: {obs}\n"
            continue

        tools_called.append({"tool": tool_name, "args": tool_args})
        try:
            fn = TOOL_REGISTRY[tool_name]
            out = fn(**tool_args)
            if isinstance(out, dict) and "tool" not in out:
                out["tool"] = tool_name
            tool_outputs.append(out)
            # Truncate observation string for prompt context
            obs_str = json.dumps(out)[:600]
            history += f"Observation: {obs_str}\n"
        except Exception as exc:
            obs = f"Error: {exc}"
            tool_outputs.append({"tool": tool_name, "error": str(exc)})
            history += f"Observation: {obs}\n"

    total_latency_ms = (time.perf_counter() - t0) * 1000

    if not final_answer:
        # Generate one final pass for answer if capped without Final Answer
        prompt = f"{_REACT_SYSTEM_PROMPT}\n\n{history}Final Answer:"
        resp = client.generate(prompt, num_predict=512)
        total_llm_ms += resp.elapsed_ms
        total_gen_tokens += resp.gen_tokens
        final_answer = resp.response.strip()

    return {
        "final_answer": final_answer,
        "tools_called": tools_called,
        "tool_outputs": tool_outputs,
        "history": history,
        "steps": steps,
        "total_latency_ms": round(total_latency_ms, 2),
        "total_llm_ms": total_llm_ms,
        "total_gen_tokens": total_gen_tokens,
        "total_prompt_tokens": total_prompt_tokens,
    }


def run_s5(split_path: str, split_name: str = "test", limit: Optional[int] = None) -> dict:
    """Run S5 evaluation on a dataset split."""
    client = OllamaClient(model=PRIMARY_MODEL, think=False, num_predict=256)
    if not client.is_available():
        raise RuntimeError(f"Model '{PRIMARY_MODEL}' not available in Ollama. Run: ollama pull {PRIMARY_MODEL}")

    queries = load_queries(split_path)
    if limit:
        queries = queries[:limit]

    records = []
    latencies = []
    steps_list = []

    print(f"--- Running S5 (ReAct agent with {PRIMARY_MODEL}) on {split_name} (n={len(queries)}) ---")

    for i, q in enumerate(queries):
        query_text = q["query"]
        qid = q.get("id", f"q_{i:03d}")
        gold_tools = q.get("gold_tools", [])
        category = q.get("category", "unknown")

        res = run_react_query(query_text, client)
        latencies.append(res["total_latency_ms"])
        steps_list.append(res["steps"])

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
            "sources": [],
            "latency_ms": res["total_latency_ms"],
            "llm_elapsed_ms": res["total_llm_ms"],
            "llm_gen_tokens": res["total_gen_tokens"],
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(queries):
            print(f"  Processed {i + 1}/{len(queries)} queries...")

    tool_eval = evaluate_tool_selection(records)
    synth_eval = evaluate_synthesis_quality(records)

    extra_summary = {
        "model": PRIMARY_MODEL,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "mean_steps": round(sum(steps_list) / len(steps_list), 2) if steps_list else 0.0,
    }

    summary_file = save_system_results(
        system_id="s5_react",
        split_name=split_name,
        records=records,
        tool_eval=tool_eval,
        synth_eval=synth_eval,
        extra_summary=extra_summary,
    )

    print(f"Results saved to: {summary_file}")
    print(f"Tool Selection F1: {tool_eval.f1_mean:.4f} {tool_eval.f1_ci}")
    print(f"Exact Match Rate:  {tool_eval.exact_match_rate:.4f} {tool_eval.exact_ci}")
    print(f"Mean Steps:        {extra_summary['mean_steps']}")
    print(f"Mean Latency:      {extra_summary['mean_latency_ms']} ms")

    return {
        "records": records,
        "tool_eval": tool_eval,
        "synth_eval": synth_eval,
        "summary_file": str(summary_file),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S5 ReAct agent evaluation")
    parser.add_argument("--split", choices=["test", "dev"], default="dev",
                        help="Split to evaluate: 'dev' (default) or 'test'")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on number of queries")
    args = parser.parse_args()

    split_path = QUERIES_TEST if args.split == "test" else QUERIES_DEV
    run_s5(split_path, split_name=args.split, limit=args.limit)
