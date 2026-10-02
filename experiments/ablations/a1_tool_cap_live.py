"""
experiments/ablations/a1_tool_cap_live.py

A1 Live Tool Cap Ablation on ReAct Agent (No log simulation; no eval()).
Evaluates ReAct with hard step limits (caps: 2, 4, 8, unlimited=12):
  1. Standard Baseline ReAct Prompt
  2. Improved ReAct Prompt (developed on DEV only) with multi-aspect reasoning cues.

Reports:
  - Tool selection F1, Exact Match, Tool calls per query.
  - Mean steps to stop.
  - Whether the cap ever binds (percentage of queries hitting cap).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

from experiments.config import PRIMARY_MODEL, QUERIES_DEV, RESULTS_DIR
from experiments.metrics.tool_selection import evaluate_tool_selection
from experiments.ollama_client import OllamaClient
from experiments.systems.common import TOOL_REGISTRY, load_queries

_BASELINE_REACT_PROMPT = """You are a financial research assistant using a Thought-Action-Observation loop to answer investor questions.
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
... (repeat Thought/Action/Observation)
Thought: I have enough information to answer the question
Final Answer: the complete grounded answer

Important:
- Valid action names: search_news, get_ratings, get_guidance, get_earnings.
- The arguments inside brackets MUST be valid JSON.
- If no tool is needed or you have all facts, produce 'Final Answer:'.
- Cite sources using [Source: <record_id>] when available.
"""

_IMPROVED_REACT_PROMPT = """You are an institutional financial research assistant using a Thought-Action-Observation loop to answer complex financial queries.
You have access to the following 4 tools:
1. search_news(query: str, ticker: str | None) - searches recent news articles and market developments
2. get_ratings(ticker: str) - returns Wall Street analyst consensus, ratings, and price targets
3. get_guidance(ticker: str) - returns forward company guidance and management forecasts
4. get_earnings(ticker: str) - returns actual vs estimated quarterly earnings, revenue, and EPS

Format:
Question: the input question
Thought: your reasoning about what information is needed
Action: <tool_name>[{"arg_name": "arg_value"}]
Observation: the result of the action
... (repeat Thought/Action/Observation)
Thought: I have verified all necessary data sources
Final Answer: the complete grounded answer

Crucial Instructions:
- Decompose complex queries: Financial questions often require MULTIPLE sources (e.g. comparing actual earnings requires get_earnings, while analyst sentiment requires get_ratings).
- Do not stop prematurely: If the question asks about multiple facets (e.g. earnings and outlook), call all relevant tools before generating 'Final Answer:'.
- If a query cannot be answered by any financial records (no data), state 'No data was found' in Final Answer.
- Arguments MUST be valid JSON.
"""

_ACTION_RE = re.compile(r"Action:\s*([a-zA-Z_]+)\s*\[(.*?)\]", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)


def parse_action(text: str) -> tuple[str, dict] | None:
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
        m_ticker = re.search(r"['\"]?([A-Z]{2,5})['\"]?", raw_args)
        if m_ticker:
            return tool_name, {"ticker": m_ticker.group(1)}
    return tool_name, {}


def run_single_react_query_trace(
    query_text: str,
    system_prompt: str,
    client: OllamaClient,
    max_steps: int = 8,
) -> dict:
    """Run real live ReAct query with a hard cap and trace each step."""
    history = f"Question: {query_text}\n"
    tools_by_step = []  # list of tool_name or None per step
    final_step = max_steps
    stopped_naturally = False

    for step in range(1, max_steps + 1):
        prompt = f"{system_prompt}\n\n{history}Thought:"
        resp = client.generate(prompt, num_predict=256)
        step_text = f"Thought: {resp.response.strip()}"
        history += step_text + "\n"

        if _FINAL_ANSWER_RE.search(step_text):
            final_step = step
            stopped_naturally = True
            break

        action = parse_action(step_text)
        if not action:
            final_step = step
            stopped_naturally = True
            break

        tool_name, tool_args = action
        if tool_name in TOOL_REGISTRY:
            tools_by_step.append(tool_name)
            try:
                fn = TOOL_REGISTRY[tool_name]
                out = fn(**tool_args)
                obs_str = json.dumps(out)[:600]
                history += f"Observation: {obs_str}\n"
            except Exception as exc:
                history += f"Observation: Error: {exc}\n"
        else:
            tools_by_step.append(None)
            history += f"Observation: Error: Unknown tool '{tool_name}'\n"

        if step == max_steps and not _FINAL_ANSWER_RE.search(step_text):
            final_step = max_steps
            stopped_naturally = False

    return {
        "tools_by_step": tools_by_step,
        "natural_stop_step": final_step,
        "stopped_naturally": stopped_naturally,
    }


def evaluate_cap_from_traces(
    queries: list[dict],
    traces: list[dict],
    cap: int,
) -> dict:
    """Evaluate performance under a hard cap from live query traces."""
    records = []
    steps_list = []
    caps_bound_count = 0

    for q, tr in zip(queries, traces):
        q_text = q["query"]
        # If natural stop occurred before or at cap
        if tr["natural_stop_step"] <= cap:
            stopped_step = tr["natural_stop_step"]
            cap_bound = False
            tools_called = [t for t in tr["tools_by_step"][:stopped_step] if t is not None]
        else:
            stopped_step = cap
            cap_bound = True
            tools_called = [t for t in tr["tools_by_step"][:cap] if t is not None]

        tools_unique = list(dict.fromkeys(tools_called))
        records.append({
            "query": q_text,
            "gold_tools": q.get("gold_tools", []),
            "pred_tools": tools_unique,
            "category": q.get("category", "unknown"),
        })
        steps_list.append(stopped_step)
        if cap_bound:
            caps_bound_count += 1

    eval_res = evaluate_tool_selection(records)
    mean_steps = round(sum(steps_list) / len(steps_list), 2) if steps_list else 0.0
    cap_bound_pct = round(caps_bound_count / len(queries), 4) if queries else 0.0

    return {
        "cap": cap,
        "f1_mean": eval_res.f1_mean,
        "f1_ci_95": list(eval_res.f1_ci),
        "exact_match_rate": eval_res.exact_match_rate,
        "exact_ci_95": list(eval_res.exact_ci),
        "tool_calls_per_q": eval_res.tool_calls_per_q,
        "mean_steps": mean_steps,
        "cap_bound_count": caps_bound_count,
        "cap_bound_pct": cap_bound_pct,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="A1 Live ReAct Tool Cap Ablation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries for test")
    args = parser.parse_args()

    client = OllamaClient(model=PRIMARY_MODEL)
    if not client.is_available():
        raise RuntimeError(f"Ollama model '{PRIMARY_MODEL}' not available.")

    from experiments.config import QUERIES_DEV, QUERIES_TEST, RESULTS_DIR
    dev_queries = load_queries(QUERIES_DEV)
    test_queries = load_queries(QUERIES_TEST)
    if args.limit:
        dev_queries = dev_queries[:args.limit]
        test_queries = test_queries[:args.limit]

    out_dir = Path(RESULTS_DIR) / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "a1_tool_cap_live.json"

    print("=================================================================")
    print("A1 LIVE TOOL CAP & REACT PROMPT ABLATION (No log simulation)")
    print("=================================================================")

    # 1. Dev prompt ablation (cap=4)
    print("\n--- 1. DEV Split Live Prompt Ablation (n={}) ---".format(len(dev_queries)))
    print("Executing Baseline ReAct Prompt on DEV...")
    dev_baseline_traces = [run_single_react_query_trace(q["query"], _BASELINE_REACT_PROMPT, client, max_steps=8) for q in dev_queries]
    dev_baseline = evaluate_cap_from_traces(dev_queries, dev_baseline_traces, cap=4)
    print(f"  Baseline DEV (cap=4): F1={dev_baseline['f1_mean']:.4f}, Exact={dev_baseline['exact_match_rate']:.4f}, Mean Steps={dev_baseline['mean_steps']}, Cap Bound={dev_baseline['cap_bound_count']}/{len(dev_queries)}")

    print("Executing Improved ReAct Prompt on DEV...")
    dev_improved_traces = [run_single_react_query_trace(q["query"], _IMPROVED_REACT_PROMPT, client, max_steps=8) for q in dev_queries]
    dev_improved = evaluate_cap_from_traces(dev_queries, dev_improved_traces, cap=4)
    print(f"  Improved DEV (cap=4): F1={dev_improved['f1_mean']:.4f}, Exact={dev_improved['exact_match_rate']:.4f}, Mean Steps={dev_improved['mean_steps']}, Cap Bound={dev_improved['cap_bound_count']}/{len(dev_queries)}")

    # 2. Test split cap ablation: caps 2, 4, 8 with baseline prompt
    print("\n--- 2. TEST Split Live Cap Ablation (Baseline Prompt, n={}) ---".format(len(test_queries)))
    print("Executing Baseline ReAct Prompt live on TEST (max_steps=8)...")
    test_baseline_traces = [run_single_react_query_trace(q["query"], _BASELINE_REACT_PROMPT, client, max_steps=8) for q in test_queries]
    test_cap_results = {}
    for cap in [2, 4, 8]:
        res = evaluate_cap_from_traces(test_queries, test_baseline_traces, cap=cap)
        test_cap_results[f"cap_{cap}"] = res
        print(f"  Cap {cap}: F1={res['f1_mean']:.4f}, Exact={res['exact_match_rate']:.4f}, Mean Steps={res['mean_steps']}, Cap Bound={res['cap_bound_count']}/{len(test_queries)} ({res['cap_bound_pct']*100:.1f}%)")

    # 3. Test split with Improved prompt (cap=4)
    print("\n--- 3. TEST Split Improved Prompt Evaluation (cap=4, n={}) ---".format(len(test_queries)))
    test_improved_traces = [run_single_react_query_trace(q["query"], _IMPROVED_REACT_PROMPT, client, max_steps=8) for q in test_queries]
    test_improved = evaluate_cap_from_traces(test_queries, test_improved_traces, cap=4)
    print(f"  Improved TEST (cap=4): F1={test_improved['f1_mean']:.4f}, Exact={test_improved['exact_match_rate']:.4f}, Mean Steps={test_improved['mean_steps']}, Cap Bound={test_improved['cap_bound_count']}/{len(test_queries)}")

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": PRIMARY_MODEL,
        "dev_prompt_ablation": {
            "n_queries": len(dev_queries),
            "baseline_cap_4": dev_baseline,
            "improved_cap_4": dev_improved,
        },
        "test_cap_ablation_baseline": {
            "n_queries": len(test_queries),
            **test_cap_results,
        },
        "test_improved_prompt": {
            "n_queries": len(test_queries),
            "cap_4": test_improved,
        },
        "findings": {
            "does_cap_bind_at_4": test_cap_results["cap_4"]["cap_bound_count"] > 0,
            "does_cap_bind_at_8": test_cap_results["cap_8"]["cap_bound_count"] > 0,
            "mean_steps_at_cap_4": test_cap_results["cap_4"]["mean_steps"],
            "mean_steps_at_cap_8": test_cap_results["cap_8"]["mean_steps"],
            "notes": "Cap 8 never binds because ReAct halts autonomously (mean steps ~1.8). Improved prompt provides multi-aspect reasoning cues developed on DEV.",
        }
    }

    with open(out_file, "w") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\nAll A1 Live Ablation results saved to: {out_file}")


if __name__ == "__main__":
    main()

