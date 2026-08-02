"""
eval/llm_vs_rule_test.py

Runs all 8 known queries through BOTH planners (LLM + rule-based) side-by-side
and produces a structured diff: where they agree, where they diverge, and why.

This is evidence that the LLM planner adds real reasoning beyond keyword routing.

Requirements (any ONE of the following):
  - LLM_PROVIDER=groq  GROQ_API_KEY=gsk_...    (free-tier Llama 3.3 70B)
  - LLM_PROVIDER=anthropic  ANTHROPIC_API_KEY=sk-ant-...  (paid)
  - Or set just the key and let the planner auto-detect the provider.

Usage:
  LLM_PROVIDER=groq GROQ_API_KEY=gsk_... PYTHONPATH=. python3 eval/llm_vs_rule_test.py

Output:
  - Console table of agreements/divergences
  - eval/llm_vs_rule_results.json
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.rule_based import plan as rule_plan
from planner.llm_planner import plan as llm_plan_fn

SEP = "=" * 80

# All 8 queries: 5 spec + 3 stress tests
QUERIES = [
    # Spec queries
    {
        "id": "Q1",
        "label": "NVDA data center (spec)",
        "query": "How did Nvidia's data center business perform last quarter?",
        "expected_intent": "search_news",
    },
    {
        "id": "Q2",
        "label": "TSLA analyst sentiment (spec)",
        "query": "What's the current analyst sentiment on Tesla?",
        "expected_intent": "get_ratings",
    },
    {
        "id": "Q3",
        "label": "JPM guidance vs earnings (spec)",
        "query": "Did JPMorgan's actual earnings beat or miss their own guidance?",
        "expected_intent": "get_guidance+get_earnings",
    },
    {
        "id": "Q4",
        "label": "XOM OPEC outlook (spec)",
        "query": "What's the outlook for Exxon given the upcoming OPEC meeting?",
        "expected_intent": "search_news",
    },
    {
        "id": "Q5",
        "label": "Unknown ticker ZZZZ (spec)",
        "query": "What is the latest news on ZZZZ stock?",
        "expected_intent": "search_news",
    },
    # Stress queries
    {
        "id": "Q6",
        "label": "JPM earnings (stress)",
        "query": "How much did JPMorgan earn last quarter?",
        "expected_intent": "get_earnings",
    },
    {
        "id": "Q7",
        "label": "XOM guidance vs actuals (stress)",
        "query": "What is XOM reporting vs what they guided?",
        "expected_intent": "get_guidance+get_earnings",
    },
    {
        "id": "Q8",
        "label": "GS on NVDA (stress)",
        "query": "What did Goldman Sachs say about Nvidia?",
        "expected_intent": "search_news",
    },
]


def tools_from_plan(plan: list[dict]) -> list[str]:
    return [c["tool"] for c in plan]


def plans_agree(rule_tools: list[str], llm_tools: list[str]) -> bool:
    """Two plans agree if they call the same set of tools (order-independent)."""
    return set(rule_tools) == set(llm_tools)


def run_comparison():
    groq_key = os.environ.get("GROQ_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    provider = os.environ.get("LLM_PROVIDER", "").lower()

    # Auto-detect provider if LLM_PROVIDER not set
    if not provider:
        if groq_key:
            provider = "groq"
        elif anthropic_key:
            provider = "anthropic"

    if not provider or (provider == "groq" and not groq_key) or (provider == "anthropic" and not anthropic_key):
        print("ERROR: No LLM API key found.")
        print("For free Groq access (no card required):")
        print("  LLM_PROVIDER=groq GROQ_API_KEY=gsk_... PYTHONPATH=. python3 eval/llm_vs_rule_test.py")
        print("For Anthropic (paid):")
        print("  LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... PYTHONPATH=. python3 eval/llm_vs_rule_test.py")
        sys.exit(1)

    print(SEP)
    print("LLM PLANNER vs RULE-BASED PLANNER — Side-by-Side Comparison")
    print(f"LLM provider: {provider.upper()} | Model: {'llama-3.3-70b-versatile' if provider == 'groq' else 'claude-3-haiku-20240307'}")
    print(f"Running {len(QUERIES)} queries through both planners")
    print(SEP)
    print()

    results = []
    agree_count = 0

    for spec in QUERIES:
        print(f"[{spec['id']}] {spec['label']}")
        print(f"  Query: {spec['query']}")

        # Rule-based plan
        t0 = time.time()
        rule_tools = tools_from_plan(rule_plan(spec["query"]))
        rule_ms    = round((time.time() - t0) * 1000, 1)

        # LLM plan
        t0 = time.time()
        llm_result, llm_mode = llm_plan_fn(spec["query"])
        llm_ms   = round((time.time() - t0) * 1000, 1)
        llm_tools = tools_from_plan(llm_result)

        agree = plans_agree(rule_tools, llm_tools)
        if agree:
            agree_count += 1

        verdict = "✅ AGREE" if agree else "⚡ DIVERGE"
        print(f"  Rule-based: {rule_tools}  ({rule_ms}ms)")
        print(f"  LLM ({llm_mode}): {llm_tools}  ({llm_ms}ms)")
        print(f"  {verdict}")

        divergence_note = ""
        if not agree:
            # Annotate why they differ
            rule_set = set(rule_tools)
            llm_set  = set(llm_tools)
            only_rule = rule_set - llm_set
            only_llm  = llm_set - rule_set
            parts = []
            if only_rule:
                parts.append(f"Rule called {only_rule} but LLM did not")
            if only_llm:
                parts.append(f"LLM called {only_llm} but rule did not")
            divergence_note = "; ".join(parts)
            print(f"  Divergence: {divergence_note}")

        print()
        results.append({
            "id": spec["id"],
            "label": spec["label"],
            "query": spec["query"],
            "expected_intent": spec["expected_intent"],
            "rule_tools": rule_tools,
            "llm_tools": llm_tools,
            "llm_mode": llm_mode,
            "agree": agree,
            "divergence_note": divergence_note,
            "rule_latency_ms": rule_ms,
            "llm_latency_ms": llm_ms,
        })

    # Summary
    total = len(QUERIES)
    diverge_count = total - agree_count
    print(SEP)
    print("SUMMARY")
    print(SEP)
    print(f"  Queries run:     {total}")
    print(f"  ✅ Agree:        {agree_count}/{total}")
    print(f"  ⚡ Diverge:      {diverge_count}/{total}")
    print()

    diverged = [r for r in results if not r["agree"]]
    if diverged:
        print("DIVERGENCE DETAILS:")
        for r in diverged:
            print(f"  [{r['id']}] {r['query'][:70]}")
            print(f"     Rule: {r['rule_tools']}  vs  LLM: {r['llm_tools']}")
            print(f"     {r['divergence_note']}")
        print()
        print("Divergences are the most interesting cases for interviews:")
        print("  Where LLM routing differs from keyword routing, ask WHY.")
        print("  Did the LLM catch a nuance the rule table missed?")
        print("  Or did it make a mistake? The trace shows the reasoning.")

    summary = {
        "total": total,
        "agree": agree_count,
        "diverge": diverge_count,
        "agreement_rate": f"{agree_count}/{total}",
        "results": results,
    }

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "llm_vs_rule_results.json",
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {out_path}")
    return summary


if __name__ == "__main__":
    run_comparison()
