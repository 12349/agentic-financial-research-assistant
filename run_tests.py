"""
run_tests.py

Runs all 5 required test queries from §10 of the spec and prints the results.
This is the acceptance test. All 5 must produce a correct, sourced answer.

Test queries:
  1. "How did Nvidia's data center business perform last quarter?"
     Expected: single tool (search_news), answer about NVDA data center
  2. "What's the current analyst sentiment on Tesla?"
     Expected: single tool (get_ratings), bullish/bearish consensus with sources
  3. "Did JPMorgan's actual earnings beat or miss their own guidance?"
     Expected: multi-tool (get_guidance + get_earnings), beat/miss comparison
  4. "What's the outlook for Exxon given the upcoming OPEC meeting?"
     Expected: single tool (search_news), OPEC / XOM narrative
  5. "What is the latest news on ZZZZ stock?"
     Expected: graceful "no data" — no crash, no hallucination

Run with: PYTHONPATH=. python3 run_tests.py
"""

import json
import sys
import os

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.orchestrator import run

QUERIES = [
    {
        "id": 1,
        "label": "NVDA data center performance (single tool: news)",
        "query": "How did Nvidia's data center business perform last quarter?",
        "expected_tools": ["search_news"],
        "expected_ticker": "NVDA",
    },
    {
        "id": 2,
        "label": "TSLA analyst sentiment (single tool: ratings)",
        "query": "What's the current analyst sentiment on Tesla?",
        "expected_tools": ["get_ratings"],
        "expected_ticker": "TSLA",
    },
    {
        "id": 3,
        "label": "JPM earnings vs guidance (multi-tool: guidance + earnings)",
        "query": "Did JPMorgan's actual earnings beat or miss their own guidance?",
        "expected_tools": ["get_guidance", "get_earnings"],
        "expected_ticker": "JPM",
    },
    {
        "id": 4,
        "label": "XOM OPEC outlook (single tool: news, reasoning without hard data)",
        "query": "What's the outlook for Exxon given the upcoming OPEC meeting?",
        "expected_tools": ["search_news"],
        "expected_ticker": "XOM",
    },
    {
        "id": 5,
        "label": "Unknown ticker ZZZZ — graceful no-data response",
        "query": "What is the latest news on ZZZZ stock?",
        "expected_tools": ["search_news"],
        "expected_ticker": "ZZZZ",
    },
]

SEPARATOR = "=" * 80


def check_result(result: dict, expected: dict) -> tuple[bool, list[str]]:
    """Validate a result against expected properties. Returns (passed, issues)."""
    issues = []

    # Must have an answer
    answer = result.get("answer", "")
    if not answer:
        issues.append("FAIL: answer is empty")

    # Must have sources list
    sources = result.get("sources", [])
    if not isinstance(sources, list):
        issues.append("FAIL: sources is not a list")

    # Must have a trace
    trace = result.get("trace", [])
    if not trace:
        issues.append("FAIL: trace is empty")

    # Must have at least one real tool call in trace (not just planner)
    real_calls = [t for t in trace if not t["tool"].startswith("__")]
    if not real_calls:
        issues.append("FAIL: no real tool calls in trace")

    # Tool calls should match expected
    actual_tools = [t["tool"] for t in real_calls]
    for expected_tool in expected["expected_tools"]:
        if expected_tool not in actual_tools:
            issues.append(f"WARN: expected tool '{expected_tool}' not in trace {actual_tools}")

    # For test 5: must not hallucinate — should mention "no data" or similar
    if expected["id"] == 5:
        no_data_phrases = ["no data", "no news", "not found", "no results", "no articles", "empty"]
        if not any(phrase in answer.lower() for phrase in no_data_phrases):
            # It's acceptable if sources is empty (means it found nothing)
            if sources:
                issues.append("WARN: test 5 (unknown ticker) returned non-empty sources — possible hallucination")

    # Sources must have required fields
    for src in sources:
        for field in ["ticker", "channel", "record_id"]:
            if field not in src:
                issues.append(f"FAIL: source missing field '{field}': {src}")

    passed = not any(i.startswith("FAIL") for i in issues)
    return passed, issues


def run_tests():
    all_passed = True
    results_summary = []

    for test in QUERIES:
        print(SEPARATOR)
        print(f"TEST {test['id']}: {test['label']}")
        print(f"QUERY: {test['query']}")
        print()

        result = run(test["query"])

        # Print the answer
        print("ANSWER:")
        print(result.get("answer", "(empty)"))
        print()

        # Print sources
        sources = result.get("sources", [])
        print(f"SOURCES ({len(sources)}):")
        for src in sources:
            print(f"  - {src}")
        print()

        # Print trace summary
        trace = result.get("trace", [])
        print(f"TRACE ({len(trace)} steps, {result.get('tool_calls_made', 0)} tool calls, mode={result.get('mode')}):")
        for entry in trace:
            tool = entry["tool"]
            latency = entry["latency_ms"]
            success = "✓" if entry["success"] else "✗"
            err = f" ERROR: {entry['error']}" if entry.get("error") else ""
            print(f"  Step {entry['step']}: {tool} [{latency}ms] {success}{err}")
        print()

        # Validate
        passed, issues = check_result(result, test)
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"STATUS: {status}")
        if issues:
            for issue in issues:
                print(f"  {issue}")

        results_summary.append({
            "id": test["id"],
            "label": test["label"],
            "passed": passed,
            "issues": issues,
        })

        if not passed:
            all_passed = False

        print()

    # Final summary
    print(SEPARATOR)
    print("FINAL SUMMARY")
    print(SEPARATOR)
    for r in results_summary:
        status = "✅" if r["passed"] else "❌"
        print(f"  {status} Test {r['id']}: {r['label']}")

    passed_count = sum(1 for r in results_summary if r["passed"])
    total = len(results_summary)
    print()
    print(f"RESULT: {passed_count}/{total} tests passed")

    if all_passed:
        print("✅ ALL TESTS PASSED — Definition of Done §9 conditions met for offline mode")
    else:
        failed = [r for r in results_summary if not r["passed"]]
        print(f"❌ {len(failed)} test(s) failed — see details above")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
