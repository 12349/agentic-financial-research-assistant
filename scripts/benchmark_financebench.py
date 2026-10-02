"""
scripts/benchmark_financebench.py

External benchmark anchor: FinanceBench (Islam et al., arXiv:2311.11944).
Downloads the 150 open-source questions from GitHub, filters to questions
mentioning in-scope tickers (NVDA, TSLA, JPM, XOM), and runs S1.

Reports honest limitations regarding document QA vs structured fixture data.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from agent.synthesizer import _synthesize_fallback
from experiments.config import FINANCEBENCH_LOCAL, FINANCEBENCH_URL, FIXTURE_TICKERS, RESULTS_DIR
from experiments.systems.common import execute_tools
from planner.rule_based import plan as rule_plan


def download_financebench() -> list[dict]:
    out_path = Path(FINANCEBENCH_LOCAL)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not out_path.exists():
        print(f"Downloading FinanceBench from {FINANCEBENCH_URL}...")
        try:
            req = urllib.request.Request(FINANCEBENCH_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            out_path.write_bytes(data)
            print(f"Downloaded and saved to {out_path}")
        except Exception as e:
            print(f"Download failed: {e}")
            return []

    try:
        items = []
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    except Exception as e:
        print(f"Failed to parse FinanceBench JSONL: {e}")
        return []


def run_financebench_eval():
    fb_data = download_financebench()
    if not fb_data:
        print("FinanceBench data unavailable; skipping.")
        return

    print(f"Total FinanceBench samples: {len(fb_data)}")

    # Filter to questions referencing in-scope tickers
    in_scope = []
    for item in fb_data:
        # Check company name / ticker fields
        doc_name = item.get("doc_name", "").upper()
        question = item.get("question", "").upper()
        matched_ticker = None
        for t in FIXTURE_TICKERS:
            name_map = {"NVDA": "NVIDIA", "TSLA": "TESLA", "JPM": "JPMORGAN", "XOM": "EXXON"}
            if t in doc_name or t in question or name_map[t] in doc_name or name_map[t] in question:
                matched_ticker = t
                break
        if matched_ticker:
            in_scope.append((item, matched_ticker))

    print(f"In-scope samples referencing fixture tickers (NVDA, TSLA, JPM, XOM): {len(in_scope)}")

    results = []
    for item, ticker in in_scope:
        q_text = item.get("question", "")
        gold_answer = item.get("answer", "")

        plan = rule_plan(q_text)
        tool_outputs, latency_ms = execute_tools(plan)
        synth = _synthesize_fallback(q_text, tool_outputs)

        results.append({
            "financebench_id": item.get("financebench_id"),
            "ticker": ticker,
            "question": q_text,
            "gold_answer": gold_answer,
            "plan": plan,
            "system_answer": synth["answer"],
            "tool_outputs_count": len(tool_outputs),
        })

    summary = {
        "benchmark": "FinanceBench (Islam et al., arXiv:2311.11944)",
        "license": "Apache 2.0",
        "total_benchmark_samples": len(fb_data),
        "in_scope_ticker_samples": len(in_scope),
        "tickers_evaluated": list(FIXTURE_TICKERS),
        "stated_limitations": [
            "FinanceBench is designed for document QA over SEC 10-K/10-Q filings.",
            "Our system queries structured Q3 2024 fixture data; many FinanceBench questions reference prior fiscal years (2022, 2023).",
            "This serves as an external sanity check for ticker and domain relevance, not an apples-to-apples full financial QA benchmark.",
        ],
        "samples": results,
    }

    out_file = Path(RESULTS_DIR) / "financebench" / "financebench_evaluation.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"FinanceBench evaluation saved to {out_file}")


if __name__ == "__main__":
    run_financebench_eval()
