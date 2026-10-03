#!/usr/bin/env python3
"""
scripts/compare_results.py

Phase 2.95 Item 7: Old-vs-new comparison table.

Reads headline numbers from pre-run summary JSONs (Phase 2.75 / Phase 2.9 results,
identified by missing git_commit or a known old commit) and from the freshly produced
Phase 2.95 summary JSONs (identified by git_commit matching the current HEAD).

Writes results/comparison_table.json and results/comparison_table.md.

Rules:
  - No re-tuning of prompts, planners, thresholds, or metrics after the run starts.
  - Numbers are copied directly from summary JSON files; no computation here.
  - If a summary JSON is missing, the column is marked NOT_RUN.
  - If a summary JSON carries the current HEAD commit, it is marked as 'new'.
  - Otherwise it is marked as 'old' with its timestamp.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "results" / "comparison_table.json"
OUT_MD   = ROOT / "results" / "comparison_table.md"


def current_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.load(open(path))
    except Exception:
        return None


def extract_headline(summary: dict | None, system_type: str) -> dict:
    """Extract headline metrics from a summary JSON."""
    if summary is None:
        return {"status": "NOT_RUN"}

    result: dict[str, Any] = {
        "status": "ok",
        "timestamp": summary.get("timestamp", "?"),
        "git_commit": summary.get("git_commit", "pre-freeze"),
        "metric_blob_hash": summary.get("metric_blob_hash", "pre-freeze"),
        "n_queries": summary.get("n_queries"),
    }

    ts = summary.get("tool_selection") or {}
    sq = summary.get("synthesis_quality") or {}

    if system_type in ("tool_selector", "react"):
        result["f1_mean"] = ts.get("f1_mean")
        result["f1_ci_95"] = ts.get("f1_ci")
        result["exact_match_rate"] = ts.get("exact_match_rate")
        result["exact_ci_95"] = ts.get("exact_ci")
        result["mean_latency_ms"] = summary.get("mean_latency_ms")

    if system_type in ("synthesis", "hybrid"):
        result["citation_valid_mean"] = sq.get("citation_valid_mean")
        result["numeric_faithfulness_mean"] = sq.get("faithfulness_mean")
        result["unsupported_rate_block"] = sq.get("unsupported_rate_mean")
        result["unsupported_rate_strict"] = sq.get("strict_unsupported_rate_mean")
        result["abstention_acc"] = sq.get("abstention_acc")
        result["mean_latency_ms"] = summary.get("mean_latency_ms")

    if system_type == "hybrid":
        result["total_revisions"] = summary.get("total_revisions")
        result["stratum_counts"] = summary.get("stratum_counts")  # new in Phase 2.95

    if system_type in ("tool_selector", "react"):
        result["f1_mean"] = ts.get("f1_mean")
        result["f1_ci_95"] = ts.get("f1_ci")
        result["exact_match_rate"] = ts.get("exact_match_rate")

    return result


# Systems to compare: (label, summary_path, system_type, split)
SYSTEMS = [
    ("S1 rule-planner",          "results/s1_rule_planner/summary_test.json",         "tool_selector", "test"),
    ("S2 Qwen planner",          "results/s2_llm_planner/summary_test.json",          "tool_selector", "test"),
    ("S4 call-all",              "results/s4_call_all/summary_test.json",             "tool_selector", "test"),
    ("S5 ReAct Qwen",            "results/s5_react/summary_test.json",                "react",         "test"),
    ("S5 ReAct Llama",           "results/s5_react_llama3.1_8b/summary_test.json",    "react",         "test"),
    ("S6 LLM synth Qwen",        "results/s6_llm_synth/summary_test.json",            "synthesis",     "test"),
    ("S6 LLM synth Llama",       "results/s6_llama3.1_8b/summary_test.json",          "synthesis",     "test"),
    ("S7 Hybrid Qwen",           "results/s7_hybrid_synth/summary_test.json",         "hybrid",        "test"),
    ("S7 Hybrid Llama",          "results/s7_hybrid_llama3.1_8b/summary_test.json",   "hybrid",        "test"),
    ("S2 Llama planner",         "results/s2_llama3.1_8b/summary_test.json",          "tool_selector", "test"),
]


def format_md_table(rows: list[dict], commit: str) -> str:
    lines = [
        "# Old vs New Comparison Table",
        "",
        f"> Phase 2.95 rerun commit: `{commit}`",
        f"> Pre-freeze results lack `git_commit`; marked as `pre-freeze`.",
        "",
        "| System | Split | Status | git_commit | F1 / Cit.Valid | EM / Unsup.Block | Latency ms | Notes |",
        "|:-------|:------|:-------|:-----------|:---------------|:-----------------|:-----------|:------|",
    ]
    for r in rows:
        h = r["headline"]
        status = "🆕 NEW" if h.get("git_commit") == commit else ("❌ NOT_RUN" if h.get("status") == "NOT_RUN" else "📦 OLD")
        gc = h.get("git_commit", "?")[:10] if h.get("git_commit") else "?"
        f1_cit = (
            f"{h.get('f1_mean', '?')}" if "f1_mean" in h
            else f"{h.get('citation_valid_mean', '?')}"
        )
        em_unsup = (
            f"{h.get('exact_match_rate', '?')}" if "exact_match_rate" in h
            else f"{h.get('unsupported_rate_block', '?')}"
        )
        lat = h.get("mean_latency_ms", "?")
        lines.append(
            f"| {r['label']} | {r['split']} | {status} | `{gc}` "
            f"| {f1_cit} | {em_unsup} | {lat} | |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    commit = current_commit()
    print(f"Current commit: {commit}")

    rows = []
    for label, rel_path, sys_type, split in SYSTEMS:
        summary_path = ROOT / rel_path
        summary = load_summary(summary_path)
        headline = extract_headline(summary, sys_type)
        is_new = headline.get("git_commit") == commit
        rows.append({
            "label": label,
            "split": split,
            "system_type": sys_type,
            "summary_path": rel_path,
            "headline": headline,
            "is_new": is_new,
        })
        sym = "🆕" if is_new else ("❌" if headline.get("status") == "NOT_RUN" else "📦")
        key_val = headline.get("f1_mean") or headline.get("citation_valid_mean") or "?"
        print(f"  {sym} {label:35s}: {key_val} | commit={headline.get('git_commit','?')[:10]}")

    # Save JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"current_commit": commit, "systems": rows}, f, indent=2, default=str)

    # Save Markdown
    md = format_md_table(rows, commit)
    with open(OUT_MD, "w") as f:
        f.write(md)

    n_new = sum(1 for r in rows if r["is_new"])
    n_old = sum(1 for r in rows if not r["is_new"] and r["headline"].get("status") != "NOT_RUN")
    n_missing = sum(1 for r in rows if r["headline"].get("status") == "NOT_RUN")
    print(f"\nComparison: {n_new} new (Phase 2.95), {n_old} old (pre-freeze), {n_missing} not run")
    print(f"Table saved to {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
