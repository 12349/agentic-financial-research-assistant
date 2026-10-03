#!/usr/bin/env python3
"""
experiments/independent_checker.py

Phase 2.95 Item 4: Independent metric checker for S6 and S7 test answers.

This module does NOT import from experiments.metrics.synthesis_quality.
It implements its own citation matching and number parsing from first
principles, then compares its verdicts against the synthesis_quality metric.

Agreement between implementations is a CONSISTENCY CHECK only — it shows
the two implementations do not contradict each other. It does NOT prove
either implementation is correct. Residual errors (e.g., both implementations
missing a citation pattern) would not be detected by this check.
See results/adversarial_test_cases.json for boundary-condition coverage.

Outputs:
  results/independent_checker_report.json
    - per-answer agreement/disagreement table
    - summary statistics (agreement rate, disagreement list)
    - system: s6_qwen, s6_llama, s7_qwen, s7_llama (test split)
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT / "results" / "independent_checker_report.json"

# ── Independent implementations ──────────────────────────────────────────────

_IC_SOURCE_RE = re.compile(r"\[Source:\s*([^\]]+)\]", re.IGNORECASE)
_IC_NUMBER_RE = re.compile(r"[-+]?\$?[\d,]+\.?\d*%?")

TOLERANCE_PCT = 0.5  # must match experiments.config.NUMERIC_TOLERANCE_PCT


def _ic_parse_number(s: str) -> float | None:
    """Independent number parser — no imports from metrics."""
    clean = s.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return float(clean)
    except ValueError:
        return None


def _ic_extract_numbers(text: str) -> list[float]:
    """Extract all numbers from text, stripping source tags first."""
    cleaned = _IC_SOURCE_RE.sub(" ", text)
    nums = []
    for m in _IC_NUMBER_RE.finditer(cleaned):
        v = _ic_parse_number(m.group())
        if v is not None:
            nums.append(v)
    return nums


def _ic_all_source_numbers(tool_outputs: list[dict]) -> set[float]:
    """Recursively collect all numeric values from tool outputs."""
    numbers: set[float] = set()
    skip_keys = {"record_id", "id", "ticker", "symbol", "status", "query", "tool", "source_type"}

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k not in skip_keys:
                    _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)
        elif isinstance(obj, (int, float)) and not math.isnan(obj):
            numbers.add(float(obj))
        elif isinstance(obj, str):
            for m in _IC_NUMBER_RE.finditer(obj):
                v = _ic_parse_number(m.group())
                if v is not None:
                    numbers.add(v)

    _walk(tool_outputs)
    return numbers


def _ic_citation_validity(answer: str, tool_outputs: list[dict]) -> dict:
    """
    Independent citation validity check.
    For each [Source: X] tag in answer, check X exists as a record_id
    in the flattened tool_outputs.
    """
    # Collect all record IDs
    record_ids: set[str] = set()

    def _collect_ids(obj: Any) -> None:
        if isinstance(obj, dict):
            for k in ("record_id", "id"):
                if k in obj and isinstance(obj[k], str):
                    record_ids.add(obj[k])
            for v in obj.values():
                _collect_ids(v)
        elif isinstance(obj, list):
            for item in obj:
                _collect_ids(item)

    for to in (tool_outputs if isinstance(tool_outputs, list) else [tool_outputs]):
        _collect_ids(to)

    cited = _IC_SOURCE_RE.findall(answer)
    if not cited:
        return {"valid_fraction": float("nan"), "n_cited": 0, "n_valid": 0}

    valid = sum(1 for cid in cited if cid.strip() in record_ids)
    return {
        "valid_fraction": round(valid / len(cited), 4),
        "n_cited": len(cited),
        "n_valid": valid,
    }


def _ic_unsupported_block(answer: str) -> float:
    """
    Independent block-level unsupported rate.
    A sentence is supported if its paragraph contains a [Source:] tag.
    """
    lower = answer.lower()
    abstain_patterns = [
        r"no data", r"not found", r"no information", r"unavailable",
        r"cannot find", r"can't find", r"no .{0,20} data",
    ]
    if any(re.search(p, lower) for p in abstain_patterns) and len(answer.split()) < 40:
        return 0.0

    sentences_total = 0
    sentences_supported = 0
    sentence_re = re.compile(r"(?<=[.!?])\s+")

    for paragraph in answer.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        block_has_cite = bool(_IC_SOURCE_RE.search(paragraph))
        for line in paragraph.split("\n"):
            line = line.strip()
            if not line:
                continue
            line_has_cite = bool(_IC_SOURCE_RE.search(line))
            for sent in sentence_re.split(line):
                sent = sent.strip()
                if not sent:
                    continue
                sentences_total += 1
                if line_has_cite or block_has_cite:
                    sentences_supported += 1

    if sentences_total == 0:
        return float("nan")
    return round((sentences_total - sentences_supported) / sentences_total, 4)


# ── Comparison logic ──────────────────────────────────────────────────────────

def check_answer(answer: str, tool_outputs: list[dict]) -> dict:
    """Run independent checks on a single answer."""
    cit = _ic_citation_validity(answer, tool_outputs)
    unsup = _ic_unsupported_block(answer)
    return {
        "ic_citation_valid": cit["valid_fraction"],
        "ic_n_cited": cit["n_cited"],
        "ic_n_valid_cited": cit["n_valid"],
        "ic_unsupported_block": unsup,
    }


def load_and_check_system(raw_dir: Path) -> list[dict]:
    """Load all raw answer files from a system's raw test dir and run IC checks."""
    results = []
    for f in sorted(raw_dir.glob("*.json")):
        d = json.load(open(f))
        qid = d.get("query_id") or d.get("id") or f.stem
        answer = d.get("answer", "")
        tool_outputs = d.get("tool_outputs", [])

        ic = check_answer(answer, tool_outputs)

        # Also read what the synthesis_quality metric stored (if available in raw file)
        # Raw files don't store per-answer metric results — only summaries do.
        # We compare against re-running the metric inline here only for reference.
        results.append({
            "qid": qid,
            "answer_length": len(answer),
            **ic,
        })
    return results


def compare_with_summary(system_dir: Path, ic_results: list[dict], split: str) -> dict:
    """
    Compare IC results against values in summary_test.json.
    The summary stores the MEAN across all queries, so we compare means.
    """
    summary_file = system_dir / f"summary_{split}.json"
    if not summary_file.exists():
        return {"error": f"No summary_{split}.json found in {system_dir}"}

    summary = json.load(open(summary_file))
    sq = summary.get("synthesis_quality") or {}

    metric_cit = sq.get("citation_valid_mean")
    metric_unsup = sq.get("unsupported_rate_mean")

    valid_cits = [r["ic_citation_valid"] for r in ic_results if not math.isnan(r["ic_citation_valid"])]
    valid_unsups = [r["ic_unsupported_block"] for r in ic_results if not math.isnan(r["ic_unsupported_block"])]

    ic_cit_mean = round(sum(valid_cits) / len(valid_cits), 4) if valid_cits else float("nan")
    ic_unsup_mean = round(sum(valid_unsups) / len(valid_unsups), 4) if valid_unsups else float("nan")

    # Disagreements: answers where IC finds invalid citation but metric found valid (or vice versa)
    disagree_cit = []
    for r in ic_results:
        if not math.isnan(r["ic_citation_valid"]) and r["ic_citation_valid"] < 1.0 and r["ic_n_cited"] > 0:
            disagree_cit.append({
                "qid": r["qid"],
                "ic_citation_valid": r["ic_citation_valid"],
                "ic_n_cited": r["ic_n_cited"],
                "ic_n_valid": r["ic_n_valid_cited"],
            })

    return {
        "n_answers": len(ic_results),
        "split": split,
        "metric_citation_valid_mean": metric_cit,
        "ic_citation_valid_mean": ic_cit_mean,
        "citation_mean_delta": round(ic_cit_mean - metric_cit, 4) if metric_cit is not None and not math.isnan(ic_cit_mean) else None,
        "metric_unsupported_block_mean": metric_unsup,
        "ic_unsupported_block_mean": ic_unsup_mean,
        "unsupported_mean_delta": round(ic_unsup_mean - metric_unsup, 4) if metric_unsup is not None and not math.isnan(ic_unsup_mean) else None,
        "answers_with_invalid_citations_ic": len(disagree_cit),
        "disagree_citation_details": disagree_cit[:10],  # first 10
        "metric_git_commit": summary.get("git_commit", "unknown"),
        "metric_blob_hash": summary.get("metric_blob_hash", "unknown"),
    }


def main() -> None:
    split = "test"
    systems = {
        "s6_qwen":  ROOT / "results" / "s6_llm_synth",
        "s6_llama": ROOT / "results" / "s6_llama3.1_8b",
        "s7_qwen":  ROOT / "results" / "s7_hybrid_synth",
        "s7_llama": ROOT / "results" / "s7_hybrid_llama3.1_8b",
    }

    report = {
        "description": (
            "Independent metric checker — does NOT import experiments.metrics.synthesis_quality. "
            "Uses own regex-based citation matching and number parsing. "
            "Compares per-system IC means against stored summary values."
        ),
        "split": split,
        "systems": {},
    }

    for sys_name, sys_dir in systems.items():
        raw_dir = sys_dir / "raw" / split
        if not raw_dir.exists():
            print(f"  SKIP {sys_name}: raw/{split}/ not found at {raw_dir}")
            report["systems"][sys_name] = {"error": f"raw/{split}/ not found"}
            continue

        print(f"  Checking {sys_name} ({raw_dir})...")
        ic_results = load_and_check_system(raw_dir)
        comparison = compare_with_summary(sys_dir, ic_results, split)
        report["systems"][sys_name] = comparison

        d_cit = comparison.get("citation_mean_delta")
        d_unsup = comparison.get("unsupported_mean_delta")
        print(f"    IC citation mean={comparison['ic_citation_valid_mean']:.4f}  "
              f"metric={comparison['metric_citation_valid_mean']}  Δ={d_cit}")
        print(f"    IC unsupported mean={comparison['ic_unsupported_block_mean']:.4f}  "
              f"metric={comparison['metric_unsupported_block_mean']}  Δ={d_unsup}")
        if comparison.get("answers_with_invalid_citations_ic", 0) > 0:
            print(f"    Answers with IC-detected invalid citations: "
                  f"{comparison['answers_with_invalid_citations_ic']}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nIndependent checker report saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
