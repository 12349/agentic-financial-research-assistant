"""
experiments/systems/common.py

Shared execution, loading, and evaluation utilities for all systems (S1-S7).

PROVENANCE GUARANTEE (Phase 2.95):
  save_system_results() automatically stamps every summary JSON with:
    - git_commit: the HEAD commit hash at run time (requires clean working tree)
    - metric_blob_hash: git hash of experiments/metrics/synthesis_quality.py
  These fields are written by code, never by hand.
  run_all.sh refuses to run on a dirty working tree (Phase 2.95 Item 1).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Optional

from experiments.config import MAX_TOOL_CALLS, RESULTS_DIR
from experiments.metrics.synthesis_quality import evaluate_synthesis_quality
from experiments.metrics.tool_selection import evaluate_tool_selection
from tools import get_earnings, get_guidance, get_ratings, search_news

TOOL_REGISTRY: dict[str, Any] = {
    "search_news": search_news,
    "get_ratings": get_ratings,
    "get_guidance": get_guidance,
    "get_earnings": get_earnings,
}

# Metric file whose blob hash is stamped into every results JSON.
_METRIC_FILE = Path(__file__).resolve().parent.parent / "metrics" / "synthesis_quality.py"


def _git_commit_hash() -> str:
    """Return the current HEAD commit hash, or 'unknown' on failure."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _metric_blob_hash() -> str:
    """Return the git blob hash of synthesis_quality.py at HEAD, or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "hash-object", str(_METRIC_FILE)], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def load_queries(split_path: str) -> list[dict]:
    """Load JSONL query records."""
    records = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def execute_tools(
    plan: list[dict],
    max_calls: int = MAX_TOOL_CALLS,
) -> tuple[list[dict], float]:
    """
    Execute tool calls in order up to max_calls.
    Returns (tool_outputs, total_execution_latency_ms).
    """
    outputs: list[dict] = []
    call_count = 0
    t0 = time.time()

    for call_spec in plan:
        if call_count >= max_calls:
            break
        tool_name = call_spec.get("tool", "")
        tool_args = call_spec.get("args", {})
        call_count += 1

        tool_fn = TOOL_REGISTRY.get(tool_name)
        if not tool_fn:
            outputs.append({
                "tool": tool_name,
                "error": f"Unknown tool: {tool_name}",
            })
            continue

        try:
            out = tool_fn(**tool_args)
            if isinstance(out, dict) and "tool" not in out:
                out["tool"] = tool_name
            outputs.append(out)
        except Exception as exc:  # noqa: BLE001
            outputs.append({
                "tool": tool_name,
                "error": str(exc),
                "results": [],
                "ratings": [],
                "guidance": [],
                "earnings": [],
            })

    total_latency_ms = round((time.time() - t0) * 1000, 2)
    return outputs, total_latency_ms


def save_system_results(
    system_id: str,
    split_name: str,
    records: list[dict],
    tool_eval: Optional[Any],
    synth_eval: Optional[Any],
    extra_summary: Optional[dict] = None,
) -> Path:
    """
    Persist per-query raw JSONs and aggregate summary JSON.

    Stamps git_commit and metric_blob_hash into the summary at write time.
    These are always set by this function — never injected by hand.
    """
    base_dir = Path(RESULTS_DIR) / system_id
    raw_dir = base_dir / "raw" / split_name
    raw_dir.mkdir(parents=True, exist_ok=True)

    for i, rec in enumerate(records):
        qid = rec.get("query_id") or rec.get("id") or f"query_{i:03d}"
        q_path = raw_dir / f"{qid}.json"
        with open(q_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)

    commit = _git_commit_hash()
    blob = _metric_blob_hash()

    summary = {
        "system_id": system_id,
        "split": split_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": commit,
        "metric_blob_hash": blob,
        "n_queries": len(records),
        "tool_selection": tool_eval.as_dict() if tool_eval else None,
        "synthesis_quality": synth_eval.as_dict() if synth_eval else None,
    }
    if extra_summary:
        summary.update(extra_summary)

    summary_file = base_dir / f"summary_{split_name}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary_file
