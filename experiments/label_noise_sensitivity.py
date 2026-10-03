#!/usr/bin/env python3
"""
experiments/label_noise_sensitivity.py

Phase 2.95 Item 6: Label-noise sensitivity analysis.

Runs S1 and S2 on ALL 50 human-labeled queries (train+dev).
Reports F1 and exact match under author labels vs human labels.
Computes interpretation text from results — not hardcoded.

Rules (from Phase 2.95 spec):
  - Do NOT modify any gold labels or tune on these queries.
  - Results are reported on ALL 50 queries regardless of split membership.
  - S1 and S2 must be run fresh; we cannot reuse existing raw files
    because the 50 queries span train split where raw results don't exist.

Outputs:
  results/label_noise_sensitivity_v2.json
"""
from __future__ import annotations

import json
import csv
import math
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

# Import S1 runner (offline, no LLM)
from experiments.systems.s1_rule_planner import run_s1_query
from experiments.metrics.tool_selection import bootstrap_ci
from planner.rule_based import plan as rule_plan


# S2 import for LLM planner — only run if model is available
try:
    from experiments.systems.s2_llm_planner import run_s2
    from experiments.ollama_client import OllamaClient
    S2_AVAILABLE = True
except ImportError:
    S2_AVAILABLE = False


OUT_FILE = ROOT / "results" / "label_noise_sensitivity_v2.json"
HUMAN_CSV = ROOT / "eval" / "human_annotation_sample_filled.csv"
QUERIES_JSONL = ROOT / "eval" / "queries.jsonl"


def load_human_labels() -> dict[str, dict]:
    """Load the 50 human-labeled queries from filled CSV."""
    labels = {}
    with open(HUMAN_CSV) as f:
        for row in csv.DictReader(f):
            qid = row["query_id"].strip()
            tools_str = row["human_tools"].strip()
            tools = sorted(set(t.strip() for t in tools_str.split(",") if t.strip())) if tools_str else []
            labels[qid] = {
                "gold_tools": tools,
                "category": row["human_category"].strip(),
            }
    return labels


def load_author_labels() -> dict[str, dict]:
    """Load author-assigned labels from queries.jsonl."""
    labels = {}
    with open(QUERIES_JSONL) as f:
        for line in f:
            q = json.loads(line)
            labels[q["id"]] = {
                "gold_tools": q.get("gold_tools", []),
                "category": q.get("category", ""),
            }
    return labels


def load_queries_for_ids(qids: list[str]) -> list[dict]:
    """Load query text for the given IDs from queries.jsonl."""
    all_queries = {}
    with open(QUERIES_JSONL) as f:
        for line in f:
            q = json.loads(line)
            all_queries[q["id"]] = q
    return [all_queries[qid] for qid in qids if qid in all_queries]


def compute_f1_em(pred_gold_pairs: list[tuple[list, list]]) -> dict:
    """Compute mean F1, EM, and bootstrap CIs from (pred_tools, gold_tools) pairs."""
    f1s, ems = [], []
    for pred, gold in pred_gold_pairs:
        pred_set = set(pred)
        gold_set = set(gold)
        if not gold_set and not pred_set:
            f1s.append(1.0); ems.append(1.0); continue
        if not gold_set or not pred_set:
            f1s.append(0.0); ems.append(1.0 if pred_set == gold_set else 0.0); continue
        prec = len(pred_set & gold_set) / len(pred_set)
        rec = len(pred_set & gold_set) / len(gold_set)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        ems.append(1.0 if pred_set == gold_set else 0.0)

    n = len(f1s)
    f1_mean = round(sum(f1s) / n, 4) if n else float("nan")
    em_mean = round(sum(ems) / n, 4) if n else float("nan")
    f1_ci = list(bootstrap_ci(f1s)) if n else [float("nan"), float("nan")]
    em_ci = list(bootstrap_ci(ems)) if n else [float("nan"), float("nan")]
    return {"n": n, "f1_mean": f1_mean, "f1_ci_95": f1_ci, "em_mean": em_mean, "em_ci_95": em_ci}


def run_s1_on_queries(queries: list[dict]) -> list[dict]:
    """Run S1 (rule planner) on a list of query dicts. Returns records."""
    records = []
    for q in queries:
        qid = q["id"]
        query_text = q["query"]
        try:
            plan = rule_plan(query_text)
            pred_tools = [c["tool"] for c in plan if "tool" in c]
        except Exception as e:
            pred_tools = []
        records.append({"qid": qid, "pred_tools": pred_tools})
    return records


def compute_sensitivity(
    records: list[dict],
    author_labels: dict,
    human_labels: dict,
    system_name: str,
) -> dict:
    """Compute label-noise sensitivity for one system's records."""
    author_pairs = []
    human_pairs = []
    disagree_qids = []

    for r in records:
        qid = r["qid"]
        pred = r.get("pred_tools", [])
        a_gold = author_labels.get(qid, {}).get("gold_tools", [])
        h_gold = human_labels.get(qid, {}).get("gold_tools", [])
        author_pairs.append((pred, a_gold))
        human_pairs.append((pred, h_gold))
        if set(a_gold) != set(h_gold):
            disagree_qids.append({
                "qid": qid,
                "author_tools": sorted(a_gold),
                "human_tools": sorted(h_gold),
            })

    author_metrics = compute_f1_em(author_pairs)
    human_metrics = compute_f1_em(human_pairs)
    delta_f1 = round(human_metrics["f1_mean"] - author_metrics["f1_mean"], 4)
    delta_em = round(human_metrics["em_mean"] - author_metrics["em_mean"], 4)

    # Interpretation computed from results, not hardcoded
    n = author_metrics["n"]
    n_disagree = len(disagree_qids)
    f1_ci_width = round(author_metrics["f1_ci_95"][1] - author_metrics["f1_ci_95"][0], 3)
    significance = "not measurable" if n < 30 else ("not significant" if abs(delta_f1) < 0.05 else "potentially significant")
    interpretation = (
        f"Label swap on {n} queries changes {system_name} F1 by {delta_f1:+.3f} "
        f"(author={author_metrics['f1_mean']:.3f}, human={human_metrics['f1_mean']:.3f}). "
        f"CI width={f1_ci_width:.3f}. "
        f"{n_disagree}/{n} queries have tool-set disagreement between author and human labels. "
        f"Statistical interpretation: {significance} (n={'<30, too small' if n < 30 else n})."
    )

    return {
        "system": system_name,
        "n_queries": n,
        "n_label_disagreements": n_disagree,
        "label_disagreements": disagree_qids,
        "under_author_labels": author_metrics,
        "under_human_labels": human_metrics,
        "delta_f1": delta_f1,
        "delta_em": delta_em,
        "interpretation": interpretation,
    }


def main(model: str = "qwen2.5:7b-instruct") -> None:
    print("=== Label-Noise Sensitivity Analysis (Phase 2.95 Item 6) ===")
    human_labels = load_human_labels()
    author_labels = load_author_labels()
    qids_50 = sorted(human_labels.keys())
    queries_50 = load_queries_for_ids(qids_50)
    print(f"Loaded {len(queries_50)}/50 human-labeled queries from queries.jsonl")

    report = {
        "description": (
            "Label-noise sensitivity: S1 and S2 run on all 50 human-labeled queries. "
            "F1/EM reported under author labels and human labels. "
            "Gold labels NOT modified. NOT a tuning set."
        ),
        "n_total": len(queries_50),
        "label_disagreements_total": sum(
            1 for qid in qids_50
            if set(author_labels.get(qid, {}).get("gold_tools", [])) !=
               set(human_labels.get(qid, {}).get("gold_tools", []))
        ),
    }

    # S1: offline, always run
    print(f"\nRunning S1 on {len(queries_50)} queries...")
    s1_records = run_s1_on_queries(queries_50)
    report["s1"] = compute_sensitivity(s1_records, author_labels, human_labels, "S1")

    # S2: LLM, requires model availability
    if S2_AVAILABLE:
        client = OllamaClient(model=model)
        if client.is_available():
            print(f"\nRunning S2 ({model}) on {len(queries_50)} queries...")
            # S2 needs a JSONL-like format — write temp JSONL and call run_s2
            import tempfile, os
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as tf:
                for q in queries_50:
                    tf.write(json.dumps(q) + "\n")
                tmp_path = tf.name
            try:
                s2_out = run_s2(tmp_path, split_name="human_labeled_50", model=model)
                s2_records = [
                    {"qid": r.get("query_id") or r.get("id"), "pred_tools": r.get("pred_tools", [])}
                    for r in s2_out.get("records", [])
                ]
                report["s2"] = compute_sensitivity(s2_records, author_labels, human_labels, "S2")
            finally:
                os.unlink(tmp_path)
        else:
            print(f"  S2 SKIPPED: {model} not available in Ollama")
            report["s2"] = {"error": f"Model {model} not available"}
    else:
        report["s2"] = {"error": "S2 module not importable"}

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nLabel-noise sensitivity report saved to {OUT_FILE}")

    # Print summary
    for sys_name in ("s1", "s2"):
        res = report.get(sys_name, {})
        if "error" in res:
            print(f"  {sys_name.upper()}: {res['error']}")
        else:
            print(f"  {sys_name.upper()}: author F1={res['under_author_labels']['f1_mean']:.4f} "
                  f"human F1={res['under_human_labels']['f1_mean']:.4f} "
                  f"Δ={res['delta_f1']:+.4f}")
            print(f"        {res['interpretation']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    args = parser.parse_args()
    main(model=args.model)
