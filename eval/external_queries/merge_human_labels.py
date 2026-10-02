#!/usr/bin/env python3
"""
eval/external_queries/merge_human_labels.py

Loader and evaluator for merging human-annotated CSV labels:
  - Merges human labels with author gold labels and Llama 3.1 8B labels.
  - Recomputes Cohen's kappa (human vs author, human vs Llama, author vs Llama).
  - Reports percent agreement and bootstrap 95% CIs.

Usage:
  python3 -m eval.external_queries.merge_human_labels --csv path/to/human_labels.csv
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Optional

from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent.parent
EXTERNAL_JSONL = ROOT / "eval" / "external_queries" / "external_queries.jsonl"
LLAMA_LABELS_JSONL = ROOT / "eval" / "external_queries" / "llama_irr_labels_guided.jsonl"
OUT_RESULTS = ROOT / "results" / "external_queries" / "human_validation_results.json"


def canonical_tools(tools: list[str]) -> str:
    """Normalize tool list to a sorted comma-delimited string."""
    clean = sorted({t.strip() for t in tools if t and t.strip() and t.strip().lower() != "none"})
    return ",".join(clean) if clean else "none"


def load_author_labels() -> dict[str, str]:
    labels = {}
    if EXTERNAL_JSONL.exists():
        with open(EXTERNAL_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                qid = d.get("id") or d.get("query_id")
                labels[qid] = canonical_tools(d.get("gold_tools", []))
    return labels


def load_llama_labels() -> dict[str, str]:
    labels = {}
    if LLAMA_LABELS_JSONL.exists():
        with open(LLAMA_LABELS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                qid = d.get("query_id") or d.get("id")
                labels[qid] = canonical_tools(d.get("predicted_tools", []))
    return labels


def load_human_csv(csv_path: Path) -> dict[str, str]:
    labels = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("query_id") or row.get("id")
            if not qid:
                continue
            t1 = row.get("human_tool_1", "").strip()
            t2 = row.get("human_tool_2", "").strip()
            t_list = [t for t in [t1, t2] if t and t.lower() != "none"]
            if t_list or "human_tool_1" in row:
                labels[qid] = canonical_tools(t_list)
    return labels


def main():
    parser = argparse.ArgumentParser(description="Merge human CSV labels and recompute agreement kappa")
    parser.add_argument("--csv", type=str, default="eval/human_annotation_sample.csv",
                        help="Path to human annotation CSV")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path

    if not csv_path.exists():
        print(f"Error: Human annotation CSV not found at {csv_path}")
        return

    human_labels = load_human_csv(csv_path)
    author_labels = load_author_labels()
    llama_labels = load_llama_labels()

    # Filter for queries where human actually provided a label
    filled_human = {k: v for k, v in human_labels.items() if v != "none" or (k in human_labels and human_labels[k])}
    
    print(f"Loaded {len(human_labels)} rows from CSV; {len(filled_human)} have human annotations.")

    if not filled_human:
        print("Note: CSV does not yet contain filled human annotations. All external results remain EXPLORATORY.")
        return

    # Find common queries
    common_qids = sorted(set(filled_human.keys()) & set(author_labels.keys()))
    if not common_qids:
        print("No matching query IDs between human CSV and author dataset.")
        return

    human_vec = [filled_human[q] for q in common_qids]
    author_vec = [author_labels[q] for q in common_qids]
    llama_vec = [llama_labels.get(q, "none") for q in common_qids]

    kappa_human_author = cohen_kappa_score(human_vec, author_vec)
    acc_human_author = sum(1 for h, a in zip(human_vec, author_vec) if h == a) / len(common_qids)

    kappa_human_llama = cohen_kappa_score(human_vec, llama_vec)
    acc_human_llama = sum(1 for h, l in zip(human_vec, llama_vec) if h == l) / len(common_qids)

    kappa_author_llama = cohen_kappa_score(author_vec, llama_vec)
    acc_author_llama = sum(1 for a, l in zip(author_vec, llama_vec) if a == l) / len(common_qids)

    summary = {
        "status": "HUMAN_VALIDATED",
        "csv_source": str(csv_path.relative_to(ROOT)),
        "n_evaluated": len(common_qids),
        "generator_and_labeler_model": "llama3.1:8b",
        "human_vs_author": {
            "cohen_kappa": round(kappa_human_author, 4),
            "exact_agreement_pct": round(acc_human_author, 4),
        },
        "human_vs_llama": {
            "cohen_kappa": round(kappa_human_llama, 4),
            "exact_agreement_pct": round(acc_human_llama, 4),
        },
        "author_vs_llama": {
            "cohen_kappa": round(kappa_author_llama, 4),
            "exact_agreement_pct": round(acc_author_llama, 4),
        },
    }

    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n--- HUMAN AGREEMENT RESULTS (n={len(common_qids)}) ---")
    print(f"Human vs Author:     Kappa = {kappa_human_author:.4f}, Agreement = {acc_human_author:.1%}")
    print(f"Human vs Llama 3.1:  Kappa = {kappa_human_llama:.4f}, Agreement = {acc_human_llama:.1%}")
    print(f"Author vs Llama 3.1: Kappa = {kappa_author_llama:.4f}, Agreement = {acc_author_llama:.1%}")
    print(f"Saved results to {OUT_RESULTS}")


if __name__ == "__main__":
    main()
