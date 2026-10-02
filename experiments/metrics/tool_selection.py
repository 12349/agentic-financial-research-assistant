"""
experiments/metrics/tool_selection.py

Computes tool-selection metrics for a list of query results.

Inputs per result record (dict):
  - gold_tools: list[str]   gold standard tool set from eval set
  - pred_tools: list[str]   tools predicted by the planner
  - category:   str         single_tool | dual_tool | ambiguous | no_data

Outputs:
  - precision, recall, F1 (micro-averaged across queries)
  - exact_set_match rate
  - abstention_accuracy on no_data queries
  - tool_calls_per_query mean
  - per-category breakdown
  - bootstrap 95% CI on all primary scalars
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from experiments.config import BOOTSTRAP_SEED, CI_ALPHA, N_BOOTSTRAP, SEED


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------

def query_metrics(gold: Sequence[str], pred: Sequence[str]) -> dict:
    """
    Return per-query precision, recall, F1, and exact_match for tool sets.

    Parameters
    ----------
    gold : list of tool names from the gold label
    pred : list of tool names returned by the planner

    Returns
    -------
    dict with keys: precision, recall, f1, exact_match
    """
    g = frozenset(gold)
    p = frozenset(pred)

    tp = len(g & p)
    fp = len(p - g)
    fn = len(g - p)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    exact     = 1.0 if g == p else 0.0

    return {
        "precision":   round(precision, 6),
        "recall":      round(recall, 6),
        "f1":          round(f1, 6),
        "exact_match": exact,
        "tp": tp, "fp": fp, "fn": fn,
    }


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = CI_ALPHA,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """
    Non-parametric bootstrap 95% CI for the mean of `values`.
    Returns (lower, upper).
    """
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(rng.choices(values, k=n)) / n
        for _ in range(n_bootstrap)
    )
    lo_idx = int((alpha / 2) * n_bootstrap)
    hi_idx = int((1 - alpha / 2) * n_bootstrap)
    return round(means[lo_idx], 4), round(means[hi_idx], 4)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

@dataclass
class ToolSelectionResult:
    n_queries: int
    precision_mean:     float
    recall_mean:        float
    f1_mean:            float
    exact_match_rate:   float
    abstention_acc:     float       # fraction of no_data queries answered correctly
    tool_calls_per_q:   float

    precision_ci:    tuple[float, float]
    recall_ci:       tuple[float, float]
    f1_ci:           tuple[float, float]
    exact_ci:        tuple[float, float]

    per_category:    dict = field(default_factory=dict)
    per_query:       list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_queries":         self.n_queries,
            "precision_mean":    self.precision_mean,
            "precision_ci_95":   list(self.precision_ci),
            "recall_mean":       self.recall_mean,
            "recall_ci_95":      list(self.recall_ci),
            "f1_mean":           self.f1_mean,
            "f1_ci_95":          list(self.f1_ci),
            "exact_match_rate":  self.exact_match_rate,
            "exact_ci_95":       list(self.exact_ci),
            "abstention_acc":    self.abstention_acc,
            "tool_calls_per_q":  self.tool_calls_per_q,
            "per_category":      self.per_category,
        }


def evaluate_tool_selection(records: list[dict]) -> ToolSelectionResult:
    """
    Aggregate tool-selection metrics across a list of result records.

    Each record must have:
      gold_tools: list[str]
      pred_tools: list[str]
      category:   str
    """
    precisions, recalls, f1s, exacts, n_calls = [], [], [], [], []
    per_cat: dict[str, list] = {}
    no_data_total, no_data_correct = 0, 0
    per_query_rows = []

    for rec in records:
        gold = rec.get("gold_tools", [])
        pred = rec.get("pred_tools", [])
        cat  = rec.get("category", "unknown")

        qm = query_metrics(gold, pred)
        precisions.append(qm["precision"])
        recalls.append(qm["recall"])
        f1s.append(qm["f1"])
        exacts.append(qm["exact_match"])
        n_calls.append(len(pred))

        per_cat.setdefault(cat, {"precision": [], "recall": [], "f1": [], "exact": []})
        per_cat[cat]["precision"].append(qm["precision"])
        per_cat[cat]["recall"].append(qm["recall"])
        per_cat[cat]["f1"].append(qm["f1"])
        per_cat[cat]["exact"].append(qm["exact_match"])

        # Abstention: on no_data queries, correct = pred_tools is empty OR
        # the answer text contains "no data" / "not found" (checked separately)
        if cat == "no_data":
            no_data_total += 1
            if len(pred) == 0 or gold == pred:
                no_data_correct += 1

        per_query_rows.append({
            "query_id": rec.get("query_id", ""),
            "category": cat,
            **qm,
            "pred_tools": pred,
            "gold_tools": gold,
        })

    def _mean(xs): return round(sum(xs) / len(xs), 4) if xs else 0.0

    # Per-category summary
    per_category_summary = {}
    for cat, vals in per_cat.items():
        per_category_summary[cat] = {
            "n": len(vals["f1"]),
            "precision_mean": _mean(vals["precision"]),
            "recall_mean":    _mean(vals["recall"]),
            "f1_mean":        _mean(vals["f1"]),
            "exact_rate":     _mean(vals["exact"]),
        }

    abstention_acc = (no_data_correct / no_data_total
                      if no_data_total > 0 else float("nan"))

    return ToolSelectionResult(
        n_queries        = len(records),
        precision_mean   = _mean(precisions),
        recall_mean      = _mean(recalls),
        f1_mean          = _mean(f1s),
        exact_match_rate = _mean(exacts),
        abstention_acc   = round(abstention_acc, 4),
        tool_calls_per_q = round(_mean(n_calls), 2),
        precision_ci     = bootstrap_ci(precisions),
        recall_ci        = bootstrap_ci(recalls),
        f1_ci            = bootstrap_ci(f1s),
        exact_ci         = bootstrap_ci(exacts),
        per_category     = per_category_summary,
        per_query        = per_query_rows,
    )


# ---------------------------------------------------------------------------
# Significance tests
# ---------------------------------------------------------------------------

def mcnemar_test(
    exact_a: list[float],
    exact_b: list[float],
) -> dict:
    """
    McNemar's test on paired exact-set-match binary vectors.
    Returns chi2 statistic, p-value, and interpretation.
    """
    import math
    assert len(exact_a) == len(exact_b), "Paired lists must have equal length"

    # Discordant pairs
    b = sum(1 for a, b_ in zip(exact_a, exact_b) if a == 1 and b_ == 0)  # A right, B wrong
    c = sum(1 for a, b_ in zip(exact_a, exact_b) if a == 0 and b_ == 1)  # A wrong, B right

    n_discordant = b + c
    if n_discordant == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": b, "c": c,
                "note": "No discordant pairs — systems agree on every query"}

    # With continuity correction (Yates)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # p-value from chi-sq distribution (1 df) — approximated
    # Using scipy if available, else normal approximation
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = float(chi2_dist.sf(chi2, df=1))
    except ImportError:
        # z approximation: z = (b-c)/sqrt(b+c)
        z = (b - c) / math.sqrt(b + c)
        # two-tailed p from standard normal CDF approximation
        t = abs(z)
        p_value = 2 * (1 - 0.5 * (1 + math.erf(t / math.sqrt(2))))

    return {
        "chi2":    round(chi2, 4),
        "p_value": round(p_value, 4),
        "b": b, "c": c,
        "n_discordant": n_discordant,
        "significant_at_0_05": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_eval_split(path: str) -> list[dict]:
    """Load a .jsonl eval split file. Returns list of query dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_result(result: ToolSelectionResult, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.as_dict(), f, indent=2)
    print(f"  Saved: {path}")
