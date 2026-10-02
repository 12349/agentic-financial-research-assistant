"""
experiments/compute_statistical_tests.py

Computes paired statistical tests and effect sizes with bootstrap 95% CIs:
  1. Synthesis metrics: S1 vs S6 vs S7 on held-out test split (n=47)
     - Citation validity
     - Numeric faithfulness
     - Block-level unsupported rate
     - Strict sentence-level unsupported rate
     - Abstention accuracy (explicitly noted as n=7, too small for significance claims)
  2. Tool selection metrics: S2 vs S5 on held-out test split (n=47)
     - Tool Selection F1
     - Exact Match Rate (McNemar's test)
     - Tool calls per query

Outputs:
  - results/statistical_significance.json
"""

import glob
import json
import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, ttest_rel

from experiments.config import BOOTSTRAP_SEED, CI_ALPHA, N_BOOTSTRAP, RESULTS_DIR
from experiments.metrics.synthesis_quality import (
    abstention_correctness,
    citation_validity,
    numeric_faithfulness,
    unsupported_claim_rate,
)
from experiments.metrics.tool_selection import query_metrics

ROOT = Path(__file__).resolve().parent.parent
S1_TEST_DIR = ROOT / "results" / "s1_rule_planner" / "raw" / "test"
S2_TEST_DIR = ROOT / "results" / "s2_llm_planner" / "raw" / "test"
S5_TEST_DIR = ROOT / "results" / "s5_react" / "raw" / "test"
S6_TEST_DIR = ROOT / "results" / "s6_llm_synth" / "raw" / "test"
S7_TEST_DIR = ROOT / "results" / "s7_hybrid_synth" / "raw" / "test"
OUT_FILE = ROOT / "results" / "statistical_significance.json"


def paired_bootstrap_ci(
    diffs: Sequence[float],
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = CI_ALPHA,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Compute bootstrap 95% CI on the mean paired difference."""
    if not diffs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_bootstrap):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low_idx = int(alpha / 2.0 * n_bootstrap)
    high_idx = int((1.0 - alpha / 2.0) * n_bootstrap)
    return round(means[low_idx], 4), round(means[high_idx], 4)


def compute_paired_comparison(vec_a: list[float], vec_b: list[float], name: str) -> dict:
    """Compute paired t-test, Wilcoxon signed-rank test, Cohen's d, and bootstrap CI."""
    pairs = [(a, b) for a, b in zip(vec_a, vec_b) if not math.isnan(a) and not math.isnan(b)]
    if not pairs:
        return {"n_pairs": 0, "note": "No valid pairs"}

    a_vals = [p[0] for p in pairs]
    b_vals = [p[1] for p in pairs]
    diffs = [a - b for a, b in pairs]

    mean_a = round(float(np.mean(a_vals)), 4)
    mean_b = round(float(np.mean(b_vals)), 4)
    mean_diff = round(float(np.mean(diffs)), 4)
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

    # Cohen's d_z for paired samples
    cohens_d = round(mean_diff / std_diff, 4) if std_diff > 1e-8 else 0.0

    # Paired t-test
    t_stat, t_pval = ttest_rel(a_vals, b_vals) if std_diff > 1e-8 else (0.0, 1.0)

    # Wilcoxon signed rank test
    try:
        # Check if all differences are zero
        if all(abs(d) < 1e-8 for d in diffs):
            w_stat, w_pval = 0.0, 1.0
        else:
            w_res = wilcoxon(a_vals, b_vals, zero_method="pratt")
            w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
    except Exception:
        w_stat, w_pval = float("nan"), float("nan")

    ci_low, ci_high = paired_bootstrap_ci(diffs)

    return {
        "n_pairs": len(pairs),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_difference": mean_diff,
        "difference_ci_95": [ci_low, ci_high],
        "cohens_d": cohens_d,
        "paired_t_stat": round(float(t_stat), 4),
        "paired_t_pvalue": round(float(t_pval), 6),
        "wilcoxon_stat": round(float(w_stat), 4),
        "wilcoxon_pvalue": round(float(w_pval), 6),
        "statistically_significant_p05": bool(w_pval < 0.05),
    }


def mcnemar_test(vec_a: list[int], vec_b: list[int]) -> dict:
    """Compute McNemar's exact test for paired binary exact-match outcomes."""
    # Contingency table:
    #             B=1    B=0
    #     A=1      a      b   (A=1, B=0)
    #     A=0      c      d   (A=0, B=1)
    b_discordant = sum(1 for a, b in zip(vec_a, vec_b) if a == 1 and b == 0)
    c_discordant = sum(1 for a, b in zip(vec_a, vec_b) if a == 0 and b == 1)

    # Binomial exact p-value for discordant pairs
    n_discordant = b_discordant + c_discordant
    if n_discordant > 0:
        p_val = stats.binomtest(b_discordant, n_discordant, 0.5, alternative="two-sided").pvalue
    else:
        p_val = 1.0

    diffs = [a - b for a, b in zip(vec_a, vec_b)]
    ci_low, ci_high = paired_bootstrap_ci(diffs)

    return {
        "n_total": len(vec_a),
        "a_correct": sum(vec_a),
        "b_correct": sum(vec_b),
        "discordant_a_only": b_discordant,
        "discordant_b_only": c_discordant,
        "mcnemar_exact_pvalue": round(float(p_val), 6),
        "difference_exact_match": round(sum(diffs) / len(diffs), 4),
        "difference_ci_95": [ci_low, ci_high],
        "statistically_significant_p05": bool(p_val < 0.05),
    }


def load_all_records(directory: Path) -> dict[str, dict]:
    recs = {}
    for p in directory.glob("*.json"):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            qid = d.get("id") or d.get("query_id")
            recs[qid] = d
    return recs


def compute_all_stats() -> dict:
    s1_recs = load_all_records(S1_TEST_DIR)
    s2_recs = load_all_records(S2_TEST_DIR)
    s5_recs = load_all_records(S5_TEST_DIR)
    s6_recs = load_all_records(S6_TEST_DIR)
    s7_recs = load_all_records(S7_TEST_DIR)

    common_qids = sorted(set(s1_recs.keys()) & set(s6_recs.keys()))
    has_s7 = bool(s7_recs)

    print(f"Loaded records on test split: S1={len(s1_recs)}, S2={len(s2_recs)}, S5={len(s5_recs)}, S6={len(s6_recs)}, S7={len(s7_recs)}")

    # 1. Synthesis metrics per query
    metrics_by_sys = {"s1": {}, "s6": {}, "s7": {}}
    for qid in common_qids:
        # S1
        ans1 = s1_recs[qid]["answer"]
        tools1 = s1_recs[qid]["tool_outputs"]
        metrics_by_sys["s1"][qid] = {
            "cit": citation_validity(ans1, tools1)["valid_fraction"],
            "faith": numeric_faithfulness(ans1, tools1)["faithful_fraction"],
            "unsup_block": unsupported_claim_rate(ans1)["unsupported_rate"],
            "unsup_strict": unsupported_claim_rate(ans1)["strict_sentence_unsupported_rate"],
        }
        # S6
        ans6 = s6_recs[qid]["answer"]
        tools6 = s6_recs[qid]["tool_outputs"]
        metrics_by_sys["s6"][qid] = {
            "cit": citation_validity(ans6, tools6)["valid_fraction"],
            "faith": numeric_faithfulness(ans6, tools6)["faithful_fraction"],
            "unsup_block": unsupported_claim_rate(ans6)["unsupported_rate"],
            "unsup_strict": unsupported_claim_rate(ans6)["strict_sentence_unsupported_rate"],
        }
        # S7
        if has_s7 and qid in s7_recs:
            ans7 = s7_recs[qid]["answer"]
            tools7 = s7_recs[qid]["tool_outputs"]
            metrics_by_sys["s7"][qid] = {
                "cit": citation_validity(ans7, tools7)["valid_fraction"],
                "faith": numeric_faithfulness(ans7, tools7)["faithful_fraction"],
                "unsup_block": unsupported_claim_rate(ans7)["unsupported_rate"],
                "unsup_strict": unsupported_claim_rate(ans7)["strict_sentence_unsupported_rate"],
            }

    # Abstention on no_data (test split)
    no_data_qids = [qid for qid in common_qids if s1_recs[qid].get("category") == "no_data"]
    abst_s1 = [1 if abstention_correctness(s1_recs[q]["answer"], "no_data")["correct"] else 0 for q in no_data_qids]
    abst_s6 = [1 if abstention_correctness(s6_recs[q]["answer"], "no_data")["correct"] else 0 for q in no_data_qids]
    abst_s7 = [1 if abstention_correctness(s7_recs[q]["answer"], "no_data")["correct"] else 0 for q in no_data_qids] if has_s7 else []

    synth_comparisons = {}
    pairs_to_test = [("s1", "s6"), ("s7", "s6"), ("s1", "s7")] if has_s7 else [("s1", "s6")]

    for sys_a, sys_b in pairs_to_test:
        pair_key = f"{sys_a}_vs_{sys_b}"
        synth_comparisons[pair_key] = {}
        for m in ["cit", "faith", "unsup_block", "unsup_strict"]:
            vec_a = [metrics_by_sys[sys_a][q][m] for q in common_qids if q in metrics_by_sys[sys_a] and q in metrics_by_sys[sys_b]]
            vec_b = [metrics_by_sys[sys_b][q][m] for q in common_qids if q in metrics_by_sys[sys_a] and q in metrics_by_sys[sys_b]]
            synth_comparisons[pair_key][m] = compute_paired_comparison(vec_a, vec_b, m)

    # 2. Tool selection: S2 vs S5 on test split
    s2_s5_qids = sorted(set(s2_recs.keys()) & set(s5_recs.keys()))
    s2_f1s, s5_f1s = [], []
    s2_exacts, s5_exacts = [], []
    s2_calls, s5_calls = [], []

    for qid in s2_s5_qids:
        gold = s2_recs[qid].get("gold_tools", [])
        m2 = query_metrics(gold, s2_recs[qid].get("pred_tools", []))
        m5 = query_metrics(gold, s5_recs[qid].get("pred_tools", []))
        s2_f1s.append(m2["f1"])
        s5_f1s.append(m5["f1"])
        s2_exacts.append(int(m2["exact_match"] == 1.0))
        s5_exacts.append(int(m5["exact_match"] == 1.0))
        s2_calls.append(len(s2_recs[qid].get("pred_tools", [])))
        s5_calls.append(len(s5_recs[qid].get("pred_tools", [])))

    tool_selection_comparison = {
        "n_queries": len(s2_s5_qids),
        "f1": compute_paired_comparison(s2_f1s, s5_f1s, "f1"),
        "exact_match": mcnemar_test(s2_exacts, s5_exacts),
        "tool_calls_per_query": compute_paired_comparison(s2_calls, s5_calls, "tool_calls"),
    }

    report = {
        "test_split_n": len(common_qids),
        "synthesis_paired_tests": synth_comparisons,
        "abstention_descriptive": {
            "n_no_data_queries": len(no_data_qids),
            "sample_size_limitation": "Abstention sample size (n=7) is too small for statistical significance claims. Results are reported descriptively without p-values.",
            "s1_abstention_accuracy": round(sum(abst_s1) / len(abst_s1), 4) if abst_s1 else 0.0,
            "s6_abstention_accuracy": round(sum(abst_s6) / len(abst_s6), 4) if abst_s6 else 0.0,
            "s7_abstention_accuracy": round(sum(abst_s7) / len(abst_s7), 4) if abst_s7 else 0.0,
        },
        "tool_selection_s2_vs_s5": tool_selection_comparison,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nStatistical significance report written to {OUT_FILE}")
    return report


if __name__ == "__main__":
    compute_all_stats()
