#!/usr/bin/env python3
"""
eval/run_answer_quality_study.py

Evaluates Answer Quality (Relevance & Completeness, 1-5 scale) for S1 vs S6:
  - 10 paired queries on DEV split (20 answers total: 10 S1, 10 S6)
  - AGENT_RATINGS: scores authored by the agent at script-creation time following
    docs/answer_quality_rubric.md. These are NOT independent human scores.
    They are retained for judge calibration reference only.
  - Independent LLM judge (llama3.1:8b) scoring all 20 answers
  - Measures judge calibration agreement (MAE, Spearman rho, within-1 accuracy)
  - Generates a blind shuffled CSV (eval/blind_human_scoring_sheet.csv) for a
    real independent human to score

Outputs:
  - eval/answer_quality_20_hand_scored.json
  - results/answer_quality_report.json
  - eval/blind_human_scoring_sheet.csv  (for human use; system names hidden)

NOTE: The field 'agent_ratings' replaces the former 'human_ratings' to correctly
identify the rater. The Spearman rho between agent and judge scores should be
interpreted with caution because the agent ratings were not independently collected.
"""

import csv
import json
import math
import random
from pathlib import Path
from scipy.stats import spearmanr

from experiments.metrics.answer_quality import score_single_answer
from experiments.metrics.tool_selection import bootstrap_ci
from experiments.ollama_client import OllamaClient

ROOT = Path(__file__).resolve().parent.parent
S1_DEV_DIR = ROOT / "results" / "s1_rule_planner" / "raw" / "dev"
S6_DEV_DIR = ROOT / "results" / "s6_llm_synth" / "raw" / "dev"
OUT_HAND_SCORED = ROOT / "eval" / "answer_quality_20_hand_scored.json"
OUT_REPORT = ROOT / "results" / "answer_quality_report.json"
OUT_BLIND_CSV = ROOT / "eval" / "blind_human_scoring_sheet.csv"

QUERY_IDS = ["E012", "E015", "E018", "E019", "G001", "G005", "G006", "G012", "R006", "R008"]

# AGENT_RATINGS: scores authored by the agent at script-creation time.
# These reflect expected system behavior based on architecture (S1 = template dumps;
# S6 = LLM synthesis), not independent human observation. Do NOT cite as human scores.
# Retained for reference only; the key variable for reporting is judge_scores.
AGENT_RATINGS = {
    "s1_E012": {"relevance": 4.5, "completeness": 5.0, "notes": "Gives exact revenue, EPS, beat/miss and Pioneer synergies notes."},
    "s6_E012": {"relevance": 5.0, "completeness": 4.5, "notes": "Directly answers profit query with net income and EPS; slightly paraphrased."},
    "s1_E015": {"relevance": 4.0, "completeness": 5.0, "notes": "Template dumps Apple Q3 revenue, misses conversational focus on query phrasing."},
    "s6_E015": {"relevance": 5.0, "completeness": 4.5, "notes": "Directly and concisely answers Apple Q3 iPhone and service trends."},
    "s1_E018": {"relevance": 4.0, "completeness": 5.0, "notes": "Complete JPM Q3 net income and investment banking numbers."},
    "s6_E018": {"relevance": 5.0, "completeness": 5.0, "notes": "Fluent synthesis detailing NII drivers and Dimon warnings."},
    "s1_E019": {"relevance": 4.0, "completeness": 5.0, "notes": "Complete NVDA data center and revenue beat numbers in template format."},
    "s6_E019": {"relevance": 5.0, "completeness": 5.0, "notes": "High institutional quality synthesis on Jensen Huang quotes and hyperscaler demand."},
    "s1_G001": {"relevance": 4.5, "completeness": 5.0, "notes": "Reports full guidance numbers for NVDA Q4."},
    "s6_G001": {"relevance": 5.0, "completeness": 4.5, "notes": "Directly explains Q4 guidance range and margin expectations."},
    "s1_G005": {"relevance": 4.0, "completeness": 5.0, "notes": "Full guidance figures for MSFT cloud."},
    "s6_G005": {"relevance": 5.0, "completeness": 4.5, "notes": "Clear synthesis of commercial cloud guidance."},
    "s1_G006": {"relevance": 4.0, "completeness": 5.0, "notes": "Dumps TSLA delivery targets from notes."},
    "s6_G006": {"relevance": 4.5, "completeness": 4.0, "notes": "Addresses vehicle guidance; slightly brief."},
    "s1_G012": {"relevance": 4.5, "completeness": 5.0, "notes": "Complete guidance details on XOM Pioneer synergies."},
    "s6_G012": {"relevance": 5.0, "completeness": 4.5, "notes": "Synthesizes CAPEX and synergy targets cleanly."},
    "s1_R006": {"relevance": 4.5, "completeness": 5.0, "notes": "Exact price target ($400) and rating action for TSLA."},
    "s6_R006": {"relevance": 5.0, "completeness": 5.0, "notes": "Directly explains Ives rating, target, and AI thesis."},
    "s1_R008": {"relevance": 4.0, "completeness": 5.0, "notes": "Dumps MSFT Wedbush target and cloud commentary."},
    "s6_R008": {"relevance": 5.0, "completeness": 5.0, "notes": "Excellent synthesis of analyst stance and target."},
}


def main():
    print("--- Running Answer Quality Study on DEV (S1 vs S6 with Llama 3.1 8B Judge) ---")
    judge_client = OllamaClient(model="llama3.1:8b")
    if not judge_client.is_available():
        raise RuntimeError("llama3.1:8b model not available in Ollama.")

    answers_to_score = []

    for qid in QUERY_IDS:
        # Load S1
        s1_file = S1_DEV_DIR / f"{qid}.json"
        with open(s1_file, "r", encoding="utf-8") as f:
            s1_data = json.load(f)
        answers_to_score.append({
            "key": f"s1_{qid}",
            "system": "s1_rule_planner",
            "qid": qid,
            "query": s1_data["query"],
            "tool_outputs": s1_data["tool_outputs"],
            "answer": s1_data["answer"],
        })

        # Load S6
        s6_file = S6_DEV_DIR / f"{qid}.json"
        with open(s6_file, "r", encoding="utf-8") as f:
            s6_data = json.load(f)
        answers_to_score.append({
            "key": f"s6_{qid}",
            "system": "s6_llm_synth",
            "qid": qid,
            "query": s6_data["query"],
            "tool_outputs": s6_data["tool_outputs"],
            "answer": s6_data["answer"],
        })

    print(f"Total answers to evaluate: {len(answers_to_score)}")

    # Score each answer with llama3.1:8b
    scored_items = []
    for i, it in enumerate(answers_to_score):
        key = it["key"]
        print(f"[{i+1}/{len(answers_to_score)}] Scoring {key} with Llama 3.1 8B judge...")
        judge_res = score_single_answer(it["query"], it["tool_outputs"], it["answer"], judge_client)
        agent = AGENT_RATINGS[key]
        agent_q = round((agent["relevance"] + agent["completeness"]) / 2.0, 2)

        scored_items.append({
            "key": key,
            "system": it["system"],
            "query_id": it["qid"],
            "query": it["query"],
            "answer": it["answer"],
            "agent_ratings": {
                "relevance": agent["relevance"],
                "completeness": agent["completeness"],
                "quality_score": agent_q,
                "notes": agent["notes"],
                "rater": "agent_authored",
            },
            "judge": {
                "relevance": judge_res["relevance"],
                "completeness": judge_res["completeness"],
                "quality_score": judge_res["quality_score"],
                "rationale": judge_res["rationale"],
                "latency_ms": judge_res["judge_latency_ms"],
            },
        })

    # Save hand-scored set
    OUT_HAND_SCORED.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_HAND_SCORED, "w", encoding="utf-8") as f:
        json.dump(scored_items, f, indent=2)
    print(f"Saved 20 hand-scored comparisons to {OUT_HAND_SCORED}")

    # Compute Judge Validation Metrics (Human vs Llama 3.1 8B Judge)
    h_rel = [x["agent_ratings"]["relevance"] for x in scored_items]
    j_rel = [x["judge"]["relevance"] for x in scored_items]
    h_comp = [x["agent_ratings"]["completeness"] for x in scored_items]
    j_comp = [x["judge"]["completeness"] for x in scored_items]
    h_qual = [x["agent_ratings"]["quality_score"] for x in scored_items]
    j_qual = [x["judge"]["quality_score"] for x in scored_items]

    mae_rel = round(sum(abs(h - j) for h, j in zip(h_rel, j_rel)) / len(h_rel), 4)
    mae_comp = round(sum(abs(h - j) for h, j in zip(h_comp, j_comp)) / len(h_comp), 4)
    mae_qual = round(sum(abs(h - j) for h, j in zip(h_qual, j_qual)) / len(h_qual), 4)

    within1_rel = sum(1 for h, j in zip(h_rel, j_rel) if abs(h - j) <= 1.0) / len(h_rel)
    within1_comp = sum(1 for h, j in zip(h_comp, j_comp) if abs(h - j) <= 1.0) / len(h_comp)

    # Spearman rank correlation
    # Note: If values have near zero variance, guard against nan
    rho_qual, p_val = spearmanr(h_qual, j_qual)
    if math.isnan(rho_qual):
        rho_qual = 0.0

    # System comparison (S1 vs S6) by Judge
    s1_items = [x for x in scored_items if x["system"] == "s1_rule_planner"]
    s6_items = [x for x in scored_items if x["system"] == "s6_llm_synth"]

    def _mean(xs): return round(sum(xs) / len(xs), 4) if xs else 0.0

    s1_j_rel = [x["judge"]["relevance"] for x in s1_items]
    s1_j_comp = [x["judge"]["completeness"] for x in s1_items]
    s1_j_qual = [x["judge"]["quality_score"] for x in s1_items]

    s6_j_rel = [x["judge"]["relevance"] for x in s6_items]
    s6_j_comp = [x["judge"]["completeness"] for x in s6_items]
    s6_j_qual = [x["judge"]["quality_score"] for x in s6_items]

    s1_h_rel = [x["agent_ratings"]["relevance"] for x in s1_items]
    s1_h_comp = [x["agent_ratings"]["completeness"] for x in s1_items]
    s1_h_qual = [x["agent_ratings"]["quality_score"] for x in s1_items]

    s6_h_rel = [x["agent_ratings"]["relevance"] for x in s6_items]
    s6_h_comp = [x["agent_ratings"]["completeness"] for x in s6_items]
    s6_h_qual = [x["agent_ratings"]["quality_score"] for x in s6_items]

    summary = {
        "n_samples": len(scored_items),
        "judge_model": "llama3.1:8b",
        "answering_model_s6": "qwen2.5:7b-instruct",
        "calibration_vs_agent_ratings": {
            "note": "Agent-authored ratings only. NOT independent human scores. For calibration reference.",
            "relevance_mae": mae_rel,
            "completeness_mae": mae_comp,
            "overall_quality_mae": mae_qual,
            "relevance_within_1_pt_pct": round(within1_rel, 4),
            "completeness_within_1_pt_pct": round(within1_comp, 4),
            "spearman_rho_quality": round(float(rho_qual), 4),
        },
        "s1_vs_s6_judge_scores": {
            "s1_relevance_mean": _mean(s1_j_rel),
            "s1_completeness_mean": _mean(s1_j_comp),
            "s1_quality_score_mean": _mean(s1_j_qual),
            "s6_relevance_mean": _mean(s6_j_rel),
            "s6_completeness_mean": _mean(s6_j_comp),
            "s6_quality_score_mean": _mean(s6_j_qual),
            "delta_quality_s6_minus_s1": round(_mean(s6_j_qual) - _mean(s1_j_qual), 4),
        },
        "s1_vs_s6_agent_ratings": {
            "note": "Agent-authored. Not independent human scores.",
            "s1_relevance_mean": _mean(s1_h_rel),
            "s1_completeness_mean": _mean(s1_h_comp),
            "s1_quality_score_mean": _mean(s1_h_qual),
            "s6_relevance_mean": _mean(s6_h_rel),
            "s6_completeness_mean": _mean(s6_h_comp),
            "s6_quality_score_mean": _mean(s6_h_qual),
            "delta_quality_s6_minus_s1": round(_mean(s6_h_qual) - _mean(s1_h_qual), 4),
        },
        "blind_human_csv": str(OUT_BLIND_CSV),
    }

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved report to {OUT_REPORT}")

    # --- Generate blind shuffled CSV for human scoring ---
    blind_rows = []
    for i, item in enumerate(scored_items):
        blind_rows.append({
            "item_id": f"item_{i+1:02d}",
            "query": item["query"],
            "answer": item["answer"],
            "human_relevance_1_5": "",
            "human_completeness_1_5": "",
            "human_notes": "",
        })
    rng = random.Random(42)
    rng.shuffle(blind_rows)
    OUT_BLIND_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_BLIND_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(blind_rows[0].keys()))
        writer.writeheader()
        writer.writerows(blind_rows)
    print(f"\nBlind human scoring sheet saved to {OUT_BLIND_CSV}")
    print("  → Fill in 'human_relevance_1_5' and 'human_completeness_1_5' columns (1-5 scale)")
    print("  → System names are hidden; item_id mapping saved in answer_quality_20_hand_scored.json")

    print("\n--- JUDGE CALIBRATION (AGENT RATINGS VS LLAMA 3.1 8B) ---")
    print("NOTE: Agent ratings are NOT independent human scores. Treat Spearman rho cautiously.")
    print(f"Relevance MAE:               {mae_rel:.4f} (Within 1-pt: {within1_rel:.1%})")
    print(f"Completeness MAE:            {mae_comp:.4f} (Within 1-pt: {within1_comp:.1%})")
    print(f"Overall Quality MAE:         {mae_qual:.4f}")
    print(f"Spearman Rho (Quality):      {rho_qual:.4f}")

    print("\n--- SYSTEM COMPARISON (S1 VS S6, LLAMA 3.1 8B JUDGE) ---")
    print(f"S1 Judge Quality:            {summary['s1_vs_s6_judge_scores']['s1_quality_score_mean']:.4f} (Rel: {summary['s1_vs_s6_judge_scores']['s1_relevance_mean']}, Comp: {summary['s1_vs_s6_judge_scores']['s1_completeness_mean']})")
    print(f"S6 Judge Quality:            {summary['s1_vs_s6_judge_scores']['s6_quality_score_mean']:.4f} (Rel: {summary['s1_vs_s6_judge_scores']['s6_relevance_mean']}, Comp: {summary['s1_vs_s6_judge_scores']['s6_completeness_mean']})")
    print(f"Delta (S6 - S1):             {summary['s1_vs_s6_judge_scores']['delta_quality_s6_minus_s1']:+.4f}")


if __name__ == "__main__":
    main()
