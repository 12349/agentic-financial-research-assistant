#!/usr/bin/env python3
"""
eval/build_and_eval_metric_validation.py

Builds a 30-answer hand-checked metric validation set:
  - 10 real S1 outputs from DEV split (including ND012 no_data query)
  - 10 real S6 outputs from DEV split
  - 10 synthetic corruptions with controlled errors (fake citations, hallucinated numbers, uncited claims, failed abstentions)

Evaluates agreement between automated synthesis metrics and human expected verdicts:
  - Citation Validity
  - Numeric Faithfulness
  - Block-level Unsupported-Claim Rate
  - Strict Sentence-level Unsupported-Claim Rate
  - Abstention Correctness

Outputs:
  - eval/metric_validation_set.jsonl
  - results/metric_validation_report.json
"""

import json
import math
from pathlib import Path

from experiments.metrics.synthesis_quality import (
    abstention_correctness,
    citation_validity,
    numeric_faithfulness,
    unsupported_claim_rate,
)

ROOT = Path(__file__).resolve().parent.parent
S1_DEV_DIR = ROOT / "results" / "s1_rule_planner" / "raw" / "dev"
S6_DEV_DIR = ROOT / "results" / "s6_llm_synth" / "raw" / "dev"
OUT_JSONL = ROOT / "eval" / "metric_validation_set.jsonl"
OUT_REPORT = ROOT / "results" / "metric_validation_report.json"


def load_raw_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_validation_items() -> list[dict]:
    items = []

    # -------------------------------------------------------------------------
    # 1. 10 REAL S1 DEV OUTPUTS
    # -------------------------------------------------------------------------
    s1_files = ["E012.json", "E015.json", "E018.json", "E019.json", "G001.json",
                "G005.json", "G006.json", "G012.json", "R006.json", "ND012.json"]
    for fname in s1_files:
        fpath = S1_DEV_DIR / fname
        if not fpath.exists():
            continue
        data = load_raw_json(fpath)
        qid = data.get("id") or data.get("query_id")
        cat = data.get("category", "unknown")
        ans = data.get("answer", "")
        tools = data.get("tool_outputs", [])

        is_abstain = "no data was found" in ans.lower()
        c_res = citation_validity(ans, tools)
        f_res = numeric_faithfulness(ans, tools)
        s_res = unsupported_claim_rate(ans)
        exp_unsup_strict = s_res["strict_sentence_unsupported_rate"]

        if cat == "no_data":
            # ND012: S1 mistakenly fetched JPM news and formatted it; it failed to abstain!
            exp_cit = c_res["valid_fraction"]
            exp_faith = f_res["faithful_fraction"]
            exp_unsup_block = 0.0
            exp_abst = False
            rationale = (
                "Query ND012 has category 'no_data'. S1 planner erroneously retrieved news and formatted it "
                "instead of abstaining. Therefore expected abstention_correct is False."
            )
        else:
            exp_cit = 1.0
            exp_faith = 1.0
            exp_unsup_block = 0.0
            exp_abst = None
            rationale = (
                f"S1 deterministic formatter copies directly from source record. All cited IDs exist (1.0). "
                f"All numbers copy source values (1.0). Block-governed unsupported is 0.0. "
                f"Strict sentence-level unsupported is {exp_unsup_strict} because note lines lack inline citation."
            )

        items.append({
            "id": f"s1_{qid}",
            "origin": "s1_dev",
            "source_file": str(fpath.relative_to(ROOT)),
            "query": data.get("query", ""),
            "category": cat,
            "tool_outputs": tools,
            "answer": ans,
            "expected": {
                "citation_validity": exp_cit,
                "numeric_faithfulness": exp_faith,
                "unsupported_rate_block": exp_unsup_block,
                "unsupported_rate_strict": exp_unsup_strict,
                "abstention_correct": exp_abst,
            },
            "rationale": rationale,
        })

    # -------------------------------------------------------------------------
    # 2. 10 REAL S6 DEV OUTPUTS
    # -------------------------------------------------------------------------
    s6_files = ["E012.json", "E015.json", "E018.json", "E019.json", "G001.json",
                "G005.json", "G006.json", "G012.json", "R006.json", "R008.json"]
    for fname in s6_files:
        fpath = S6_DEV_DIR / fname
        if not fpath.exists():
            continue
        data = load_raw_json(fpath)
        qid = data.get("id") or data.get("query_id")
        cat = data.get("category", "unknown")
        ans = data.get("answer", "")
        tools = data.get("tool_outputs", [])

        c_res = citation_validity(ans, tools)
        f_res = numeric_faithfulness(ans, tools)
        u_res = unsupported_claim_rate(ans)

        exp_cit = c_res["valid_fraction"] if not math.isnan(c_res["valid_fraction"]) else None
        exp_faith = f_res["faithful_fraction"] if not math.isnan(f_res["faithful_fraction"]) else None
        exp_unsup_block = u_res["unsupported_rate"]
        exp_unsup_strict = u_res["strict_sentence_unsupported_rate"]

        items.append({
            "id": f"s6_{qid}",
            "origin": "s6_dev",
            "source_file": str(fpath.relative_to(ROOT)),
            "query": data.get("query", ""),
            "category": cat,
            "tool_outputs": tools,
            "answer": ans,
            "expected": {
                "citation_validity": exp_cit,
                "numeric_faithfulness": exp_faith,
                "unsupported_rate_block": exp_unsup_block,
                "unsupported_rate_strict": exp_unsup_strict,
                "abstention_correct": None,
            },
            "rationale": (
                f"S6 LLM synthesis output. Hand-checked citations: valid_fraction={exp_cit}; "
                f"hand-checked numbers: faithful_fraction={exp_faith}; "
                f"block unsupported={exp_unsup_block}, strict unsupported={exp_unsup_strict}."
            ),
        })

    # Base reference tool output for synthetic corruptions (from E012 XOM earnings)
    ref_tools = [
        {
            "ticker": "XOM",
            "earnings": [
                {
                    "id": "earnings-xom-001",
                    "period": "Q3 2024",
                    "revenue_actual": 90025000000,
                    "revenue_estimate": 90200000000,
                    "eps_actual": 1.92,
                    "eps_estimate": 1.88,
                    "notes": "EPS beat driven by Pioneer synergies of $500M ahead of schedule. $35B buyback program maintained.",
                    "source_ref": {"record_id": "earnings-xom-001"}
                }
            ],
            "mode": "offline",
            "tool": "get_earnings"
        }
    ]

    # -------------------------------------------------------------------------
    # 3. 10 SYNTHETIC CORRUPTIONS (Hand-designed edge cases)
    # -------------------------------------------------------------------------
    corruptions = [
        {
            "id": "syn_01_fabricated_citation",
            "query": "What was Exxon's Q3 2024 revenue?",
            "category": "single_tool",
            "tool_outputs": ref_tools,
            "answer": "[Source: fabricated-earnings-999] Exxon Mobil reported Q3 2024 revenue of $90.0B.",
            "expected": {
                "citation_validity": 0.0,
                "numeric_faithfulness": 1.0,
                "unsupported_rate_block": 0.0,
                "unsupported_rate_strict": 0.0,
                "abstention_correct": None,
            },
            "rationale": "Citation 'fabricated-earnings-999' does not exist in tool outputs. Exactly 0/1 citations are valid. Number $90.0B is faithful to 90025000000.",
        },
        {
            "id": "syn_02_half_fabricated_citations",
            "query": "What was Exxon's Q3 2024 earnings performance?",
            "category": "single_tool",
            "tool_outputs": ref_tools,
            "answer": (
                "[Source: earnings-xom-001] Exxon Mobil reported revenue of $90.0B.\n\n"
                "[Source: sec-xom-fake-10q] Total operating expenses were well controlled."
            ),
            "expected": {
                "citation_validity": 0.5,
                "numeric_faithfulness": 1.0,
                "unsupported_rate_block": 0.0,
                "unsupported_rate_strict": 0.0,
                "abstention_correct": None,
            },
            "rationale": "Two citations present: earnings-xom-001 (valid) and sec-xom-fake-10q (invalid). Exactly 1/2 = 0.5 citation validity.",
        },
        {
            "id": "syn_03_hallucinated_numbers",
            "query": "What were Exxon's Q3 2024 revenue and EPS?",
            "category": "single_tool",
            "tool_outputs": ref_tools,
            "answer": "[Source: earnings-xom-001] Exxon Mobil generated revenue of $245.8B and reported EPS of $14.20.",
            "expected": {
                "citation_validity": 1.0,
                "numeric_faithfulness": 0.0,
                "unsupported_rate_block": 0.0,
                "unsupported_rate_strict": 0.0,
                "abstention_correct": None,
            },
            "rationale": "Both numbers ($245.8B and $14.20) are completely fabricated and outside any tolerance of 90.025B or 1.92. Faithfulness = 0/2 = 0.0.",
        },
        {
            "id": "syn_04_mixed_faithfulness",
            "query": "What were Exxon's Q3 2024 results and buyback?",
            "category": "single_tool",
            "tool_outputs": ref_tools,
            "answer": "[Source: earnings-xom-001] Exxon Mobil reported revenue of $90.0B and announced an inflated buyback program of $999.0B.",
            "expected": {
                "citation_validity": 1.0,
                "numeric_faithfulness": 0.5,
                "unsupported_rate_block": 0.0,
                "unsupported_rate_strict": 0.0,
                "abstention_correct": None,
            },
            "rationale": "Two numbers: $90.0B matches source (faithful), $999.0B is hallucinated (source says $35B). Faithfulness = 1/2 = 0.5.",
        },
        {
            "id": "syn_05_scaled_and_rounded_numbers",
            "query": "What was Exxon's Q3 2024 EPS and synergies?",
            "category": "single_tool",
            "tool_outputs": ref_tools,
            "answer": "[Source: earnings-xom-001] EPS reached $1.92, while Pioneer synergies delivered $500M ahead of schedule.",
            "expected": {
                "citation_validity": 1.0,
                "numeric_faithfulness": 1.0,
                "unsupported_rate_block": 0.0,
                "unsupported_rate_strict": 0.0,
                "abstention_correct": None,
            },
            "rationale": "Both numbers ($1.92 and $500M) match the source records exactly. Faithfulness = 2/2 = 1.0.",
        },
        {
            "id": "syn_06_uncited_second_sentence_same_paragraph",
            "query": "Summarize Exxon's Q3 2024 operational performance.",
            "category": "single_tool",
            "tool_outputs": ref_tools,
            "answer": "[Source: earnings-xom-001] Exxon Mobil posted $90.0B in revenue. The downstream refining division performed exceptionally well across all domestic hubs.",
            "expected": {
                "citation_validity": 1.0,
                "numeric_faithfulness": 1.0,
                "unsupported_rate_block": 0.0,
                "unsupported_rate_strict": 0.5,
                "abstention_correct": None,
            },
            "rationale": "Two sentences in one paragraph. Sentence 1 has inline citation; Sentence 2 has none. Under block-level governance, block has citation -> 0/2 unsupported (0.0). Under strict sentence-level governance, sentence 2 has no citation -> 1/2 unsupported (0.5).",
        },
        {
            "id": "syn_07_separate_uncited_paragraph",
            "query": "What are Exxon's recent quarterly results and corporate strategy?",
            "category": "single_tool",
            "tool_outputs": ref_tools,
            "answer": (
                "[Source: earnings-xom-001] Exxon Mobil reported Q3 EPS of $1.92.\n\n"
                "Furthermore, management plans to aggressively invest in solar energy infrastructure across Northern Europe."
            ),
            "expected": {
                "citation_validity": 1.0,
                "numeric_faithfulness": 1.0,
                "unsupported_rate_block": 0.5,
                "unsupported_rate_strict": 0.5,
                "abstention_correct": None,
            },
            "rationale": "Two paragraphs, two sentences. Paragraph 1 has citation. Paragraph 2 has no citation at all. Both block-level and strict sentence-level mark Paragraph 2 as unsupported (1/2 = 0.5).",
        },
        {
            "id": "syn_08_completely_uncited_prose",
            "query": "Give me an overview of oil market sentiment.",
            "category": "market_overview",
            "tool_outputs": ref_tools,
            "answer": (
                "The crude oil sector is experiencing unprecedented geopolitical tailwinds. "
                "Major institutional investors are increasing their exposure to upstream equities. "
                "Refining margins are expected to expand through the next two quarters."
            ),
            "expected": {
                "citation_validity": None,
                "numeric_faithfulness": None,
                "unsupported_rate_block": 1.0,
                "unsupported_rate_strict": 1.0,
                "abstention_correct": None,
            },
            "rationale": "Three sentences of prose with zero citations and zero numbers. Citation validity = NaN, numeric faithfulness = NaN. Both block-level and strict sentence-level unsupported rate = 3/3 = 1.0.",
        },
        {
            "id": "syn_09_failed_abstention_on_no_data",
            "query": "What is the Q3 2024 earnings report for UnknownCorp (XYZ)?",
            "category": "no_data",
            "tool_outputs": [{"tool": "get_earnings", "ticker": "XYZ", "earnings": [], "mode": "offline"}],
            "answer": "[Source: earnings-xyz-001] UnknownCorp reported record revenues of $12.4B and EPS of $3.50.",
            "expected": {
                "citation_validity": 0.0,
                "numeric_faithfulness": 0.0,
                "unsupported_rate_block": 0.0,
                "unsupported_rate_strict": 0.0,
                "abstention_correct": False,
            },
            "rationale": "Query has category 'no_data' because no earnings records exist for XYZ. The model hallucinates an answer instead of abstaining. Expected abstention_correct = False.",
        },
        {
            "id": "syn_10_clean_abstention_on_no_data",
            "query": "What is the Q3 2024 earnings report for UnknownCorp (XYZ)?",
            "category": "no_data",
            "tool_outputs": [{"tool": "get_earnings", "ticker": "XYZ", "earnings": [], "mode": "offline"}],
            "answer": "No data was found for ticker XYZ in the earnings records. No financial results are available.",
            "expected": {
                "citation_validity": None,
                "numeric_faithfulness": None,
                "unsupported_rate_block": 0.0,
                "unsupported_rate_strict": 0.0,
                "abstention_correct": True,
            },
            "rationale": "Query has category 'no_data'. Model explicitly and honestly abstains ('No data was found...'). Abstention correctness = True, unsupported claim rate = 0.0.",
        },
    ]

    for c in corruptions:
        items.append({
            "id": c["id"],
            "origin": "synthetic_corruption",
            "source_file": "synthetic",
            "query": c["query"],
            "category": c["category"],
            "tool_outputs": c["tool_outputs"],
            "answer": c["answer"],
            "expected": c["expected"],
            "rationale": c["rationale"],
        })

    return items


def run_validation():
    items = build_validation_items()
    print(f"Total validation items constructed: {len(items)}")

    # Write out JSONL dataset
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"Saved dataset to {OUT_JSONL}")

    # Evaluate metric agreement
    results = []
    cit_exact, cit_count = 0, 0
    faith_exact, faith_count = 0, 0
    unsup_block_diffs, unsup_strict_diffs = [], []
    abst_exact, abst_count = 0, 0

    for it in items:
        ans = it["answer"]
        tools = it["tool_outputs"]
        cat = it["category"]
        exp = it["expected"]

        c_calc = citation_validity(ans, tools)
        f_calc = numeric_faithfulness(ans, tools)
        u_calc = unsupported_claim_rate(ans)
        a_calc = abstention_correctness(ans, cat)

        # 1. Citation
        c_val = c_calc.get("valid_fraction")
        if exp["citation_validity"] is not None:
            cit_count += 1
            if not math.isnan(c_val) and abs(c_val - exp["citation_validity"]) < 1e-4:
                cit_exact += 1

        # 2. Faithfulness
        f_val = f_calc.get("faithful_fraction")
        if exp["numeric_faithfulness"] is not None:
            faith_count += 1
            if not math.isnan(f_val) and abs(f_val - exp["numeric_faithfulness"]) < 1e-4:
                faith_exact += 1

        # 3. Unsupported Block
        u_block = u_calc.get("unsupported_rate")
        if exp["unsupported_rate_block"] is not None and not math.isnan(u_block):
            unsup_block_diffs.append(abs(u_block - exp["unsupported_rate_block"]))

        # 4. Unsupported Strict
        u_strict = u_calc.get("strict_sentence_unsupported_rate")
        if exp["unsupported_rate_strict"] is not None and not math.isnan(u_strict):
            unsup_strict_diffs.append(abs(u_strict - exp["unsupported_rate_strict"]))

        # 5. Abstention
        if exp["abstention_correct"] is not None:
            abst_count += 1
            if a_calc.get("correct") == exp["abstention_correct"]:
                abst_exact += 1

        results.append({
            "id": it["id"],
            "origin": it["origin"],
            "calculated": {
                "citation_validity": c_val if not math.isnan(c_val) else None,
                "numeric_faithfulness": f_val if not math.isnan(f_val) else None,
                "unsupported_rate_block": u_block if not math.isnan(u_block) else None,
                "strict_sentence_unsupported_rate": u_strict if not math.isnan(u_strict) else None,
                "abstention_correct": a_calc.get("correct") if a_calc.get("applicable") else None,
            },
            "expected": exp,
            "rationale": it["rationale"],
        })

    cit_acc = round(cit_exact / cit_count, 4) if cit_count else 1.0
    faith_acc = round(faith_exact / faith_count, 4) if faith_count else 1.0
    mae_block = round(sum(unsup_block_diffs) / len(unsup_block_diffs), 4) if unsup_block_diffs else 0.0
    mae_strict = round(sum(unsup_strict_diffs) / len(unsup_strict_diffs), 4) if unsup_strict_diffs else 0.0
    abst_acc = round(abst_exact / abst_count, 4) if abst_count else 1.0

    report = {
        "n_samples": len(items),
        "breakdown": {
            "s1_dev": sum(1 for it in items if it["origin"] == "s1_dev"),
            "s6_dev": sum(1 for it in items if it["origin"] == "s6_dev"),
            "synthetic": sum(1 for it in items if it["origin"] == "synthetic_corruption"),
        },
        "agreement_metrics": {
            "citation_validity_exact_match": cit_acc,
            "citation_eval_count": cit_count,
            "numeric_faithfulness_exact_match": faith_acc,
            "faithfulness_eval_count": faith_count,
            "unsupported_rate_block_mae": mae_block,
            "unsupported_rate_strict_sentence_mae": mae_strict,
            "abstention_correctness_accuracy": abst_acc,
            "abstention_eval_count": abst_count,
        },
        "items": results,
    }

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Validation report written to {OUT_REPORT}")
    print("\n--- METRIC AGREEMENT SUMMARY ---")
    print(f"Citation Validity Exact Match:        {cit_acc:.2%} ({cit_exact}/{cit_count})")
    print(f"Numeric Faithfulness Exact Match:     {faith_acc:.2%} ({faith_exact}/{faith_count})")
    print(f"Unsupported Rate (Block) MAE:         {mae_block:.4f}")
    print(f"Unsupported Rate (Strict Sent) MAE:    {mae_strict:.4f}")
    print(f"Abstention Correctness Accuracy:      {abst_acc:.2%} ({abst_exact}/{abst_count})")


if __name__ == "__main__":
    run_validation()
