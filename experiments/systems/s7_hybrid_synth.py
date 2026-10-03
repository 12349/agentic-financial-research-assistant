"""
experiments/systems/s7_hybrid_synth.py

System 7: Hybrid Synthesis with Automated Verifier & Revision.
1. Rule planner selects tools (identical to S1/S6).
2. LLM generates draft free-text answer from tool records.
3. Automated verifier checks:
   - Every citation ID against flattened tool records.
   - Every number against tool records (tolerance 0.5%).
   - Inline citation coverage for every sentence.
4. If violations exist, prompts the LLM to revise once with targeted feedback.
5. If any sentence remains ungrounded after revision, it is deterministically pruned.
   If the answer becomes empty, falls back to deterministic formatter.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Optional

from agent.synthesizer import _SYNTHESIS_SYSTEM, _build_synthesis_message, _extract_sources, _synthesize_fallback
from experiments.config import NUMERIC_TOLERANCE_PCT, PRIMARY_MODEL, QUERIES_DEV, QUERIES_TEST
from experiments.metrics.synthesis_quality import (
    _flatten_tool_outputs,
    _SENTENCE_RE,
    _SOURCE_RE,
    evaluate_synthesis_quality,
    extract_all_source_numbers,
    extract_numbers_from_text,
)
from experiments.metrics.tool_selection import evaluate_tool_selection
from experiments.ollama_client import OllamaClient, OllamaResponse, synthesizer_client
from experiments.systems.common import execute_tools, load_queries, save_system_results
from planner.rule_based import plan as rule_plan


def verify_answer(
    answer: str,
    tool_outputs: list[dict],
    tol_pct: float = NUMERIC_TOLERANCE_PCT,
) -> tuple[bool, list[str], list[str]]:
    """
    Check citation IDs and numbers in `answer` against tool_outputs.
    Returns (is_valid, list_of_issues, list_of_valid_sentences).
    """
    lower = answer.lower()
    if "no data was found" in lower or "no data found" in lower:
        return True, [], [answer]

    flat_records = _flatten_tool_outputs(tool_outputs)
    src_numbers = extract_all_source_numbers(tool_outputs)
    tol = tol_pct / 100.0

    issues = []
    valid_sentences = []

    paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]

    for p in paragraphs:
        lines = [l.strip() for l in p.split("\n") if l.strip()]
        block_has_cit = any(_SOURCE_RE.search(l) for l in lines)

        for line in lines:
            line_has_cit = bool(_SOURCE_RE.search(line))
            sents = [s.strip() for s in _SENTENCE_RE.split(line) if s.strip()]

            for s in sents:
                sent_citations = _SOURCE_RE.findall(s)
                sent_nums = extract_numbers_from_text(s, strip_sources=True)
                sent_ok = True

                # Check citation existence
                if sent_citations:
                    for cid in sent_citations:
                        cid_clean = cid.strip()
                        if cid_clean not in flat_records:
                            issues.append(f"Citation [Source: {cid_clean}] does not exist in retrieved tool records.")
                            sent_ok = False
                elif not (line_has_cit or block_has_cit):
                    issues.append(f"Sentence lacks source citation: \"{s[:80]}...\"")
                    sent_ok = False

                # Check numbers
                for val, candidates in sent_nums:
                    matched = False
                    for c in candidates:
                        for sn in src_numbers:
                            if sn == 0 and abs(c) < 1e-6:
                                matched = True
                                break
                            elif sn != 0 and abs(c - sn) / abs(sn) <= tol:
                                matched = True
                                break
                        if matched:
                            break
                    if not matched:
                        issues.append(f"Number '{val}' in sentence not found in source records.")
                        sent_ok = False

                if sent_ok:
                    valid_sentences.append(s)

    is_valid = len(issues) == 0
    return is_valid, issues, valid_sentences


def synthesize_hybrid(
    query: str,
    tool_outputs: list[dict],
    client: OllamaClient,
) -> tuple[str, list[dict], float, dict]:
    """
    Generate with LLM -> Verify -> Optional Single Revision -> Prune.
    """
    t0 = time.perf_counter()

    # Step 1: Draft generation
    message = _build_synthesis_message(query, tool_outputs)
    prompt = f"{_SYNTHESIS_SYSTEM}\n\n{message}"
    resp_draft = client.generate(prompt, num_predict=1024)
    draft_text = resp_draft.response.strip()

    # Step 2: Verification
    is_valid, issues, valid_sents = verify_answer(draft_text, tool_outputs)
    revised = False
    pruned_count = 0
    final_text = draft_text

    # Step 3: Single revision if invalid
    if not is_valid and issues:
        issues_summary = "\n".join(f"- {iss}" for iss in issues[:4])
        revision_prompt = (
            f"{_SYNTHESIS_SYSTEM}\n\n{message}\n\n"
            f"Draft answer:\n{draft_text}\n\n"
            f"Verification Critique:\n{issues_summary}\n\n"
            f"Please revise the answer to fix these issues. Ensure every cited [Source: id] exists in the records, "
            f"every number matches the source records, and uncited assertions are removed."
        )
        resp_revised = client.generate(revision_prompt, num_predict=1024)
        revised_text = resp_revised.response.strip()
        revised = True

        # Re-verify revised answer
        is_valid_2, issues_2, valid_sents_2 = verify_answer(revised_text, tool_outputs)
        if is_valid_2:
            final_text = revised_text
        else:
            # Step 4: Prune remaining ungrounded sentences
            if valid_sents_2:
                final_text = " ".join(valid_sents_2)
                pruned_count = len(draft_text.split(".")) - len(valid_sents_2)
            else:
                # Fallback to deterministic formatted output
                final_text = _synthesize_fallback(query, tool_outputs)["answer"]
    else:
        final_text = draft_text

    elapsed_ms = (time.perf_counter() - t0) * 1000
    sources = _extract_sources(tool_outputs)

    # Stratum: how was the final answer produced?
    if not is_valid and issues:
        # Revision was attempted
        is_valid_2_check = len(issues_2) == 0  # already computed above
        if is_valid_2_check:
            stratum = "revised_passed"
        elif valid_sents_2 if not is_valid and issues else []:
            stratum = "pruned"
        else:
            stratum = "fell_back_to_formatter"
    else:
        stratum = "draft_passed"

    meta = {
        "revised": revised,
        "n_initial_issues": len(issues),
        "pruned_count": max(0, pruned_count),
        "synthesis_stratum": stratum,
    }
    return final_text, sources, elapsed_ms, meta


def run_s7(
    split_path: str,
    split_name: str = "test",
    limit: Optional[int] = None,
    model: str = PRIMARY_MODEL,
) -> dict:
    """Run S7 hybrid synthesis evaluation."""
    client = synthesizer_client(model=model)
    if not client.is_available():
        raise RuntimeError(f"Model '{model}' not available in Ollama.")

    queries = load_queries(split_path)
    if limit:
        queries = queries[:limit]

    records = []
    latencies = []

    print(f"--- Running S7 (Hybrid Synth with {model}) on {split_name} (n={len(queries)}) ---")

    for i, q in enumerate(queries):
        query_text = q["query"]
        qid = q.get("id", f"q_{i:03d}")
        gold_tools = q.get("gold_tools", [])
        category = q.get("category", "unknown")

        t0 = time.perf_counter()
        plan = rule_plan(query_text)
        tool_outputs, exec_ms = execute_tools(plan)
        answer_text, sources, synth_ms, meta = synthesize_hybrid(query_text, tool_outputs, client)
        t_total_ms = (time.perf_counter() - t0) * 1000

        latencies.append(t_total_ms)
        pred_tools = [c["tool"] for c in plan if "tool" in c]

        records.append({
            "id": qid,
            "query_id": qid,
            "query": query_text,
            "category": category,
            "gold_tools": gold_tools,
            "pred_tools": pred_tools,
            "plan": plan,
            "tool_outputs": tool_outputs,
            "answer": answer_text,
            "sources": sources,
            "verification_meta": meta,
            "synthesis_stratum": meta["synthesis_stratum"],
            "latency_ms": round(t_total_ms, 2),
            "tool_exec_ms": exec_ms,
            "synth_ms": round(synth_ms, 2),
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(queries):
            print(f"  Processed {i + 1}/{len(queries)} queries...")

    tool_eval = evaluate_tool_selection(records)
    synth_eval = evaluate_synthesis_quality(records)

    extra_summary = {
        "model": model,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "total_revisions": sum(1 for r in records if r["verification_meta"]["revised"]),
        "stratum_counts": {
            s: sum(1 for r in records if r["synthesis_stratum"] == s)
            for s in ["draft_passed", "revised_passed", "pruned", "fell_back_to_formatter"]
        },
    }

    system_id = f"s7_hybrid_{model.replace(':', '_')}" if model != PRIMARY_MODEL else "s7_hybrid_synth"

    summary_file = save_system_results(
        system_id=system_id,
        split_name=split_name,
        records=records,
        tool_eval=tool_eval,
        synth_eval=synth_eval,
        extra_summary=extra_summary,
    )

    print(f"\nResults saved to: {summary_file}")
    print(f"Citation Validity:           {synth_eval.citation_valid_mean:.4f} {synth_eval.citation_ci}")
    print(f"Faithfulness:                {synth_eval.faithfulness_mean:.4f} {synth_eval.faithfulness_ci}")
    print(f"Unsupported Rate (Block):    {synth_eval.unsupported_rate_mean:.4f} {synth_eval.unsupported_ci}")
    print(f"Unsupported Rate (Strict):   {synth_eval.strict_unsupported_rate_mean:.4f} {synth_eval.strict_unsupported_ci}")
    print(f"Abstention Accuracy:         {synth_eval.abstention_acc:.4f}")
    print(f"Total Revisions Triggered:   {extra_summary['total_revisions']}/{len(queries)}")

    return {
        "records": records,
        "tool_eval": tool_eval,
        "synth_eval": synth_eval,
        "summary_file": str(summary_file),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S7 Hybrid Synthesis Evaluation")
    parser.add_argument("--split", choices=["test", "dev"], default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", type=str, default=PRIMARY_MODEL)
    args = parser.parse_args()

    split_path = QUERIES_TEST if args.split == "test" else QUERIES_DEV
    run_s7(split_path, split_name=args.split, limit=args.limit, model=args.model)
