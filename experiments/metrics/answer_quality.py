"""
experiments/metrics/answer_quality.py

Evaluates answer relevance and completeness using a rubric-based LLM judge.
To avoid circular evaluation bias, the judge is run on a different model family
(llama3.1:8b) than the synthesizer (qwen2.5:7b-instruct).

Rubric dimensions (1 to 5):
  - relevance: Does the answer address the query without tangential noise?
  - completeness: Does the answer cover the needed facts, metrics, and comparisons?
  - quality_score: Arithmetic mean of (relevance + completeness) / 2.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from experiments.metrics.tool_selection import bootstrap_ci
from experiments.ollama_client import OllamaClient


_JUDGE_SYSTEM = """You are an expert financial research evaluator. Your task is to score the quality of an AI research assistant's answer on two criteria: Query Relevance (1-5) and Information Completeness (1-5).

Scoring Rubric:
- Relevance (1-5):
  1 = Completely off-topic or answers wrong company/metrics.
  2 = Weakly relevant, mentions entity but fails to address core question.
  3 = Moderately relevant, addresses question but includes notable fluff or tangents.
  4 = Relevant and focused, directly addresses the prompt.
  5 = Perfectly targeted to the specific user intent and entities requested.

- Completeness (1-5):
  1 = Missing all required financial figures or facts.
  2 = Severely deficient, mentions trends without actual numbers/data.
  3 = Partially complete, reports primary figures but misses estimates or context.
  4 = Substantially complete, covers actuals, estimates, beats/misses, or key quotes.
  5 = Comprehensive coverage of all query aspects based on evidence. On queries with no data found, honest abstention scores 5, whereas fabricating facts scores 1.

You must respond ONLY with valid JSON in this exact format:
{
  "relevance": <integer 1-5>,
  "completeness": <integer 1-5>,
  "rationale": "<1-2 concise sentences>"
}"""


def _summarize_evidence(tool_outputs: list[dict] | dict) -> str:
    """Summarize tool outputs compactly for the judge prompt."""
    if not tool_outputs:
        return "No tool records available (empty output)."
    lines = []
    if isinstance(tool_outputs, dict):
        tool_outputs = list(tool_outputs.values())

    for out in tool_outputs:
        if not isinstance(out, dict):
            continue
        tool = out.get("tool", "unknown")
        ticker = out.get("ticker", "")
        # Earnings
        for e in out.get("earnings", []):
            lines.append(
                f"- Earnings ({ticker} {e.get('period')}): Rev actual=${e.get('revenue_actual')}, "
                f"est=${e.get('revenue_estimate')}, EPS actual={e.get('eps_actual')}, est={e.get('eps_estimate')}, "
                f"Notes: {e.get('notes')}"
            )
        # Ratings
        for r in out.get("ratings", []):
            lines.append(
                f"- Rating ({ticker}): {r.get('firm')} -> {r.get('rating')}, target=${r.get('target_price')}, "
                f"action={r.get('action')}, date={r.get('date')}"
            )
        # News
        for n in out.get("results", []):
            lines.append(f"- News ({ticker}): \"{n.get('headline')}\" - {n.get('body', '')[:140]}...")

    if not lines:
        return "Tool executed with empty results."
    return "\n".join(lines[:10])


def score_single_answer(
    query: str,
    tool_outputs: list[dict],
    answer: str,
    client: OllamaClient,
) -> dict:
    """Score a single answer with the LLM judge."""
    evidence = _summarize_evidence(tool_outputs)
    user_prompt = (
        f"User Query: {query}\n\n"
        f"Retrieved Evidence:\n{evidence}\n\n"
        f"Assistant Answer:\n{answer}\n\n"
        "Provide your evaluation JSON:"
    )
    prompt = f"{_JUDGE_SYSTEM}\n\n{user_prompt}"

    resp = client.generate(prompt, num_predict=256)
    raw = resp.response.strip()

    # Parse JSON from response
    relevance, completeness, rationale = 3, 3, "Failed to parse judge output."
    try:
        # Extract markdown json block if present
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            relevance = int(data.get("relevance", 3))
            completeness = int(data.get("completeness", 3))
            rationale = str(data.get("rationale", ""))
    except Exception:
        pass

    # Constrain range
    relevance = max(1, min(5, relevance))
    completeness = max(1, min(5, completeness))
    quality = round((relevance + completeness) / 2.0, 2)

    return {
        "relevance": relevance,
        "completeness": completeness,
        "quality_score": quality,
        "rationale": rationale,
        "judge_latency_ms": resp.elapsed_ms,
    }


def evaluate_dataset_quality(
    records: list[dict],
    client: OllamaClient,
) -> dict:
    """Evaluate answer quality across an entire split of records."""
    rel_scores = []
    comp_scores = []
    qual_scores = []
    scored_records = []

    for i, rec in enumerate(records):
        qid = rec.get("id") or rec.get("query_id")
        q = rec.get("query", "")
        tools = rec.get("tool_outputs", [])
        ans = rec.get("answer", "")

        score = score_single_answer(q, tools, ans, client)
        rel_scores.append(score["relevance"])
        comp_scores.append(score["completeness"])
        qual_scores.append(score["quality_score"])

        rec_scored = dict(rec)
        rec_scored["quality_eval"] = score
        scored_records.append(rec_scored)

    def _mean(xs): return round(sum(xs) / len(xs), 4) if xs else float("nan")

    return {
        "n_evaluated": len(records),
        "mean_relevance": _mean(rel_scores),
        "relevance_ci_95": list(bootstrap_ci(rel_scores)) if rel_scores else [0, 0],
        "mean_completeness": _mean(comp_scores),
        "completeness_ci_95": list(bootstrap_ci(comp_scores)) if comp_scores else [0, 0],
        "mean_quality_score": _mean(qual_scores),
        "quality_score_ci_95": list(bootstrap_ci(qual_scores)) if qual_scores else [0, 0],
        "records": scored_records,
    }
