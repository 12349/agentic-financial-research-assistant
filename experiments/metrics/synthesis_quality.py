"""
experiments/metrics/synthesis_quality.py

Measures synthesis quality for the core paper claim:
deterministic formatter (S1) vs LLM synthesis (S6).

Metrics:
  1. citation_validity    — every [Source: X] in answer has X in tool_outputs
  2. numeric_faithfulness — numbers in answer match source fields ±tolerance
  3. unsupported_rate     — sentences with no citation / total sentences
  4. abstention_correct   — on no_data queries: answer contains no fabricated facts

FRAMING NOTE (from paper/audit.md):
  The formatter's citation_validity = 100% is true BY CONSTRUCTION — do not
  present it as an achievement. The real contribution is the GAP between
  formatter (structural guarantee) and LLM synthesis (instructed but unverified).
  Present numeric_faithfulness as the honest comparison metric.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from experiments.config import NUMERIC_TOLERANCE_PCT
from experiments.metrics.tool_selection import bootstrap_ci


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Matches [Source: some-id-001] or [Source: some_id]
_SOURCE_RE = re.compile(r"\[Source:\s*([^\]]+)\]", re.IGNORECASE)

# Matches numbers: integers, decimals, negative, with optional $ or %
_NUMBER_RE = re.compile(r"[-+]?\$?[\d,]+\.?\d*%?")

# Sentence splitter (simple; good enough for structured financial answers)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _clean_number(s: str) -> float | None:
    """Parse a number string like '$4.37', '43,315', '-12.5%' → float."""
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from text."""
    nums = []
    for m in _NUMBER_RE.finditer(text):
        v = _clean_number(m.group())
        if v is not None:
            nums.append(v)
    return nums


def _flatten_tool_outputs(tool_outputs: list[dict] | dict) -> dict[str, Any]:
    """
    Flatten tool_outputs into a dict of record_id → record dict.
    Handles both list-of-records and nested dicts.
    """
    flat: dict[str, Any] = {}
    if isinstance(tool_outputs, dict):
        tool_outputs = list(tool_outputs.values())
    for output in tool_outputs:
        if not isinstance(output, dict):
            continue
        # Try to extract record_id at top level
        rid = output.get("record_id") or output.get("id")
        if rid:
            flat[str(rid)] = output
        # Recurse into 'ratings', 'earnings', 'guidance', 'articles', etc.
        for val in output.values():
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        iid = item.get("record_id") or item.get("id")
                        if not iid and isinstance(item.get("source_ref"), dict):
                            iid = item["source_ref"].get("record_id")
                        if iid:
                            flat[str(iid)] = item
    return flat


def _all_numbers_from_records(records: dict[str, Any]) -> list[float]:
    """Extract all numeric field values from a flat record dict."""
    nums = []
    for rec in records.values():
        if isinstance(rec, dict):
            for v in rec.values():
                if isinstance(v, (int, float)):
                    nums.append(float(v))
                elif isinstance(v, str):
                    n = _clean_number(v)
                    if n is not None:
                        nums.append(n)
    return nums


# ---------------------------------------------------------------------------
# Per-answer metrics
# ---------------------------------------------------------------------------

def citation_validity(answer: str, tool_outputs: list[dict]) -> dict:
    """
    Check that every [Source: X] in `answer` has X in the flattened tool_outputs.

    Returns
    -------
    dict with:
      valid_fraction: float  (1.0 = all citations valid, 0.0 = none)
      n_citations:    int
      n_valid:        int
      invalid_ids:    list[str]
    """
    cited = _SOURCE_RE.findall(answer)
    cited = [c.strip() for c in cited]
    if not cited:
        return {
            "valid_fraction": float("nan"),  # nan = no citations to check
            "n_citations": 0,
            "n_valid": 0,
            "invalid_ids": [],
            "note": "No [Source: X] citations found in answer",
        }

    flat = _flatten_tool_outputs(tool_outputs)
    invalid = [c for c in cited if c not in flat]
    n_valid = len(cited) - len(invalid)
    return {
        "valid_fraction": n_valid / len(cited),
        "n_citations":    len(cited),
        "n_valid":        n_valid,
        "invalid_ids":    invalid,
    }


def numeric_faithfulness(answer: str, tool_outputs: list[dict], tol_pct: float = NUMERIC_TOLERANCE_PCT) -> dict:
    """
    For every number in `answer`, check whether it appears in any source field
    within `tol_pct`% relative tolerance.

    This is the primary metric for the synthesis comparison. The formatter
    copies fields directly (100% faithful by construction). The LLM synthesizer
    may paraphrase, round, or hallucinate numbers.

    Returns
    -------
    dict with:
      faithful_fraction: float
      n_numbers:         int
      n_faithful:        int
      unfaithful_values: list[float]
    """
    answer_nums = _extract_numbers(answer)
    if not answer_nums:
        return {
            "faithful_fraction": float("nan"),
            "n_numbers": 0,
            "n_faithful": 0,
            "unfaithful_values": [],
            "note": "No numbers found in answer",
        }

    source_nums = _all_numbers_from_records(_flatten_tool_outputs(tool_outputs))
    tol = tol_pct / 100.0

    unfaithful = []
    for v in answer_nums:
        found = False
        for sv in source_nums:
            if sv == 0:
                found = (abs(v) < 1e-9)
            else:
                found = abs(v - sv) / abs(sv) <= tol
            if found:
                break
        if not found:
            unfaithful.append(v)

    n_faithful = len(answer_nums) - len(unfaithful)
    return {
        "faithful_fraction": n_faithful / len(answer_nums),
        "n_numbers":         len(answer_nums),
        "n_faithful":        n_faithful,
        "unfaithful_values": unfaithful,
    }


def unsupported_claim_rate(answer: str) -> dict:
    """
    Fraction of sentences in `answer` that contain NO [Source: X] citation.

    A high rate indicates the synthesizer is making claims not tied to any
    source record. For the formatter this is 0 by construction (every sentence
    begins with [Source:]). For the LLM synthesizer this measures how often
    it generates unchained claims.
    """
    sentences = [s.strip() for s in _SENTENCE_RE.split(answer) if s.strip()]
    if not sentences:
        return {"unsupported_rate": float("nan"), "n_sentences": 0, "n_unsupported": 0}

    unsupported = [s for s in sentences if not _SOURCE_RE.search(s)]
    return {
        "unsupported_rate": len(unsupported) / len(sentences),
        "n_sentences":      len(sentences),
        "n_unsupported":    len(unsupported),
    }


def abstention_correctness(answer: str, category: str) -> dict:
    """
    For no_data queries: does the answer correctly abstain (say "no data",
    "not found", "I don't have", etc.) rather than fabricating an answer?

    Only meaningful when category == 'no_data'.
    """
    if category != "no_data":
        return {"applicable": False}

    # Phrases indicating honest abstention
    _ABSTAIN_PATTERNS = [
        r"no data", r"not found", r"no (information|record|results?)",
        r"(don't|do not|doesn't|does not) have",
        r"(unavailable|not available)", r"outside.*scope",
        r"(cannot|can't|unable to) (find|answer|provide)",
        r"no (earnings|ratings|guidance|news).*found",
    ]
    lower = answer.lower()
    abstained = any(re.search(p, lower) for p in _ABSTAIN_PATTERNS)

    # Penalise if it contains numeric claims that look like made-up data
    has_numbers = bool(_NUMBER_RE.search(answer))

    return {
        "applicable":  True,
        "abstained":   abstained,
        "has_numbers": has_numbers,
        # Correct = abstained AND (no suspicious numbers OR numbers are from tool_outputs)
        # We use the simple heuristic: abstained = correct for this metric
        "correct":     abstained,
    }


# ---------------------------------------------------------------------------
# Aggregate synthesis quality
# ---------------------------------------------------------------------------

@dataclass
class SynthesisQualityResult:
    n_queries:             int
    citation_valid_mean:   float          # NaN-safe mean (excluding no-citation answers)
    faithfulness_mean:     float
    unsupported_rate_mean: float
    abstention_acc:        float          # fraction correct on no_data queries

    citation_ci:     tuple[float, float]
    faithfulness_ci: tuple[float, float]
    unsupported_ci:  tuple[float, float]

    per_query: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_queries":             self.n_queries,
            "citation_valid_mean":   self.citation_valid_mean,
            "citation_ci_95":        list(self.citation_ci),
            "faithfulness_mean":     self.faithfulness_mean,
            "faithfulness_ci_95":    list(self.faithfulness_ci),
            "unsupported_rate_mean": self.unsupported_rate_mean,
            "unsupported_ci_95":     list(self.unsupported_ci),
            "abstention_acc":        self.abstention_acc,
        }


def evaluate_synthesis_quality(records: list[dict]) -> SynthesisQualityResult:
    """
    Aggregate synthesis quality metrics.

    Each record must have:
      answer:       str          — system answer
      tool_outputs: list[dict]   — raw tool output dicts
      category:     str          — query category
    """
    cit_vals, faith_vals, unsup_vals = [], [], []
    abst_total, abst_correct = 0, 0
    per_query_rows = []

    for rec in records:
        answer   = rec.get("answer", "")
        outputs  = rec.get("tool_outputs", [])
        category = rec.get("category", "unknown")

        cit  = citation_validity(answer, outputs)
        faith = numeric_faithfulness(answer, outputs)
        unsup = unsupported_claim_rate(answer)
        abst  = abstention_correctness(answer, category)

        # NaN-safe accumulation
        if not math.isnan(cit.get("valid_fraction", float("nan"))):
            cit_vals.append(cit["valid_fraction"])
        if not math.isnan(faith.get("faithful_fraction", float("nan"))):
            faith_vals.append(faith["faithful_fraction"])
        if not math.isnan(unsup.get("unsupported_rate", float("nan"))):
            unsup_vals.append(unsup["unsupported_rate"])

        if abst.get("applicable"):
            abst_total += 1
            if abst.get("correct"):
                abst_correct += 1

        per_query_rows.append({
            "query_id": rec.get("query_id", ""),
            "category": category,
            "citation":    cit,
            "faithfulness": faith,
            "unsupported": unsup,
            "abstention":  abst,
        })

    def _mean(xs): return round(sum(xs) / len(xs), 4) if xs else float("nan")

    return SynthesisQualityResult(
        n_queries             = len(records),
        citation_valid_mean   = _mean(cit_vals),
        faithfulness_mean     = _mean(faith_vals),
        unsupported_rate_mean = _mean(unsup_vals),
        abstention_acc        = (round(abst_correct / abst_total, 4)
                                 if abst_total > 0 else float("nan")),
        citation_ci           = bootstrap_ci(cit_vals)   if cit_vals   else (float("nan"), float("nan")),
        faithfulness_ci       = bootstrap_ci(faith_vals) if faith_vals else (float("nan"), float("nan")),
        unsupported_ci        = bootstrap_ci(unsup_vals) if unsup_vals else (float("nan"), float("nan")),
        per_query             = per_query_rows,
    )
