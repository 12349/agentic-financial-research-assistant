"""
experiments/metrics/synthesis_quality.py

Measures synthesis quality for comparing deterministic formatting (S1)
and LLM synthesis (S6).

Metrics:
  1. citation_validity    — fraction of cited [Source: X] tags where X exists in tool_outputs
  2. numeric_faithfulness — fraction of numbers in answer that match numbers in source records
  3. unsupported_rate     — fraction of sentences not governed by a source citation
  4. abstention_correct   — on no_data queries: does the answer honestly abstain?
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from experiments.config import NUMERIC_TOLERANCE_PCT
from experiments.metrics.tool_selection import bootstrap_ci

# Matches [Source: some-id-001] or [Source: some_id]
_SOURCE_RE = re.compile(r"\[Source:\s*([^\]]+)\]", re.IGNORECASE)

# Matches numbers with optional unit suffix: e.g. $35.1B, 462,890, 6.0%, -12.5
_NUM_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])([-+]?\$?[\d,]+\.?\d*)\s*(billion|million|thousand|[BMK%])?",
    re.IGNORECASE,
)

# Sentence splitter
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def extract_numbers_from_text(text: str, strip_sources: bool = True) -> list[tuple[float, list[float]]]:
    """
    Extract numbers from text.
    If strip_sources is True, strips [Source: ...] tags first so record IDs like
    'rating-tsla-001' are not erroneously parsed as negative numbers like '-1.0'.
    Returns list of (display_val, candidate_vals).
    """
    target = _SOURCE_RE.sub(" ", text) if strip_sources else text
    results = []

    for m in _NUM_PATTERN.finditer(target):
        raw = m.group(1).replace("$", "").replace(",", "").strip()
        unit = (m.group(2) or "").upper()
        if not raw or raw in ("-", "+"):
            continue
        try:
            val = float(raw)
        except ValueError:
            continue

        candidates = [val]
        if unit in ("BILLION", "B"):
            candidates.append(val * 1e9)
        elif unit in ("MILLION", "M"):
            candidates.append(val * 1e6)
        elif unit in ("THOUSAND", "K"):
            candidates.append(val * 1e3)
        elif unit == "%":
            candidates.append(val / 100.0)

        results.append((val, candidates))

    return results


def extract_all_source_numbers(tool_outputs: list[dict] | dict) -> set[float]:
    """
    Recursively extract all numeric values from tool_outputs, including
    scaled representations (billions, millions), percentages, and numbers
    within text fields (headlines, bodies, notes, summaries).
    """
    numbers: set[float] = set()
    SKIP_KEYS = {
        "record_id", "id", "ticker", "symbol", "status", "query", "tool",
        "source_type", "doc_id", "author", "source", "source_ref",
    }

    def _add_num(v: float):
        numbers.add(round(v, 4))
        numbers.add(round(v, 2))
        numbers.add(round(v, 1))
        try:
            numbers.add(float(f"{v:.1f}"))
            numbers.add(float(f"{v:.2f}"))
        except Exception:
            pass
        if abs(v) >= 1e6:
            numbers.add(round(v / 1e9, 1))
            numbers.add(round(v / 1e9, 2))
            numbers.add(round(v / 1e9, 3))
            numbers.add(round(v / 1e6, 1))
            numbers.add(round(v / 1e6, 2))
        elif 0 < abs(v) < 1.0:
            numbers.add(round(v * 100.0, 1))
            numbers.add(round(v * 100.0, 2))

    def _walk(obj, key=None):
        if key in SKIP_KEYS:
            return
        if isinstance(obj, (int, float)):
            _add_num(float(obj))
        elif isinstance(obj, str):
            extracted = extract_numbers_from_text(obj, strip_sources=True)
            for _, candidates in extracted:
                for c in candidates:
                    _add_num(c)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, key=k)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item, key=key)

    _walk(tool_outputs)
    return numbers


def _flatten_tool_outputs(tool_outputs: list[dict] | dict) -> dict[str, Any]:
    """Flatten tool_outputs into a dict of record_id -> record dict."""
    flat: dict[str, Any] = {}
    if isinstance(tool_outputs, dict):
        tool_outputs = list(tool_outputs.values())
    for output in tool_outputs:
        if not isinstance(output, dict):
            continue
        rid = output.get("record_id") or output.get("id")
        if rid:
            flat[str(rid)] = output
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


# ---------------------------------------------------------------------------
# Per-answer metrics
# ---------------------------------------------------------------------------

def citation_validity(answer: str, tool_outputs: list[dict]) -> dict:
    """Check that every [Source: X] in `answer` has X in flattened tool_outputs."""
    cited = _SOURCE_RE.findall(answer)
    cited = [c.strip() for c in cited]
    if not cited:
        return {
            "valid_fraction": float("nan"),
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


def numeric_faithfulness(
    answer: str,
    tool_outputs: list[dict],
    tol_pct: float = NUMERIC_TOLERANCE_PCT,
) -> dict:
    """
    Check whether every number in `answer` matches a number in the tool outputs
    within `tol_pct`% relative tolerance.
    """
    ans_nums = extract_numbers_from_text(answer, strip_sources=True)
    if not ans_nums:
        return {
            "faithful_fraction": float("nan"),
            "n_numbers": 0,
            "n_faithful": 0,
            "unfaithful_values": [],
            "note": "No numbers found in answer",
        }

    src_nums = extract_all_source_numbers(tool_outputs)
    tol = tol_pct / 100.0

    unfaithful = []
    faithful_count = 0

    for val, candidates in ans_nums:
        match = False
        for c in candidates:
            for s in src_nums:
                if s == 0:
                    if abs(c) < 1e-6:
                        match = True
                        break
                elif abs(c - s) / abs(s) <= tol:
                    match = True
                    break
            if match:
                break
        if match:
            faithful_count += 1
        else:
            unfaithful.append(val)

    return {
        "faithful_fraction": faithful_count / len(ans_nums),
        "n_numbers":         len(ans_nums),
        "n_faithful":        faithful_count,
        "unfaithful_values": unfaithful,
    }


def unsupported_claim_rate(answer: str) -> dict:
    """
    Fraction of sentences in `answer` that are not governed by a citation.
    A sentence is supported if:
      (1) It contains [Source: ...]; OR
      (2) It belongs to a cited block/paragraph that starts with or contains [Source: ...].
    Answers that explicitly abstain ('No data was found...') contain no factual claims
    and are assigned unsupported_rate = 0.0.
    """
    lower = answer.lower()
    if "no data was found" in lower or "no data found" in lower:
        return {"unsupported_rate": 0.0, "n_sentences": 1, "n_unsupported": 0}

    paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]
    total_sentences = 0
    supported_sentences = 0
    strict_inline_count = 0

    for p in paragraphs:
        lines = [l.strip() for l in p.split("\n") if l.strip()]
        block_has_citation = any(_SOURCE_RE.search(l) for l in lines)

        for line in lines:
            line_has_citation = bool(_SOURCE_RE.search(line))
            sents = [s.strip() for s in _SENTENCE_RE.split(line) if s.strip()]
            for s in sents:
                total_sentences += 1
                if line_has_citation:
                    strict_inline_count += 1
                if line_has_citation or block_has_citation:
                    supported_sentences += 1

    if not total_sentences:
        return {"unsupported_rate": float("nan"), "n_sentences": 0, "n_unsupported": 0}

    unsupported = total_sentences - supported_sentences
    return {
        "unsupported_rate": round(unsupported / total_sentences, 4),
        "strict_inline_unsupported_rate": round((total_sentences - strict_inline_count) / total_sentences, 4),
        "n_sentences": total_sentences,
        "n_unsupported": unsupported,
    }


def abstention_correctness(answer: str, category: str) -> dict:
    """
    For no_data queries: does the answer correctly abstain rather than fabricating
    or returning unrelated retrieved content?
    """
    if category != "no_data":
        return {"applicable": False}

    _ABSTAIN_PATTERNS = [
        r"no data", r"not found", r"no (information|record|results?)",
        r"(don't|do not|doesn't|does not) have",
        r"(unavailable|not available)", r"outside.*scope",
        r"(cannot|can't|unable to) (find|answer|provide)",
        r"no (earnings|ratings|guidance|news).*found",
        r"no .* data was available",
    ]
    lower = answer.lower()
    abstained = any(re.search(p, lower) for p in _ABSTAIN_PATTERNS)

    return {
        "applicable": True,
        "abstained":  abstained,
        "correct":    abstained,
    }


# ---------------------------------------------------------------------------
# Aggregate synthesis quality
# ---------------------------------------------------------------------------

@dataclass
class SynthesisQualityResult:
    n_queries:             int
    citation_valid_mean:   float
    faithfulness_mean:     float
    unsupported_rate_mean: float
    abstention_acc:        float

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
    """Aggregate synthesis quality metrics across queries."""
    cit_vals, faith_vals, unsup_vals = [], [], []
    abst_total, abst_correct = 0, 0
    per_query_rows = []

    for rec in records:
        answer   = rec.get("answer", "")
        outputs  = rec.get("tool_outputs", [])
        category = rec.get("category", "unknown")

        cit   = citation_validity(answer, outputs)
        faith = numeric_faithfulness(answer, outputs)
        unsup = unsupported_claim_rate(answer)
        abst  = abstention_correctness(answer, category)

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
            "query_id":     rec.get("query_id") or rec.get("id", ""),
            "category":     category,
            "citation":     cit,
            "faithfulness": faith,
            "unsupported":  unsup,
            "abstention":   abst,
        })

    def _mean(xs): return round(sum(xs) / len(xs), 4) if xs else float("nan")

    return SynthesisQualityResult(
        n_queries             = len(records),
        citation_valid_mean   = _mean(cit_vals),
        faithfulness_mean     = _mean(faith_vals),
        unsupported_rate_mean = _mean(unsup_vals),
        abstention_acc        = round(abst_correct / abst_total, 4) if abst_total > 0 else float("nan"),
        citation_ci           = bootstrap_ci(cit_vals)   if cit_vals   else (float("nan"), float("nan")),
        faithfulness_ci       = bootstrap_ci(faith_vals) if faith_vals else (float("nan"), float("nan")),
        unsupported_ci        = bootstrap_ci(unsup_vals) if unsup_vals else (float("nan"), float("nan")),
        per_query             = per_query_rows,
    )
