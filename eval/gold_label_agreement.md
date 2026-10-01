# Gold Label Self-Consistency Report

> **Honest scope**: Both annotation passes were performed by the same author.
> This document reports **within-author labeling consistency**, not independent
> inter-rater reliability (IRR). A genuinely independent rater (different person
> or different model family with documented prompts) is required before claiming
> IRR for publication. Phase 2 will add that.

## Methodology

Two annotation passes were performed on the same 15% random sample (n=34)
drawn with `random.Random(999)` from the full 229-query set.

| | Pass 1 (Primary) | Pass 2 (Re-annotation) |
|---|---|---|
| **Rater** | Same author | Same author (different analytical frame) |
| **Approach** | Semantic/intent-based | Information-needs analysis |
| **Key principle** | "What communicative intent does this query express?" | "What INFORMATION does the user need, and which tool(s) provide it?" |
| **Category logic** | Derived from intended tool routing | Derived from count of minimally necessary tools |
| **Fact sourcing** | Traced from query intent to fixture records | Traced from information need to fixture records |

Both passes had access to the same rubric: tool descriptions, category definitions,
and fixture data. The second pass re-analyzed each query from scratch using the
information-needs framework to surface framing-driven inconsistencies.

> [!NOTE]
> Key rubric decisions applied in Pass 2:
> - "consensus estimate" for EPS/revenue → available in earnings fixture (`eps_estimate`, `revenue_estimate`), does NOT require `get_ratings`
> - "analyst forecasts" in guidance context → `analyst_estimate_prior` field in guidance fixture, does NOT require `get_ratings`
> - "what happened at [event]" → narrative question requiring news; structured data may supplement
> - "the Street" → context-dependent: "Street estimate" = estimate fields in earnings/guidance; "Street thinking" = analyst ratings

## Self-Consistency Scores

| Metric | Value | 95% CI (bootstrap, B=10,000, n=34) |
|--------|-------|-------------------------------------|
| **Tool-set exact-match %** | 97.1% | — |
| **Tool-set Cohen's κ** | 0.9671 | [0.90, 1.00] |
| **Category exact-match %** | 94.1% | — |
| **Category Cohen's κ** | 0.9183 | [0.80, 1.00] |
| **Avg fact-overlap ratio (Jaccard)** | 0.976 | — |

> **Note on CIs**: upper bounds hit 1.00 due to the discrete distribution artifact
> at near-ceiling agreement with n=34. The bootstrap used agreement-vector resampling;
> analytical Fleiss SE gives Tool-set 95% CI [0.90, 1.03] and Category [0.81, 1.03]
> (capped at 1.0). Both methods are consistent. The wide CIs appropriately reflect
> the small sample size.

> **What these numbers mean**: High self-consistency confirms the labeling rubric
> is unambiguous for the vast majority of query types. It does NOT confirm that
> the labels are correct relative to an external standard.

### κ Interpretation Scale

| κ Range | Interpretation |
|---------|----------------|
| > 0.80 | Almost perfect |
| 0.61–0.80 | Substantial |
| 0.41–0.60 | Moderate |
| 0.21–0.40 | Fair |
| < 0.20 | Slight |

## Disagreements Between Passes (2 of 34)

| # | ID | Query | R1 Tools | R2 Tools | Tools | R1 Cat | R2 Cat | Cat | Fact Overlap |
|---|-----|-------|----------|----------|-------|--------|--------|-----|--------------|
| 1 | N003 | What happened at JPMorgan's latest earnings announcemen… | search_news | get_earnings, search_news | ❌ | single_tool | dual_tool | ❌ | 0.400 |
| 2 | A013 | What's the Street thinking about NVDA stock? | get_ratings, search_news | get_ratings, search_news | ✅ | ambiguous | dual_tool | ❌ | 1.000 |

## Detailed Disagreement Analysis

### Case 1: N003

**Query**: "What happened at JPMorgan's latest earnings announcement?"

| | Pass 1 | Pass 2 |
|---|---------|---------|
| Tools | `['search_news']` | `['get_earnings', 'search_news']` |
| Category | `single_tool` | `dual_tool` |
| Facts | `['news-jpm-001', 'net income of $12.9 billion']…` | `['news-jpm-001', 'net income of $12.9 billion', 'earnings-jpm-001']…` |
| Fact overlap | 0.400 | |

> **Diagnosis**: Pass 1 labeled `search_news` only (narrative question about an event).
> Pass 2 labeled `get_earnings` + `search_news` (the event IS an earnings announcement;
> a complete answer requires both the narrative AND the structured financial data).
> **Recommendation**: Pass 2's label is more complete — see §Flagged Cases below.

### Case 2: A013

**Query**: "What's the Street thinking about NVDA stock?"

| | Pass 1 | Pass 2 |
|---|---------|---------|
| Tools | `['get_ratings', 'search_news']` | `['get_ratings', 'search_news']` |
| Category | `ambiguous` | `dual_tool` |
| Facts | `['rating-nvda-001', 'rating-nvda-002', 'news-nvda-003']…` | `['rating-nvda-001', 'rating-nvda-002', 'news-nvda-003']…` |
| Fact overlap | 1.000 | |

> **Diagnosis**: Pass 1 labeled `ambiguous` (ratings + news, ≥2 tools but vague intent).
> Pass 2 labeled `dual_tool` ("the Street thinking" = analyst ratings + news context).
> The disagreement is on whether 2 tools constitutes `ambiguous` or `dual_tool`.
> Per the rubric, `ambiguous` requires ≥3 tools OR genuinely unclear intent;
> this query has a focused intent (Street sentiment) answerable with 2 tools.
> **Recommendation**: Pass 2's `dual_tool` is more precise.

---

## Flagged Cases: E014 and N003

### E014: "What was Nvidia's Q3 FY2025 EPS and how did it compare with the consensus estimate?"

| Field | Value |
|-------|-------|
| **Pass 1 tools** | `['get_earnings']` |
| **Pass 2 tools** | `['get_earnings']` |
| **Agreement** | ✅ Both passes agree |

**Analysis**: The "consensus estimate" here refers to the analyst consensus EPS forecast,
which is stored as `eps_estimate=0.74` in `earnings-nvda-001`. The earnings fixture contains
both `eps_actual` and `eps_estimate` plus `eps_surprise_pct`, making `get_earnings` alone
sufficient to fully answer this query. `get_ratings` provides analyst *ratings* and *price
targets*, NOT EPS estimates — these are different data. No correction needed.

> [!TIP]
> **Verdict: Primary gold label CONFIRMED correct.** No revision needed for E014.

### N003: "What happened at JPMorgan's latest earnings announcement?"

| Field | Value |
|-------|-------|
| **Pass 1 tools** | `['search_news']` |
| **Pass 2 tools** | `['get_earnings', 'search_news']` |
| **Agreement** | ❌ Disagree |

**Analysis**: The query asks "what happened at" an "earnings announcement." Two considerations:

1. **Narrative framing** ("what happened"): The user is asking for a story/summary of the
   event → `search_news` provides this via `news-jpm-001`.
2. **Specific data reference** ("earnings announcement"): The event IS an earnings release.
   A complete, grounded answer should cite the actual financial data (revenue=$43.3B,
   EPS=$4.37, beat on both) that the news is reporting on. `get_earnings` provides these
   definitive numbers with source references.

While `search_news` alone surfaces a news article that *mentions* some numbers, the
structured earnings data provides the authoritative actuals-vs-estimates comparison.
For conference-paper grading, a correct answer that only uses news would be citing
secondary reporting rather than the primary source.

> [!WARNING]
> **Verdict: Primary gold label REVISED.** N003 changed from `single_tool` → `dual_tool`,
> `gold_tools` from `['search_news']` → `['get_earnings', 'search_news']`,
> `gold_facts` expanded to include `earnings-jpm-001` data.

---

## Gold Label Corrections Applied

| ID | Field | Before | After | Rationale |
|-----|-------|--------|-------|-----------|
| N003 | category | `single_tool` | `dual_tool` | Earnings announcement requires both narrative + structured data |
| N003 | gold_tools | `['search_news']` | `['get_earnings', 'search_news']` | Complete answer needs authoritative actuals |
| N003 | gold_facts | `['news-jpm-001', …]` | + `'earnings-jpm-001'`, `'revenue_actual=43315000000'`, `'eps_actual=4.37'` | Primary source data |
| A013 | category | `ambiguous` | `dual_tool` | 2 tools with focused intent = dual_tool, not ambiguous per rubric |
| E014 | — | (no change) | (no change) | Consensus estimate available in earnings fixture |

### Post-Correction Distribution

| Category | Count | % |
|----------|-------|---|
| single_tool | 82 | 35.8% |
| dual_tool | 82 | 35.8% |
| ambiguous | 32 | 14.0% |
| no_data | 33 | 14.4% |

