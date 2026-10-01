# Gold Label Agreement Report (Revised)

## Methodology

Two independent labeling passes were performed on the same 15% random
sample (n=34) drawn with `random.Random(999)` from the full 229-query set.

| | Rater 1 (Primary) | Rater 2 (Independent) |
|---|---|---|
| **Approach** | Semantic/intent-based | Information-needs analysis |
| **Key principle** | "What communicative intent does this query express?" | "What INFORMATION does the user need, and which tool(s) provide it?" |
| **Category logic** | Derived from intended tool routing | Derived from the count of minimally necessary tools |
| **Fact sourcing** | Traced from query intent to fixture records | Traced from information need to fixture records |

Both raters had access to the same rubric: tool descriptions, category definitions,
and fixture data. The second pass was performed independently, re-analyzing each
query from scratch with the information-needs framework to reduce correlated error.

> [!NOTE]
> Key rubric decisions applied by Rater 2:
> - "consensus estimate" for EPS/revenue → available in earnings fixture (`eps_estimate`, `revenue_estimate`), does NOT require `get_ratings`
> - "analyst forecasts" in guidance context → `analyst_estimate_prior` field in guidance fixture, does NOT require `get_ratings`
> - "what happened at [event]" → narrative question requiring news; structured data may supplement
> - "the Street" → context-dependent: "Street estimate" = estimate fields in earnings/guidance; "Street thinking" = analyst ratings

## Agreement Scores

| Metric | Value |
|--------|-------|
| **Tool-set exact-match %** | 97.1% |
| **Tool-set Cohen's κ** | 0.9671 |
| **Category exact-match %** | 94.1% |
| **Category Cohen's κ** | 0.9183 |
| **Avg fact-overlap ratio (Jaccard)** | 0.976 |

### Interpretation Guide

| κ Range | Interpretation |
|---------|----------------|
| > 0.80 | Almost perfect agreement |
| 0.61–0.80 | Substantial agreement |
| 0.41–0.60 | Moderate agreement |
| 0.21–0.40 | Fair agreement |
| < 0.20 | Slight agreement |

## Disagreement Cases (2 of 34)

| # | ID | Query | R1 Tools | R2 Tools | Tools | R1 Cat | R2 Cat | Cat | Fact Overlap |
|---|-----|-------|----------|----------|-------|--------|--------|-----|--------------|
| 1 | N003 | What happened at JPMorgan's latest earnings announcemen… | search_news | get_earnings, search_news | ❌ | single_tool | dual_tool | ❌ | 0.400 |
| 2 | A013 | What's the Street thinking about NVDA stock? | get_ratings, search_news | get_ratings, search_news | ✅ | ambiguous | dual_tool | ❌ | 1.000 |

## Detailed Disagreement Analysis

### Case 1: N003

**Query**: "What happened at JPMorgan's latest earnings announcement?"

| | Rater 1 | Rater 2 |
|---|---------|---------|
| Tools | `['search_news']` | `['get_earnings', 'search_news']` |
| Category | `single_tool` | `dual_tool` |
| Facts | `['news-jpm-001', 'net income of $12.9 billion']…` | `['news-jpm-001', 'net income of $12.9 billion', 'earnings-jpm-001']…` |
| Fact overlap | 0.400 | |

> **Diagnosis**: Rater 1 labeled `search_news` only (narrative question about an event).
> Rater 2 labeled `get_earnings` + `search_news` (the event IS an earnings announcement;
> a complete answer requires both the narrative AND the structured financial data).
> **Recommendation**: Rater 2's label is more complete — see §Flagged Cases below.

### Case 2: A013

**Query**: "What's the Street thinking about NVDA stock?"

| | Rater 1 | Rater 2 |
|---|---------|---------|
| Tools | `['get_ratings', 'search_news']` | `['get_ratings', 'search_news']` |
| Category | `ambiguous` | `dual_tool` |
| Facts | `['rating-nvda-001', 'rating-nvda-002', 'news-nvda-003']…` | `['rating-nvda-001', 'rating-nvda-002', 'news-nvda-003']…` |
| Fact overlap | 1.000 | |

> **Diagnosis**: Rater 1 labeled `ambiguous` (ratings + news, ≥2 tools but vague intent).
> Rater 2 labeled `dual_tool` ("the Street thinking" = analyst ratings + news context).
> The disagreement is on whether 2 tools constitutes `ambiguous` or `dual_tool`.
> Per the rubric, `ambiguous` requires ≥3 tools OR genuinely unclear intent;
> this query has a focused intent (Street sentiment) answerable with 2 tools.
> **Recommendation**: Rater 2's `dual_tool` is more precise.

---

## Flagged Cases: E014 and N003

### E014: "What was Nvidia's Q3 FY2025 EPS and how did it compare with the consensus estimate?"

| Field | Value |
|-------|-------|
| **R1 (Primary) tools** | `['get_earnings']` |
| **R2 tools** | `['get_earnings']` |
| **Agreement** | ✅ Both raters agree |

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
| **R1 (Primary) tools** | `['search_news']` |
| **R2 tools** | `['get_earnings', 'search_news']` |
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

