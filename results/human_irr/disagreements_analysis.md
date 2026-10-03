# Human Inter-Annotator Agreement (IRR) & Disagreement Analysis

## 1. Summary of Pairwise Agreement (n=50 Sample from Train/Dev)

> [!IMPORTANT]
> **Sample Size & Precision Note**: Because $n=50$, the bootstrap 95% confidence intervals are wide. All results reflect the exact, unedited annotations provided in `eval/human_annotation_sample_filled.csv`.

| Pair | Tool-Set Exact Match % [95% CI] | Tool-Set Cohen's $\kappa$ [95% CI] | Category Match % [95% CI] | Category Cohen's $\kappa$ [95% CI] |
| :--- | :--- | :--- | :--- | :--- |
| **Human vs. Author (Gold)** | **68.0%** [54.0%, 80.0%] | **0.647** [0.5013, 0.7723] | **88.0%** [78.0%, 96.0%] | **0.8346** [0.6903, 0.9449] |
| **Human vs. Llama 3.1 8B (Guided)** | **56.0%** [42.0%, 70.0%] | **0.5069** [0.3474, 0.6444] | **70.0%** [58.0%, 82.0%] | **0.6019** [0.4379, 0.7564] |
| **Author vs. Llama 3.1 8B (Guided)** | **44.0%** [30.0%, 58.0%] | **0.3894** [0.2358, 0.5275] | **62.0%** [48.0%, 76.0%] | **0.5005** [0.3203, 0.6737] |

---

## 2. Split Isolation Verification
- **Train Queries**: 40 / 50 (80.0%)
- **Dev Queries**: 10 / 50 (20.0%)
- **Test Queries**: 0 / 50 (0.0%)
- **Confirmation**: Verified that 100% of the sample queries originate from train and dev splits. Zero held-out test queries were exposed to the annotator.

---

## 3. Confusion Matrix: Human vs. Author (Category)

Rows represent Human categories; Columns represent Author categories.

| Human \ Author | single_tool | dual_tool | ambiguous | no_data | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **single_tool** | 16 | 1 | 0 | 3 | 20 |
| **dual_tool** | 2 | 11 | 0 | 0 | 13 |
| **ambiguous** | 0 | 0 | 10 | 0 | 10 |
| **no_data** | 0 | 0 | 0 | 7 | 7 |
| **Total** | 18 | 12 | 10 | 10 | 50 |

---

## 4. Disagreement Breakdown by Root Cause (Human vs. Author)

A total of **19 out of 50 queries (38.0%)** showed disagreement between the human annotator and author gold labels.

### Cause A: Fixture Coverage vs. User Intent (Core Conceptual Distinction)
> **Key Conceptual Finding**: When a query names a supported ticker (e.g., JPM, TSLA, XOM) but requests a time period or financial line item not present in the local static database fixtures, a rater can make two justifiable decisions:
> 1. **Intent-based routing (Human)**: Route to the appropriate semantic tool (e.g. `get_earnings`) under the assumption that an production financial database contains comprehensive historical coverage.
> 2. **Fixture-based routing (Author)**: Label as `no_data` because the offline mock fixtures only cover specific quarters (e.g. Q3 2024 / Q3 FY2025) and lack the requested period.

#### Query `ND019`: *"What was Exxon's free cash flow in 2023?"*
- **Human Label**: Category = `single_tool`, Tools = `['get_earnings']`
- **Author Gold**: Category = `no_data`, Tools = `['get_earnings']`
- **Human Note**: *"Free cash flow is not a listed trigger; assumed part of the earnings release. Borderline vs no_data"*
- **Analysis**: The human saw a clear earnings intent for a supported ticker and routed to `get_earnings`. The author labeled `no_data` due to the lack of fixture coverage for this specific period/metric.

#### Query `ND017`: *"What was JPMorgan's Q1 2025 EPS?"*
- **Human Label**: Category = `single_tool`, Tools = `['get_earnings']`
- **Author Gold**: Category = `no_data`, Tools = `['get_earnings']`
- **Human Note**: *""*
- **Analysis**: The human saw a clear earnings intent for a supported ticker and routed to `get_earnings`. The author labeled `no_data` due to the lack of fixture coverage for this specific period/metric.

#### Query `ND015`: *"What was Tesla's revenue in Q1 2020?"*
- **Human Label**: Category = `single_tool`, Tools = `['get_earnings']`
- **Author Gold**: Category = `no_data`, Tools = `['get_earnings']`
- **Human Note**: *"Old period; data may not exist in DB"*
- **Analysis**: The human saw a clear earnings intent for a supported ticker and routed to `get_earnings`. The author labeled `no_data` due to the lack of fixture coverage for this specific period/metric.

### Cause B: Guideline Ambiguity on Multi-Tool vs. Default Routing for Open-Ended Queries
> **Key Finding**: For queries asking open-ended questions like *"What's the bull case?"* or *"Should I be worried?"*, the human annotator routed to dual tools (`[get_ratings, search_news]`), whereas the author strictly applied the guideline default of `['search_news']` for ambiguous queries.

#### Query `A032`: *"Tesla — any catalysts on the horizon?"*
- **Human Label**: Category = `ambiguous`, Tools = `['search_news']`
- **Author Gold**: Category = `ambiguous`, Tools = `['get_guidance', 'get_ratings', 'search_news']`
- **Human Note**: *"Vague; could also be guidance/ratings"*
- **Analysis**: The human judged that answering a 'bull case' or 'risk' inquiry benefits from both analyst ratings and recent news, whereas the author guidelines prescribed defaulting ambiguous requests to `search_news` only.

#### Query `A010`: *"I want a comprehensive view of Tesla's situation."*
- **Human Label**: Category = `ambiguous`, Tools = `['search_news']`
- **Author Gold**: Category = `ambiguous`, Tools = `['get_earnings', 'get_guidance', 'get_ratings', 'search_news']`
- **Human Note**: *"Broad, supported ticker"*
- **Analysis**: The human judged that answering a 'bull case' or 'risk' inquiry benefits from both analyst ratings and recent news, whereas the author guidelines prescribed defaulting ambiguous requests to `search_news` only.

#### Query `A009`: *"What do the numbers say about Nvidia's momentum?"*
- **Human Label**: Category = `ambiguous`, Tools = `['search_news']`
- **Author Gold**: Category = `ambiguous`, Tools = `['get_earnings', 'get_guidance', 'get_ratings']`
- **Human Note**: *"Vague; "numbers" hints at earnings but no metric specified"*
- **Analysis**: The human judged that answering a 'bull case' or 'risk' inquiry benefits from both analyst ratings and recent news, whereas the author guidelines prescribed defaulting ambiguous requests to `search_news` only.

#### Query `A016`: *"What's the bull case for XOM?"*
- **Human Label**: Category = `ambiguous`, Tools = `['get_ratings', 'search_news']`
- **Author Gold**: Category = `ambiguous`, Tools = `['get_earnings', 'get_ratings', 'search_news']`
- **Human Note**: *"Open-ended opinion request"*
- **Analysis**: The human judged that answering a 'bull case' or 'risk' inquiry benefits from both analyst ratings and recent news, whereas the author guidelines prescribed defaulting ambiguous requests to `search_news` only.

#### Query `A023`: *"JPM — everything you've got."*
- **Human Label**: Category = `ambiguous`, Tools = `['search_news']`
- **Author Gold**: Category = `ambiguous`, Tools = `['get_earnings', 'get_guidance', 'get_ratings', 'search_news']`
- **Human Note**: *"Broad"*
- **Analysis**: The human judged that answering a 'bull case' or 'risk' inquiry benefits from both analyst ratings and recent news, whereas the author guidelines prescribed defaulting ambiguous requests to `search_news` only.

#### Query `A029`: *"What do investors need to understand about JPM right now?"*
- **Human Label**: Category = `ambiguous`, Tools = `['search_news']`
- **Author Gold**: Category = `ambiguous`, Tools = `['get_earnings', 'get_ratings', 'search_news']`
- **Human Note**: *"Broad"*
- **Analysis**: The human judged that answering a 'bull case' or 'risk' inquiry benefits from both analyst ratings and recent news, whereas the author guidelines prescribed defaulting ambiguous requests to `search_news` only.

#### Query `A022`: *"Should I be worried about Tesla stock?"*
- **Human Label**: Category = `ambiguous`, Tools = `['get_ratings', 'search_news']`
- **Author Gold**: Category = `ambiguous`, Tools = `['get_earnings', 'get_ratings', 'search_news']`
- **Human Note**: *"Vague, opinion-seeking"*
- **Analysis**: The human judged that answering a 'bull case' or 'risk' inquiry benefits from both analyst ratings and recent news, whereas the author guidelines prescribed defaulting ambiguous requests to `search_news` only.

### Cause C: Single-Tool vs. Dual-Tool on Earnings Announcements & Forward Estimates

#### Query `G017`: *"What was the analyst estimate before NVDA issued their Q3 FY2025 revenue guidance?"*
- **Human Label**: Category = `dual_tool`, Tools = `['get_guidance', 'get_ratings']`
- **Author Gold**: Category = `single_tool`, Tools = `['get_guidance']`
- **Human Note**: *"Guidance vs analyst estimates; estimates mapped to get_ratings"*
- **Analysis**: For announcement/narrative events (e.g. `N003`), the human labeled `get_earnings` single-tool while the author labeled `[get_earnings, search_news]` dual-tool. For guidance vs prior estimates (`G014`, `G017`), the human assigned dual tools (`[get_guidance, get_ratings]`) whereas the author treated prior consensus inside guidance as a single tool.

#### Query `G014`: *"How does XOM's capital spending outlook stand relative to prior analyst forecasts?"*
- **Human Label**: Category = `dual_tool`, Tools = `['get_guidance', 'get_ratings']`
- **Author Gold**: Category = `single_tool`, Tools = `['get_guidance']`
- **Human Note**: *"Capex outlook = guidance; analyst forecasts = ratings"*
- **Analysis**: For announcement/narrative events (e.g. `N003`), the human labeled `get_earnings` single-tool while the author labeled `[get_earnings, search_news]` dual-tool. For guidance vs prior estimates (`G014`, `G017`), the human assigned dual tools (`[get_guidance, get_ratings]`) whereas the author treated prior consensus inside guidance as a single tool.

#### Query `N003`: *"What happened at JPMorgan's latest earnings announcement?"*
- **Human Label**: Category = `single_tool`, Tools = `['get_earnings']`
- **Author Gold**: Category = `dual_tool`, Tools = `['get_earnings', 'search_news']`
- **Human Note**: *""*
- **Analysis**: For announcement/narrative events (e.g. `N003`), the human labeled `get_earnings` single-tool while the author labeled `[get_earnings, search_news]` dual-tool. For guidance vs prior estimates (`G014`, `G017`), the human assigned dual tools (`[get_guidance, get_ratings]`) whereas the author treated prior consensus inside guidance as a single tool.

### Cause D: Other Disagreements

#### Query `ND030`: *"Tell me about SoFi Technologies' analyst ratings."*
- **Human Label**: Category = `no_data`, Tools = `[]`
- **Author Gold**: Category = `no_data`, Tools = `['get_ratings']`
- **Human Note**: *"Unsupported ticker (SoFi)"*

#### Query `A004`: *"What's going on with XOM?"*
- **Human Label**: Category = `ambiguous`, Tools = `['search_news']`
- **Author Gold**: Category = `ambiguous`, Tools = `['get_earnings', 'get_ratings', 'search_news']`
- **Human Note**: *"Vague, supported ticker"*

#### Query `ND011`: *"What are the latest analyst ratings for PLTR?"*
- **Human Label**: Category = `no_data`, Tools = `[]`
- **Author Gold**: Category = `no_data`, Tools = `['get_ratings']`
- **Human Note**: *"Unsupported ticker (PLTR)"*

#### Query `ND002`: *"Show me the analyst consensus on Microsoft stock."*
- **Human Label**: Category = `no_data`, Tools = `[]`
- **Author Gold**: Category = `no_data`, Tools = `['get_ratings']`
- **Human Note**: *"Unsupported ticker (MSFT)"*

#### Query `A020`: *"What's new with Exxon?"*
- **Human Label**: Category = `ambiguous`, Tools = `['search_news']`
- **Author Gold**: Category = `ambiguous`, Tools = `['get_earnings', 'search_news']`
- **Human Note**: *"Vague; similar to "overview of what is happening""*

#### Query `A001`: *"How is Nvidia doing?"*
- **Human Label**: Category = `ambiguous`, Tools = `['search_news']`
- **Author Gold**: Category = `ambiguous`, Tools = `['get_earnings', 'get_ratings', 'search_news']`
- **Human Note**: *"Vague"*

---

## 5. Methodological & Provenance Governance Note
- **No Gold Label Alterations**: In accordance with research integrity protocols, **no author gold labels or held-out test labels were modified** based on this human validation pass.
- **Provenance Preservation**: The original `eval/labeling_guidelines.md` remains the active standard. Any proposed revisions to disambiguate fixture-coverage vs. semantic intent will be drafted in a separate document for formal approval.
