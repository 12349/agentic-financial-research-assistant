# FinanceBench Qualitative Case Study (n=5)

> **Important Framing Notice**: This evaluation is **strictly a qualitative case study**, NOT an empirical benchmark comparison. FinanceBench was designed for long-document question answering over historical SEC 10-K/10-Q PDF filings, whereas our assistant is architected for tool routing over structured financial databases and recent news feeds.

---

## 1. Study Motivation & Scope

FinanceBench (Islam et al., 2023; arXiv:2311.11944) comprises 150 open-source financial question-answering pairs targeting SEC filings.

To investigate how an agentic tool-routing assistant interacts with realistic external financial inquiries, we extracted all questions mentioning the four supported companies in our benchmark:
- **Total Questions in FinanceBench**: 150
- **Questions Referencing In-Scope Tickers (NVDA, TSLA, JPM, XOM)**: **5** (3.3%)
- **Target Filings in FinanceBench**: 2020–2022 Form 10-K and 10-Q filings.
- **Data Coverage in Assistant**: Q3 2024 structured earnings, forward guidance, analyst ratings, and Q4 2024 news.

---

## 2. Qualitative Case Analysis

### Case 1: Historical Segment Detail (`financebench_id_00299`)
- **Question**: *"Which of JPM's business segments had the lowest net revenue in 2021 Q1?"*
- **Gold Answer**: Corporate segment (net revenue was -$473 million).
- **Assistant Routing**: Routes to `search_news(ticker='JPM')`.
- **System Output**: Returns Q3 2024 JPMorgan earnings news (record net income of $12.9B).
- **Analysis**: **Temporal & Granularity Mismatch**. The query requires multi-year historical segment breakdown from the 2021 Q1 10-Q filing. The assistant's structured database only indexes the latest reporting period (Q3 2024) and does not retain granular segment tables from three years prior.

### Case 2: Hypothetical Balance Sheet Liquidation (`financebench_id_02119`)
- **Question**: *"If JPM went bankrupt by the end of 2021 Q1 and liquidated all of its assets to pay its shareholders, how much could each shareholder get?"*
- **Gold Answer**: $66.56 per share (derived by dividing book equity by diluted share count).
- **Assistant Routing**: Routes to `search_news(ticker='JPM')`.
- **System Output**: Returns recent earnings commentary.
- **Analysis**: **Modality Mismatch**. The question requires analytical reasoning over historical balance-sheet lines (total assets, total liabilities, share count). Our system provides high-level corporate metrics (revenue, EPS, guidance, price targets) rather than full multi-page balance-sheet balance equations.

### Case 3: Conceptual Accounting Applicability (`financebench_id_00206`)
- **Question**: *"Are JPM's gross margins historically consistent? If gross margins are not a relevant metric for a company like this, then please state that and explain why."*
- **Gold Answer**: Commercial banks do not report gross margins because their primary revenue engine is net interest income and non-interest fees; gross margin is not an applicable banking metric.
- **Assistant Routing**: Routes to `search_news(ticker='JPM')`.
- **System Output**: News summary of interest income trends.
- **Analysis**: **Domain Reasoning Deficit**. Answering requires meta-accounting knowledge about bank financial reporting standards rather than data retrieval.

### Cases 4 & 5: Complex Footnote Extraction
- Similar divergences occur for questions requiring footnotes on stock-based compensation and depreciation schedules from 2021 filings.

---

## 3. Methodological Conclusion

FinanceBench cannot be evaluated as an apples-to-apples benchmark for structured tool-routing assistants due to:
1. **Temporal Disconnect**: 2021/2022 retrospective filings vs. contemporary 2024 operational data.
2. **Document vs. Database Retrieval**: Unstructured full-text SEC search vs. API-style structured metric retrieval.

This case study demonstrates that general financial QA requires hybrid architectures that combine structured API tools for timely metric lookup with full-text filing RAG for historical deep-dives.
