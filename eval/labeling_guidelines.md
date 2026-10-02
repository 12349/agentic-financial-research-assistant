# Annotation & Labeling Guidelines for Financial Query Routing

This document defines the formal annotation guidelines for categorizing financial research queries and assigning gold-standard tool-selection labels. These guidelines serve as the authoritative standard for human annotators and model-based raters.

---

## 1. Supported Scope & System Capabilities

The research assistant operates over a fixed database of four equity tickers:
- **NVDA** (NVIDIA Corporation)
- **TSLA** (Tesla, Inc.)
- **JPM** (JPMorgan Chase & Co.)
- **XOM** (Exxon Mobil Corporation)

Queries concerning any other company (e.g., Apple, Amazon, Google, Microsoft, Coinbase, Boeing) are strictly **out of scope**.

### Tool Definitions & Explicit Boundaries

| Tool | Semantic Scope | Triggering Concepts | Exclusion / Do Not Use |
| :--- | :--- | :--- | :--- |
| `get_earnings` | Historical reported quarterly/annual earnings releases | Revenue, EPS, net income, profit, earnings beat/miss, earnings release date, YoY growth | Forward forecasts, price targets, general news stories |
| `get_guidance` | Official corporate forward-looking management guidance | Management guidance, forward revenue guidance, delivery targets, fiscal outlook, guidance raise/cut | Analyst estimates (unless comparing to guidance), historical reported earnings |
| `get_ratings` | Wall Street sell-side analyst research and consensus | Analyst ratings, upgrades, downgrades, price targets, Wall Street sentiment, consensus | General news headlines, corporate earnings reports |
| `search_news` | Press coverage, market developments, regulatory affairs | News, articles, headlines, press releases, events, lawsuits, product launches, macroeconomic context | Specific earnings numbers or price targets when asked without news context |

---

## 2. Query Categories

Every query is assigned to exactly one of the four mutually exclusive categories below.

### 2.1 `single_tool`
- **Definition**: The query has a single, unambiguous financial intent that requires exactly one tool.
- **Rule**: Assign the one tool that directly answers the question.
- **Examples**:
  - *"What was Nvidia's revenue last quarter?"* $\to$ `[get_earnings]`
  - *"What is Tesla's delivery target for FY2024?"* $\to$ `[get_guidance]`
  - *"What is Morgan Stanley's price target for JPM?"* $\to$ `[get_ratings]`
  - *"Any recent headlines about Exxon's litigation?"* $\to$ `[search_news]`

### 2.2 `dual_tool`
- **Definition**: The query explicitly joins two distinct information needs that each map to different tools. Both tools are strictly necessary to answer the prompt completely.
- **Rule**: Assign exactly the two tools required. Do not add `search_news` as a catch-all if the query asks for earnings and ratings.
- **Examples**:
  - *"Show me Tesla's Q3 earnings and where analysts have their price targets."* $\to$ `[get_earnings, get_ratings]`
  - *"What did Exxon guide for capex, and what is the latest news on OPEC?"* $\to$ `[get_guidance, search_news]`
  - *"Did Nvidia beat earnings estimates and what is their forward Q4 guidance?"* $\to$ `[get_earnings, get_guidance]`

### 2.3 `ambiguous`
- **Definition**: The query concerns a **supported company** (NVDA, TSLA, JPM, XOM) but is broad, underspecified, or open-ended, such that no specific metric or table is exclusively requested.
- **Rule**: Assign the tool(s) that provide high-level contextual research. By default, broad exploratory queries route to `search_news` or a combination of context tools (`search_news`, `get_ratings`).
- **Critical Distinction**: If the query is vague but mentions a supported company (e.g., *"What is the outlook for Tesla?"* or *"Research Nvidia for me"*), it is **`ambiguous`**, **NEVER** `no_data`.
- **Examples**:
  - *"What's the outlook for Tesla?"* $\to$ `[get_guidance, get_ratings, search_news]` (or `[search_news]` depending on intent specificity)
  - *"Give me an overview of what is happening with JPM."* $\to$ `[search_news]`

### 2.4 `no_data`
- **Definition**: The query cannot be answered by the assistant because the required data does not exist in the system.
- **Rule**: The gold tool set for all `no_data` queries is strictly **empty** (`[]`). The correct system action is honest abstention.
- **Trigger Conditions**:
  1. **Unsupported Tickers**: Any query about entities other than NVDA, TSLA, JPM, XOM (e.g., *"What is Amazon's revenue guidance?"*, *"What's the news on Coinbase?"*).
  2. **Unsupported Data Modalities**: Financial metrics not tracked by the four tools, even for supported companies:
     - Options flow, implied volatility, Greeks, put/call ratios
     - Credit default swap (CDS) spreads, bond yields
     - Technical charting indicators (RSI, MACD, moving averages)
     - Order book / Level 2 dark pool data
     - Insider transactions / Form 4 filings
     - Dividend payment schedules (unless part of earnings)
- **Examples**:
  - *"What is Amazon's forward revenue guidance?"* $\to$ `[]` (unsupported ticker)
  - *"Show me NVDA call option open interest for next week."* $\to$ `[]` (unsupported modality)
  - *"What are JPMorgan's CDS spreads?"* $\to$ `[]` (unsupported modality)

---

## 3. Decision Flowchart for Annotators

```
1. Does the query request data about an unsupported ticker (not NVDA, TSLA, JPM, XOM)
   OR an unsupported financial modality (options, CDS, technical indicators)?
   ├── YES ──> Category = "no_data", Gold Tools = []
   └── NO  ──> Proceed to step 2.

2. Does the query contain specific requests for data?
   ├── NO (vague/broad, e.g. "tell me about NVDA", "Tesla outlook")
   │   └──> Category = "ambiguous", Gold Tools = [search_news] (or relevant context tools)
   └── YES ──> Proceed to step 3.

3. How many distinct tool domains are explicitly requested?
   ├── Exactly 1 domain (e.g. only earnings, or only ratings)
   │   └──> Category = "single_tool", Gold Tools = [<matching_tool>]
   └── 2 distinct domains (e.g. earnings + analyst ratings)
       └──> Category = "dual_tool", Gold Tools = [<tool1>, <tool2>]
```

---

## 4. Quality Control & Independence

1. **Independence**: Annotations must NEVER be produced or influenced by the system under test (e.g., running `planner.rule_based.plan()` or any model planner to generate labels).
2. **Gold Label Integrity**: Any match between gold labels and system predictions must arise solely from independent adherence to these guidelines, never from circular derivation.
