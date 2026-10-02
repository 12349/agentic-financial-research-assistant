# Phase 1 — Research Artifact Audit

> **Principle**: Every claim here is sourced to the actual code or data.
> Nothing is invented. Weaknesses are stated plainly.

---

## 1. Architecture As Actually Implemented

### 1.1 Component Map

The pipeline is: Query → llm_planner.plan() → orchestrator.run() → [tools] → synthesizer.synthesize() → TraceLogger → Response dict.

**LLM Planner** (planner/llm_planner.py):
- Auto-detects GROQ_API_KEY → Groq llama-3.3-70b-versatile (temperature=0)
- Auto-detects ANTHROPIC_API_KEY → Anthropic claude-3-haiku (temperature=0)
- Neither key → calls rule_based.plan() as fallback
- Single-shot prompt: returns JSON array of ≤2 tool calls, no chain-of-thought

**Orchestrator** (agent/orchestrator.py):
- Hard cap: MAX_TOOL_CALLS = 4 (configurable env var)
- Executes plan sequentially (no parallel calls)
- Per-call try/except → logs error, appends partial output, continues
- All calls logged via TraceLogger

**Tools**:
- search_news(query, ticker): all-MiniLM-L6-v2 + FAISS IndexFlatIP in-memory, cosine threshold 0.30, TF-IDF fallback if faiss/sentence-transformers absent. Source: 14 fixture articles (4 tickers) or REST API.
- get_ratings(ticker): ratings fixtures + two-layer sentiment (DistilBERT if weights present, else keyword bucket). 
- get_guidance(ticker): guidance fixtures or REST API.
- get_earnings(ticker): earnings fixtures or REST API.

**Synthesizer** (agent/synthesizer.py):
- Fallback (deterministic): template-fills named fields from tool_output dicts. Every sentence starts with "[Source: record_id]". Cannot hallucinate — citation is structural.
- LLM path (Groq/Anthropic): serialises tool_outputs as JSON → LLM prompt that instructs citing record_ids. Citation NOT structurally verified.

**Trace Logger** (logger/trace_logger.py): writes logs/<hash>.json per query.

**Go Ingestor** (go-ingestor/): standalone Go service, NOT wired to Flask app. No /ingest endpoint, no shared queue. Architecture demo only.

### 1.2 Rule-Based Planner Design

- 6 priority-ordered keyword buckets: guidance+earnings (multi-tool), earnings-only, guidance-only, ratings, news (catch-all)
- Ticker extraction: company name map (nvidia→NVDA, tesla→TSLA, jpmorgan→JPM, exxon→XOM) + regex fallback
- Hardcoded known tickers: NVDA, TSLA, JPM, XOM only
- Max 2 tool calls ever emitted
- CRITICAL: keyword table was written alongside the 5 acceptance test queries — it is partially in-sample on the eval set

### 1.3 Fixture Data Scale

| Fixture | Tickers | Est. Records |
|---------|---------|-------------|
| news_fixtures.json | NVDA, TSLA, JPM, XOM | 14 articles |
| ratings_fixtures.json | NVDA, TSLA, JPM, XOM | ~12 ratings |
| guidance_fixtures.json | NVDA, TSLA, JPM, XOM | 4 records (1 per ticker) |
| earnings_fixtures.json | NVDA, TSLA, JPM, XOM | 4 records (1 per ticker) |

### 1.4 Classifier Metrics (Real Numbers from classifier/training_results.md)

- Model: distilbert-base-uncased, 3 epochs, batch 32, lr=2e-5, CPU, 541.4s
- Dataset: 9,543 Twitter Financial News Sentiment + 25 hand-labeled fixture notes
- Split: 80/20 stratified, held-out test n=1,914
- Weighted F1: 0.850 | Macro F1: 0.804
- Per-class: bearish F1=0.726, neutral F1=0.899, bullish F1=0.787
- Integration: OPTIONAL. In default offline deployment, keyword bucket is used.

---

## 2. Novelty Assessment

### 2.1 Standard / Prior Art

| Component | Prior Art |
|-----------|-----------|
| Tool-augmented LLM planning | ReAct (Yao+2023), ToolFormer (Schick+2023), OpenAI function calling |
| Keyword intent routing | Classical slot-filling, intent detection (Tur & De Mori 2011) |
| FAISS dense retrieval | DPR (Karpukhin+2020), RAG (Lewis+2020) |
| DistilBERT fine-tune on Twitter finance | Sanh+2019; twitter-financial-news-sentiment benchmark |
| Attribution / grounded generation | AIS (Rashkin+2023), AttributionQA (Bohnet+2022) |
| Financial QA | FinQA (Chen+2021), TAT-QA (Zhu+2021), DocFinQA |

### 2.2 Defensible Contributions

**C1 — Grounded-by-Construction Synthesis (Fallback Path)**
The deterministic formatter in synthesizer._synthesize_fallback() produces answers where citation is a structural property, not a soft instruction. Every sentence is built by filling named fields from a specific fixture record; the [Source: record_id] prefix is emitted by the template. This guarantees zero false-citation rate in fallback mode — verifiable by code inspection, not just by measurement.

Boundary: LLM synthesis path prompts for citation but does not structurally verify. The guarantee is path-specific.

**C2 — Bounded Tool-Calling with Logged Cap**
MAX_TOOL_CALLS=4 is enforced as a hard cap in the orchestrator, independent of the planner. Cap events are logged. This is a defense against unbounded agent loops, which are a documented failure mode. The cap is configurable and auditable.

Boundary: with a planner that emits ≤2 calls, the cap of 4 is never triggered in the current codebase. The contribution is architectural, not empirically demonstrated.

**C3 — Dual-Planner with Measured Divergence**
Two planners (rule-based, LLM) with automatic fallback. The 8-query divergence measurement is concrete and reproducible: 6/8 agree, 2/8 LLM correctly expands the tool set, latency tradeoff quantified.

Boundary: only 8 queries. No full-benchmark LLM planner results.

**C4 — 229-Query Financial Tool-Selection Benchmark with IRR**
Formally structured eval set: 60/20/20 splits, category balance, fixture-grounded gold labels, second-pass IRR (κ=0.967 tool-set, κ=0.918 category). Corrections applied from IRR outcome (N003, A013).

Boundary: self-generated queries, possible same-rater IRR, no external human queries.

### 2.3 Honest Summary

This is a well-engineered system with clear architectural decisions and an unusually rigorous documentation culture for a project of this size. The strongest paper claim is the grounded-by-construction synthesis design pattern empirically compared against the LLM synthesis path. The benchmark is the second claim. Neither alone is likely to clear a top-venue full-paper bar, but together they support a credible system-track or workshop paper.

---

## 3. Weaknesses a Reviewer Will Attack

### W1 [CRITICAL]: Tiny Self-Generated Fixture Data
4 tickers, 1 period each, ~34 total fixture records. All 229 eval queries were generated by the system's author. Gold labels are field values from the same fixtures the system retrieves — circular evaluation. Attack: "The benchmark is not independent of the system." Mitigation: external human-written queries; blind annotator; crowd-source annotation.

### W2 [SERIOUS]: Rule Planner Tuned to Eval Set
The keyword table in rule_based.py was written by the same person who wrote the eval queries. 100% on the 5 acceptance + 8 stress queries is by design. The only honest number is 14/15 on the 15 blind held-out queries — but those were also written by the same author. Mitigation: report planner accuracy separately on the held-out 47-query test split; treat train/dev queries as off-limits.

### W3 [SERIOUS]: LLM Planner Not Evaluated on Benchmark
Zero end-to-end LLM planner results on the 229-query set. The blind test used the rule planner. Attack: "The paper proposes an LLM planner but presents no evaluation of it." Mitigation: run LLM planner on test split; report precision/recall/F1 for tool selection.

### W4 [MODERATE]: IRR Not Genuinely Independent
irr_second_pass.py contains hand-coded labels by the same developer. No evidence of a different person or different model family as rater. κ=0.967 is suspiciously high for genuinely independent annotation on a nuanced task — it is consistent with one person re-labeling their own work with a slightly different framing. Mitigation: a human annotator OR a clearly different model family (e.g., GPT-4 if Claude was used for development) with documented prompts.

### W5 [MODERATE]: Grounding Claim Overstated for LLM Path
The paper's strongest claim (zero false-citation) applies only to the deterministic formatter. In LLM mode, citation is instructed but not verified. The system has no post-hoc grounding verifier. Mitigation: implement citation precision check: does cited record_id exist in tool_output, and do quoted numbers match the source field? Report separately for both paths.

### W6 [MODERATE]: No Live-Data Results
All experiments are on fixture data. Live API paths exist but are untested in any reported experiment. Mitigation: at minimum, a qualitative test on 5 live queries.

### W7 [MINOR]: Latency Numbers Anecdotal
"280–1610ms" for Groq planner is from 8 queries, no variance reported, no cold/warm distinction. Mitigation: 30-run average with stddev.

### W8 [MINOR]: DistilBERT Sentiment Not Ablated End-to-End
Isolated classifier F1=0.850 is measured, but no ablation of ML vs. keyword sentiment in the full agent's final answer quality. Mitigation: ablation comparing keyword-only vs. DistilBERT sentiment on get_ratings output.

---

## 4. Reproducibility Matrix

| Experiment | Status | Command |
|------------|--------|---------|
| 5 acceptance tests | ✅ Fully reproducible | `PYTHONPATH=. python3 run_tests.py` |
| Rule planner on 8 queries | ✅ Fully reproducible | `PYTHONPATH=. python3 eval/llm_vs_rule_test.py` (no key) |
| 15 blind queries (rule planner) | ✅ Fully reproducible | `PYTHONPATH=. python3 eval/blind_test.py` |
| 229-query eval set generation | ✅ Fully reproducible | `PYTHONPATH=. python3 eval/generate_eval_set.py` |
| IRR second pass | ✅ Fully reproducible | `PYTHONPATH=. python3 eval/irr_second_pass.py` |
| DistilBERT classifier train/eval | ✅ With pip install | `python3 classifier/train.py && python3 classifier/evaluate.py` |
| LLM planner on 8 queries | ⚠️ Needs GROQ_API_KEY | `GROQ_API_KEY=... python3 eval/llm_vs_rule_test.py` |
| LLM planner on full benchmark | ❌ Not implemented | — needs building |
| End-to-end LLM agent results | ❌ Not implemented | — needs building |
| Citation precision measurement | ❌ Not implemented | — needs building |
| Live-data test | ❌ Not implemented | — needs credentials |
| Grounding ablation (LLM vs fallback) | ❌ Not implemented | — needs building |
| Sentiment ablation (ML vs keyword) | ❌ Not implemented | — needs building |

---

## 5. Recommended Paper Framing

**Venue**: System description or resource paper track. Best fits: ACL System Demonstrations, EMNLP Findings, ECIR Industry Track, IEEE SSCI, FinNLP workshop (co-located with ACL/EMNLP). For a full research track submission, W1–W3 must be addressed first.

**Honest abstract draft**:
> We present FinAgent, a tool-augmented financial QA system with a dual-planner architecture (rule-based and LLM-based) and a grounded-by-construction answer synthesis mode. Our primary contribution is a synthesis design pattern that achieves zero false-citation rate by construction in its deterministic path, without LLM involvement in citation generation. We additionally contribute a 229-query financial tool-selection benchmark with inter-rater reliability κ=0.967. We evaluate honestly: end-to-end results use the rule-based planner on offline fixture data (4 tickers); the LLM planner is evaluated on 8 queries only; the benchmark consists entirely of author-generated queries. We identify these as limitations and describe the experimental work needed to address them.

**Do NOT claim in the paper**:
- State-of-the-art on any existing benchmark (no comparison made)
- That the LLM synthesis path is grounded (it is not structurally enforced)
- That IRR demonstrates independent agreement (independence not established)
- That the system generalizes beyond 4 tickers (untested)
- That the rule-based planner has 100% accuracy (in-sample result)

---

## 6. Phase 2 Empirical Findings & Verified Results (2026-10-02)

All experiments run locally on Apple M4 (16 GB unified memory, macOS 27.0) via Ollama with zero paid cloud APIs ($0.00 cost). Seed=42, temperature=0.

### 6.1 Tool Selection on Held-Out Test Split (n=47)
| System | Planner | Precision | Recall | Tool-Set F1 [95% CI] | Exact Match [95% CI] | Tool Calls / Q | Mean Latency (ms) |
|--------|---------|-----------|--------|----------------------|----------------------|----------------|-------------------|
| S1 | Rule-Based | 0.628 | 0.495 | 0.522 [0.41, 0.64] | 0.319 [0.19, 0.47] | 1.21 | 11.9 (warm) / 52.0 (eval) |
| S2 | Qwen2.5-7B | 0.713 | 0.674 | **0.677** [0.55, 0.79] | **0.638** [0.49, 0.77] | 1.13 | 1061.4 (warm) / 1647.4 (eval) |
| S3 | No-Tools | 0.000 | 0.000 | 0.000 [0.00, 0.00] | 0.064 [0.00, 0.15] | 0.00 | 2505.7 |
| S4 | Call-All | 0.383 | **0.936** | 0.516 [0.45, 0.58] | 0.064 [0.00, 0.15] | 4.00 | 18.8 |
| S5 | ReAct (Qwen2.5-7B) | 0.486 | 0.436 | 0.433 [0.31, 0.56] | 0.340 [0.21, 0.49] | 0.89 | 17,140.3 |

**Statistical Significance**:
- Paired McNemar's test on exact match (S1 vs S2): $\chi^2 = 8.5217$, **$p = 0.0035$** (statistically significant at $p < 0.01$).
- Wilcoxon signed-rank test on F1 (S1 vs S2): $W = 130.0$, $p = 0.0935$.

### 6.2 Synthesis Grounding Comparison (Held-Out Test Split, n=47)

| System | Citation Validity | Numeric Faithfulness | Unsupported Claim Rate | Abstention Accuracy | Latency (Mean) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **S1 (Deterministic Formatter)** | **1.000** [1.0, 1.0] *(By construction)* | **1.000** [1.0, 1.0] | **0.000** [0.0, 0.0] | **0.429** (3/7) | **9.8 ms** |
| **S6 (LLM Synthesis, Qwen2.5-7B)** | 0.859 [0.76, 0.94] | 0.975 [0.95, 0.99] | 0.238 [0.18, 0.30] | 0.143 (1/7) | 9,574.2 ms |

**Reassessment of the Core Synthesis Finding**:
- **Status: CONFIRMED & SOLIDIFIED**.
- Under rigorous per-sentence citation governance and scaled numeric parsing, the deterministic programmatic formatter (S1) achieves **100% citation validity**, **100% numeric faithfulness**, and **0.0% unsupported statements** at 9.8 ms latency.
- In contrast, generative LLM synthesis (S6, Qwen 2.5 7B) exhibits clear empirical degradation:
  - 14.1% of citations reference non-existent records or omit required source tags.
  - 23.8% of generated sentences are unsupported by any retrieved record.
  - Numeric faithfulness drops to 97.5% due to occasional hallucinated figures or rounding mismatches.
  - Abstention on unanswerable/out-of-scope queries drops from 42.9% to 14.3% (the generative model fabricates plausible-sounding explanations when tool records are absent).
  - Latency is ~970× higher (9.6 seconds vs. 9.8 milliseconds).
- **Framing Correction**: The phrase *"mathematical grounding guarantee"* is permanently removed. The defensible framing is **"Programmatic Structural Attribution vs. Generative LLM Synthesis"**: programmatic field copying structurally prevents citation fabrication by design, whereas prompt-instructed LLM generation degrades across all fidelity dimensions.

---

### 6.3 Inter-Rater Reliability & Human Annotation Protocol

To assess labeling reliability beyond the author's self-consistency pass ($\kappa=0.967$), an independent LLM rater from an external model family (`llama3.1:8b`) annotated the 15% sample ($n=34$, seed 999):

| Condition | Tool-Set Exact Match | Tool-Set Cohen's κ | Category Exact Match | Category Cohen's κ |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (No Guidelines)** | 29.4% (10/34) | **0.2507** (Fair) | 44.1% (15/34) | **0.1654** (Slight) |
| **Guided Pass (With Guidelines)** | 38.2% (13/34) | **0.3419** (Fair) | 61.8% (21/34) | **0.3652** (Fair) |

- **Key Finding**: Providing explicit written guidelines ([`eval/labeling_guidelines.md`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/eval/labeling_guidelines.md)) increased tool-set agreement by $\Delta\kappa = +0.091$ and category agreement by $\Delta\kappa = +0.200$.
- **Human Annotation Sheet**: A stratified 50-query subset from dev/train was prepared in [`eval/human_annotation_sample.csv`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/eval/human_annotation_sample.csv) with blank columns for human rating.
- **Label Leakage Resolution**: Identified and eliminated circular derivation in `external_queries.jsonl` (previously 100% agreed with the rule planner). Regenerated independently with `llama3.1:8b` under formal guidelines; agreement with the rule planner dropped to **48.0%**, verified by CI regression tests in [`tests/test_label_integrity.py`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/tests/test_label_integrity.py).

---

### 6.4 External Data Case Study (FinanceBench Qualitative Analysis)

- **Framing**: Reframed strictly as a **qualitative case study** ($n=5$ in-scope samples from Islam et al., arXiv:2311.11944), **NOT** a benchmark score.
- **Key Observation**: Documents the fundamental boundary between SEC 10-K document QA and structured database routing:
  1. *Temporal Mismatch*: FinanceBench targets 2020–2022 retrospective annual filings; the assistant's fixtures index Q3 2024 operating metrics.
  2. *Modality Mismatch*: Questions asking for hypothetical liquidation values or balance sheet restructuring require multi-page financial accounting reasoning, whereas the assistant's tools query structured metric endpoints.
- Detailed case-by-case findings are archived in [`results/financebench/qualitative_case_study.md`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/results/financebench/qualitative_case_study.md).

---

### 6.5 Ablations & Empirical Pathologies

1. **A1 (ReAct Tool Cap on Dev, $n=45$)**:
   - Caps tested: 2, 4, 8, unlimited (max 10).
   - Finding: ReAct agent with `qwen2.5:7b-instruct` exhibits an **early-stopping pathology** (mean steps to termination: **1.38**). The agent almost never engages in runaway tool loops; instead, it prematurely emits `Final Answer:` without invoking necessary tools. Tool calls per query remain virtually flat: 0.36 (cap 2) vs. 0.38 (cap 4, 8, unlimited).
2. **A5 (News Search Engine on Dev, $n=17$)**:
   - Both FAISS Semantic and TF-IDF achieve identical Top-1 Precision: **0.941** (16/17 relevant; 1 query unretrieved under threshold). Top-1 agreement is **0.941** (16/17 identical articles).
   - **Hypothesis**: In a compact, curated 12-document corpus (3 distinct articles per ticker), distinct company names and topic vocabularies allow keyword matching to converge with dense embeddings.

---

### 6.6 Reassessed Paper Contributions & Publication Framing

Based strictly on verified, audited empirical data:

1. **Contribution 1 (Empirical Evaluation of Programmatic vs. Generative Synthesis in Financial QA)**:
   - Programmatic structural attribution achieves 100% citation validity and 100% numeric faithfulness with sub-10ms latency.
   - Generative LLM synthesis degrades citation validity by 14.1%, generates 23.8% unsupported statements, and increases latency by ~970×.
   - This provides actionable empirical guidance for production financial systems: structured metrics should be rendered programmatically, reserving generative LLMs for natural-language query planning.
2. **Contribution 2 (Single-Shot Structured Planning Outperforms Multi-Step ReAct in Constrained Tool Routing)**:
   - Single-shot LLM planning (S2) achieves **0.677 F1** and **63.8% Exact Match** ($p=0.0035$ over rule-based baseline via McNemar's test).
   - ReAct (S5) achieves only **0.433 F1** and **34.0% Exact Match**, hobbled by early stopping (mean 1.38 steps).
3. **Contribution 3 (Rigorous Evaluation Methodology & Provenance for Applied Financial Agent Research)**:
   - Full test-set isolation ($n=47$ held-out split never used for tuning).
   - CI-enforced regression tests guarding against circular label leakage.
   - Transparent documentation of inter-rater reliability boundaries ($\kappa=0.25$ unprompted $\to 0.34$ guided).


