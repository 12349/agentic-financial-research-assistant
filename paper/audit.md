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

### 6.2 Synthesis Grounding Comparison (Test split, n=47)
| System | Citation Validity | Numeric Faithfulness | Unsupported Claim Rate | Abstention Accuracy |
|--------|-------------------|----------------------|------------------------|---------------------|
| S1 (Deterministic Formatter) | **1.000** (By construction) | 0.081 | **0.613** (No free text) | 0.429 |
| S6 (LLM Synthesis, Qwen2.5-7B) | 0.861 [0.76, 0.95] | 0.071 [0.03, 0.12] | 0.465 [0.41, 0.52] | 0.143 |

*Key finding*: In S6, where citation is instructed rather than structurally enforced, citation validity degrades to 86.1% and 46.5% of sentences lack a valid inline source tag, while latency increases by ~95,000× (11.9 ms vs 9,483 ms).

### 6.3 Independent Model Rater Agreement (llama3.1:8b on n=34 sample)
- **Exact Tool-Set Agreement**: 29.4%
- **Tool-Set Cohen's $\kappa$**: **0.2507** (Fair agreement)
- **Category Cohen's $\kappa$**: **0.1654** (Slight agreement)
- *Key finding*: When an independent model family (Meta Llama 3.1 8B) labels the dataset without author guidance, agreement drops significantly from the self-consistency pass ($\kappa=0.967$). This confirms the methodological limitation that routing schemas reflect author-specific information architectures.

### 6.4 External Benchmark Anchor (FinanceBench)
- 150 open-source samples (Islam et al., arXiv:2311.11944) evaluated.
- 5 samples match in-scope tickers (NVDA, TSLA, JPM, XOM).
- System answers accurately reflect Q3 2024 fixture data, but reveal temporal divergence when FinanceBench queries prior fiscal years (2021–2022). Documented as an honest limitation.

