# Phase 2 — Experiment Plan
**Status: AWAITING APPROVAL — nothing runs until approved.**
**Last revised**: 2026-10-01

---

## 0. Governing Constraints

| Constraint | Value |
|-----------|-------|
| Budget | **$0** — all inference local via Ollama |
| Machine | Mac mini M4, 16 GB unified memory, macOS 27.0 |
| Primary LLM | `qwen3.6:latest` (36B MoE, Q4_K_M, ~23 GB, ~3.6B active params/token) |
| Second-family LLM | `llama3.1:8b` (Meta, Q4_K_M, ~4.7 GB) — different vendor family |
| Temperature | 0 for all LLM calls |
| Seed | 42 everywhere |
| Test split | Sacred: 47 queries, touched once per system at final reporting time |
| Tuning | Train (n=137) + dev (n=45) only |
| Weak results | Reported as-is; audit.md updated; experiment never re-run to improve numbers |

**Key model notes:**
- `qwen3.6` has an extended thinking mode (chain-of-thought) that fires by default.
  It **must be disabled** with `think=false` / `/no_think` in all planner calls.
  Thinking mode inflates latency 5–30× for short structured JSON tasks.
- Both models are on-disk already (qwen3.6) or will be pulled (llama3.1:8b via `scripts/setup_ollama.sh`).
- **Never load both models simultaneously** — 16 GB RAM fits one at a time.
  The external-rater pass (llama3.1:8b) runs in a separate subprocess after qwen3.6 is unloaded.

---

## 1. Systems Under Test

| ID | Name | Planner | Synthesizer | LLM? | Notes |
|----|------|---------|-------------|------|-------|
| S1 | `rule-planner` | Rule-based (`planner/rule_based.py`) | Deterministic formatter | No | Current default; offline |
| S2 | `llm-planner` | qwen3.6 JSON output | Deterministic formatter | Yes | Planner only; isolates planner contribution |
| S3 | `no-tools` | — | qwen3.6 single-shot (no tools) | Yes | Parametric baseline; P/R/F1 = 0 by definition |
| S4 | `call-all` | Always call all 4 tools | Deterministic formatter | No | Max-recall baseline |
| S5 | `react` | qwen3.6 ReAct loop | qwen3.6 | Yes | Think→Act→Observe, capped at 4 steps |
| S6 | `llm-synth` | Rule-based | qwen3.6 LLM synthesis | Yes | Tests synthesis claim; same planner as S1 |

**Model pinning** (written to `results/metadata.json` by `scripts/setup_ollama.sh`):
```
Primary:  qwen3.6:latest  — digest recorded at setup time
          architecture: qwen35moe, quantization: Q4_K_M, 36B total / ~3.6B active
          temperature: 0, think: false (extended thinking disabled for all structured tasks)
          seed: 42 (passed via Ollama API options.seed)

Rater:    llama3.1:8b     — digest recorded at setup time
          architecture: llama3.1 dense 8B, quantization: Q4_K_M
          temperature: 0, seed: 42
          vendor: Meta/Llama (different family from qwen/Alibaba)
```

**Existing Groq/Anthropic code paths** in `planner/llm_planner.py` and `agent/synthesizer.py`
are left unchanged and fully functional. They are not tested in this phase. All LLM
experiments use the new Ollama path in `experiments/ollama_client.py`.

---

## 2. Evaluation Metrics

### 2A. Tool-Selection (Planner)
Applied to S1, S2, S4, S5. S3 is always zero.

| Metric | Definition |
|--------|-----------|
| Tool-set Precision | TP / (TP+FP) — micro-averaged per query |
| Tool-set Recall | TP / (TP+FN) |
| Tool-set F1 | Harmonic mean of P and R |
| Exact-set Match | 1 if predicted_tools == gold_tools (frozensets) else 0 |
| Tool calls / query | mean(len(plan)) across queries |
| Abstention accuracy | Fraction of `no_data` queries where system calls 0 tools OR returns "No data found" |

### 2B. Synthesis Quality — Core Claim
For all systems (S1 formatter vs S6 LLM synthesis), same planner (rule-based):

| Metric | Method |
|--------|--------|
| Citation validity | Script: every `[Source: X]` in answer → X exists in tool_outputs keys |
| Numeric faithfulness | Script: extract all numbers from answer → verify each against corresponding source field ±0.5% |
| Unsupported-claim rate | Sentences containing no citation / total sentences |
| Abstention correctness | On `no_data` queries: does answer say "No data" / contain no made-up numbers? |
| Answer length (tokens) | `len(answer.split())` as verbosity proxy |

**Framing rule**: The formatter's citation validity = 100% is **true by construction**,
not an achievement. Do NOT present it as one. The contribution is:
(a) that the guarantee holds structurally, and
(b) measuring the **gap** to LLM synthesis (S6) — where citation is instructed but not enforced.

### 2C. Latency & Hardware
- 2 cold-start warm-up runs (discarded), then 30 timed runs per system
- Cold: first call with Ollama model just loaded (process just started)
- Warm: runs 3–32 (model already in memory)
- Report: mean ± std (ms), median, p95, cold separately
- Tokens: prompt tokens + completion tokens per call
- Hardware: M4 chip, 16 GB RAM, Q4_K_M — reported in every table caption

### 2D. Statistical Tests
- Bootstrap 95% CIs: B=10,000, seed=42, on all primary metrics
- Paired McNemar's test: exact-set match between S1 vs S2
- Wilcoxon signed-rank test: F1 scores between paired systems
- Bonferroni correction across 5 system pairs × 3 metrics

---

## 3. Independence Fixes

### 3A. External Query Set (≥50 queries)
**Generator model**: `llama3.1:8b` (Meta/Llama, different vendor family from qwen3.6)
**Not used for labels**: generation only; labels applied by the same human-authored rubric

Prompt (committed to `eval/external_queries/generation_prompt.txt`):
```
You are a financial analyst. Write 50 diverse questions that a real investor would ask
about the following companies and their financial data. Vary question type:
- Some require only one tool (news, ratings, guidance, OR earnings)
- Some require two tools (e.g. earnings + guidance comparison)
- Some have no answer in this dataset (unknown ticker or period)
- Use natural, varied phrasing — not the same sentence structure

Companies: NVDA (Nvidia), TSLA (Tesla), JPM (JPMorgan), XOM (ExxonMobil)
Available data: news articles, analyst ratings, company guidance, earnings reports
Ticker NOT in dataset (for no-data queries): AAPL, MSFT, GOOG

Output format: JSON array of strings. One question per item. No numbering.
Do NOT reproduce any of these existing queries: [first 10 from existing set]
```

Output saved to `eval/external_queries/external_queries_raw.jsonl`.
Human review: author verifies all 50 for answerability and removes duplicates.
Gold labels applied per standard rubric by the same author (same process as 229-query set).
Final output: `eval/external_queries/external_queries.jsonl`.

**Stated limitation**: these queries were generated by a language model (llama3.1:8b),
not by a human financial analyst. They satisfy the "different model family" independence
requirement but not human authorship. Reported plainly.

### 3B. Second-Rater Label Pass (Model Rater)
**Rater model**: `llama3.1:8b` (different family from qwen3.6)
**Sample**: the same 34-query IRR sample from `irr_second_pass.py`
**Stated plainly**: rater is a language model, not a human

Prompt committed to `eval/external_queries/labeling_prompt.txt`:
```
You are a financial research assistant. Label this query for a tool-routing evaluation.

Available tools:
- search_news(query, ticker): news articles, narratives, market events
- get_ratings(ticker): analyst ratings, consensus, price targets
- get_guidance(ticker): company-issued revenue/EPS forecasts
- get_earnings(ticker): actual vs estimated quarterly results

For the query below, output ONLY a JSON object:
{
  "gold_tools": ["<tool1>", "<tool2>"],   // list of tools needed; may be empty for no-data queries
  "category": "<single_tool|dual_tool|ambiguous|no_data>",
  "rationale": "<one sentence>"
}

Query: {{QUERY}}
Ticker in our dataset: NVDA, TSLA, JPM, XOM. Others = no_data.
```

Output: `eval/external_queries/llama_irr_labels.jsonl`
Cohen's κ computed between author Pass 1 labels and llama3.1:8b labels.
Reported as "LLM second-rater (llama3.1:8b) agreement", not "IRR".

---

## 4. External Benchmark Anchor: FinanceBench

**Confirmed existing**: Islam et al., "FinanceBench: A New Benchmark for Financial
Question Answering", arXiv:2311.11944. Apache 2.0 license. 150 open-source samples
at `patronus-ai/financebench` on GitHub.

**The honest mapping problem**: FinanceBench is document-QA over SEC filings (10-K, 10-Q).
Our system uses structured fixture data (4 tickers, JSON schemas). Tool interfaces do not
map directly to SEC filing retrieval. We cannot honestly claim full comparability.

**What we will do**:
1. Download the 150-sample JSON from GitHub: `eval/financebench/financebench_oss.json`
2. Filter to questions involving NVDA, TSLA, JPM, or XOM (any period)
3. For each in-scope question, run S1 (rule planner + deterministic formatter)
4. Manual evaluation: does the answer contain the right number? Recorded in `results/financebench/`
5. Report: "X of 150 FinanceBench questions involve our 4 tickers; system answers Y correctly.
   Caveat: our fixture data covers only Q3 2024; many FinanceBench questions may reference
   different periods."

**Stated limitation**: comparison is partial, period-limited, and manually evaluated.
It serves as an external sanity check, not a state-of-the-art comparison.

---

## 5. Ablations

All ablations run on **dev split only** (n=45). Never on test.

| ID | What changes | Base system | Metric to watch |
|----|-------------|-------------|-----------------|
| A1 | Tool cap: 4 (default) vs. ∞ (unlimited) | S1 | Exact match, tool calls/query |
| A2 | Synthesis: formatter vs. LLM (qwen3.6) | S1 planner | Citation validity, faithfulness |
| A3 | Sentiment: DistilBERT vs. keyword bucket | S1 on ratings queries | get_ratings output quality |
| A4 | Planner: rule vs. LLM | Formatter synthesizer | Tool-set F1 |
| A5 | Search: FAISS semantic vs. TF-IDF | S1 on news queries | Recall, answer relevance |

A1, A5 run offline (no LLM). A2, A4 require qwen3.6. A3 requires DistilBERT weights.

---

## 6. File and Directory Structure

```
experiments/
├── config.py                    # All constants: SEED=42, model names, thresholds
├── ollama_client.py             # Thin wrapper: Ollama REST API, think=false, seed=42
├── systems/
│   ├── s1_rule_planner.py
│   ├── s2_llm_planner.py
│   ├── s3_no_tools.py
│   ├── s4_call_all.py
│   ├── s5_react.py
│   └── s6_llm_synth.py
├── metrics/
│   ├── tool_selection.py        # P/R/F1, exact match, abstention accuracy
│   ├── synthesis_quality.py     # Citation validity, numeric faithfulness, unsupported rate
│   └── latency.py               # 30-run timing harness (cold/warm)
├── ablations/
│   ├── a1_tool_cap.py
│   ├── a2_synthesis_mode.py
│   ├── a3_sentiment_mode.py
│   ├── a4_planner_type.py
│   └── a5_search_engine.py
└── generate_tables.py           # All tables + figures from results/; never hand-typed

scripts/
├── setup_ollama.sh              # Pull + verify models, write results/metadata.json
└── run_all.sh                   # One-command reproduce: setup → offline → LLM → tables

results/
├── metadata.json                # Hardware, model digests, Ollama version
├── s1_rule_planner/raw/         # One JSON per query: {query, plan, answer, metrics}
├── s2_llm_planner/raw/
├── s3_no_tools/raw/
├── s4_call_all/raw/
├── s5_react/raw/
├── s6_llm_synth/raw/
├── ablations/
│   ├── a1_tool_cap/
│   ├── a2_synthesis_mode/
│   ├── a3_sentiment_mode/
│   ├── a4_planner_type/
│   └── a5_search_engine/
├── external_queries/
├── financebench/
└── tables/                      # CSV + PNG figures, generated by generate_tables.py

eval/
└── external_queries/
    ├── README.md                # Model used, prompt, date, limitations
    ├── generation_prompt.txt    # llama3.1:8b generation prompt (committed)
    ├── labeling_prompt.txt      # llama3.1:8b label prompt (committed)
    ├── external_queries.jsonl   # 50 llama-generated queries + author-reviewed labels
    └── llama_irr_labels.jsonl   # llama3.1:8b labels for 34-query sample
```

---

## 7. Execution Order + Runtime Estimates (M4, 16 GB)

| Phase | Steps | Estimated time | Requires LLM |
|-------|-------|---------------|--------------|
| Setup | `bash scripts/setup_ollama.sh` | 10–30 min (llama3.1:8b pull ~4.7 GB) | — |
| Offline eval | S1, S4, A1, A5 on dev+test | ~5 min | No |
| External query gen | 50 queries via llama3.1:8b | ~15–25 min | Yes (llama3.1) |
| LLM-rater pass | llama3.1:8b labels 34 queries | ~10–15 min | Yes (llama3.1) |
| **Unload llama3.1:8b** | `ollama stop llama3.1:8b` | — | — |
| LLM planner (S2) | 47 test + 45 dev × 30 latency = ~120 calls | ~60–90 min | Yes (qwen3.6) |
| No-tools (S3) | 47 test queries | ~20–30 min | Yes (qwen3.6) |
| ReAct (S5) | 47 test, up to 4 LLM calls/query | ~90–150 min | Yes (qwen3.6) |
| LLM synthesis (S6) | 47 test queries | ~30–50 min | Yes (qwen3.6) |
| Ablations A2, A4 | dev split only, ~45 queries each | ~60–90 min total | Yes (qwen3.6) |
| Tables + figures | `python3 experiments/generate_tables.py` | ~2 min | No |
| **Total** | | **~5–8 hours** | — |

**One-command reproduce**: `bash scripts/run_all.sh`
(Will prompt before any LLM step if `OLLAMA_SKIP_LLM=1` is set, for offline-only runs.)

---

## 8. Tables Planned

Generated by `experiments/generate_tables.py` from `results/`. Never typed by hand.

| # | Title | Systems | Split |
|---|-------|---------|-------|
| T1 | Main tool-selection results | S1–S5 | Test (n=47) |
| T2 | Synthesis quality comparison | S1 (formatter) vs S6 (LLM) | Test (n=47) |
| T3 | Latency and efficiency | S1–S6 | 30-run timing |
| T4 | Ablation results | A1–A5 | Dev (n=45) |
| T5 | External query slice | S1 | Ext (n=50) |
| T6 | FinanceBench anchor | S1 | FB in-scope subset |
| T7 | Second-rater agreement | llama3.1:8b vs author Pass 1 | n=34 sample |

---

## 9. Figures Planned

| # | Type | Content |
|---|------|---------|
| F1 | Bar chart + CI error bars | Tool-set F1 by system (T1) |
| F2 | Grouped bar | Citation validity vs numeric faithfulness (T2) |
| F3 | Box plot | Latency cold vs warm by system |
| F4 | Confusion matrix | Predicted vs gold tool set, S1 and S2 |
| F5 | Category breakdown | Pass/fail rate by query category per system |

---

## 10. Pre-committed Policy on Weak Results

| Scenario | Action |
|----------|--------|
| LLM planner (S2) ≤ rule planner (S1) on F1 | Report as-is; update audit.md; reframe paper around benchmark + synthesis design |
| LLM synthesis (S6) has similar faithfulness to formatter (S1) | Report; strengthens "by-construction" claim as design choice, not just performance |
| call-all (S4) ≈ rule planner (S1) on F1 | Report; indicates fixture set is too narrow to differentiate planners (4 tickers, no conflicts) |
| ReAct (S5) loops or fails JSON parse | Report loop rate; count fallbacks; never suppress |
| FinanceBench: 0 in-scope questions | Report honestly; remove table; note fixture-QA mismatch |
| llama IRR κ < 0.6 | Report; weakens benchmark validity claim; update audit.md |

---

## 11. Step A Status (already completed)

- ✅ `git diff aa751b8 HEAD -- README.md`: Only cosmetic deletions remain (hardcoded example answer text replaced with `"..."`). Substantive content (LLM_PROVIDER note, 4 Tools table) restored in commit `8942138`.
- ✅ IRR relabeled as "label self-consistency check" in README and `eval/gold_label_agreement.md`.
- ✅ Bootstrap 95% CIs added: Tool-set κ=0.9671 [0.90, 1.00]; Category κ=0.9183 [0.80, 1.00].
- ✅ `WEDDING PROJECT/` added to `.gitignore`.
- All committed and pushed; remote SHA confirmed.
