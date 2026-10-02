# Phase 2 — Experiment Plan
**Status: AWAITING APPROVAL — nothing that costs API money will run until approved.**

---

## 0. Governing Constraints

- **Test split is sacred**: the 47-query held-out test split (`eval/queries_test.jsonl`) is touched
  exactly once per system, at final reporting time. All tuning and threshold selection is done
  on train (n=137) + dev (n=45) only.
- **No API calls run without approval**: LLM-planner and LLM-synthesis experiments are listed
  here but will not execute until the user says go. Cost estimates are provided per experiment.
- **Fixed seeds everywhere**: `SEED=42` for all random operations; temperature=0 for all LLMs.
- **Raw outputs always saved first**: `results/<experiment_name>/raw/` before any aggregation.
- **If a result is weak or contradicts the paper**: update `paper/audit.md` and report as-is.
  The experiment harness does not filter or retry to improve numbers.

---

## 1. Systems Under Test

| ID | Name | Description | Cost/47 queries |
|----|------|-------------|-----------------|
| S1 | `rule-planner` | Rule-based planner + deterministic formatter (current default) | Free |
| S2 | `llm-planner-groq` | Groq `llama-3.3-70b-versatile` planner + deterministic formatter | ~$0.00 (free tier) |
| S3 | `no-tools-llm` | Single-shot LLM with NO tool calls (pure parametric knowledge) | ~$0.01–0.05 |
| S4 | `call-all-tools` | Always call all 4 tools regardless of query (max-recall baseline) | Free |
| S5 | `react-agent` | ReAct-style loop: think → act → observe, capped at 4 steps | ~$0.05–0.15 |
| S6 | `llm-planner+llm-synth` | Groq planner + Groq LLM synthesis (tests grounding claim) | ~$0.05–0.15 |

**Models pinned:**
- LLM planner: `groq/llama-3.3-70b-versatile` temperature=0 (already tested in the codebase)
- LLM synthesis: `groq/llama-3.3-70b-versatile` temperature=0
- No-tools single-shot: `groq/llama-3.3-70b-versatile` temperature=0
- ReAct agent: `groq/llama-3.3-70b-versatile` temperature=0

**Rationale for Groq only**: free tier, reproducible without credit card, consistent with prior
8-query comparison. Anthropic Claude can be added later if the user provides a key.

**S5 (ReAct)**: Implement a simple ReAct loop inside `experiments/react_agent.py`:
- System prompt: "Think step-by-step. At each step, either call a tool or produce a final answer."
- Loop: parse `Action:` / `Action Input:` / `Observation:` triples
- Cap at MAX_TOOL_CALLS=4 (same constraint as S1)
- This is NOT a full ToolFormer or complex framework — it is a minimal ReAct that demonstrates
  the contrast with single-shot planning

---

## 2. Evaluation Metrics

### 2A. Tool-Selection (planner) — primary metric
Evaluated per query against gold `gold_tools` label. For systems S1, S2, S4, S5:

| Metric | Formula | Notes |
|--------|---------|-------|
| Precision | TP / (TP+FP) | Per-query micro-averaged |
| Recall | TP / (TP+FN) | Per-query micro-averaged |
| F1 | 2PR/(P+R) | Harmonic mean |
| Exact-set match | 1 if tools==gold else 0 | Strict |
| Tool calls / query | mean(len(plan)) | Efficiency |
| Abstention accuracy | Correct "no tools needed" on no_data queries | Robustness |

**No-tools baseline (S3) has P=R=F1=0 by definition for tool-selection** — reported to anchor the table.

### 2B. Synthesis Quality — the core claim
Evaluated per answer for all systems:

| Metric | Method | What it tells us |
|--------|--------|-----------------|
| Citation validity | Script: does every `[Source: X]` in the answer have X in tool_outputs? | Grounding accuracy |
| Numeric faithfulness | Script: extract all numbers from answer, check each against source field ±0.5% | Hallucination rate |
| Unsupported-claim rate | Script: sentences with no citation / all sentences | Grounding coverage |
| Abstention correctness | On no_data queries: does system say "no data"? | Safety |
| Answer length (tokens) | Proxy for verbosity | Efficiency |

**Critical framing for the paper**: the deterministic formatter has 100% citation validity
*by construction*. The paper claim is NOT that this is an achievement — it's that this is a
design property worth measuring. The real contribution is the **comparison**: LLM synthesis
path's citation validity vs. the formatter's, and the numeric faithfulness gap.

### 2C. Latency & Cost
- 30 warm runs per system after 2 cold-start warm-up runs
- Cold: first call with process restart
- Warm: subsequent calls without restart
- Report: mean ± std (ms), cold/warm separated, API calls counted
- Cost: tokens in + tokens out for LLM systems at Groq published rates

### 2D. Bootstrap 95% CIs + Significance Tests
- Bootstrap B=10,000, SEED=42 on all primary metrics
- Paired McNemar's test for exact-set match between S1 and S2
- Wilcoxon signed-rank on F1 scores between paired systems
- Bonferroni correction for multiple comparisons (6 systems × 3 primary metrics)

---

## 3. Independence Fixes

### 3A. External Query Set (50+ queries)
**Source**: Generate 50 queries using a different model family from the development stack.

**Plan**:
1. Use `claude-3-5-sonnet-20241022` (Anthropic, different family from Groq/Llama used in development)
   to generate 50 queries about the same fixture data.
   - Prompt: "You are a financial analyst. Write 50 diverse questions about these companies
     that a real investor would ask. Vary: single-tool, multi-tool, no-data, adversarial phrasing.
     Do not repeat queries from this list: [existing queries]. Return as JSON."
   - Inject fixture summaries (not the queries) to ground questions in reality
2. Manually verify all 50 for answerability against fixtures
3. Label gold tools (same rubric) and record generation model + prompt in
   `eval/external_queries/README.md`
4. Reserve as a separate evaluation slice: report separately, never mix with the 229-query set

**Cost estimate**: ~$0.05–0.10 for Claude Sonnet 3.5 at current rates.
**Requires**: `ANTHROPIC_API_KEY` — user must confirm before running.

### 3B. Real Second Rater for a Labeled Subset
For the gold-label self-consistency claim, we need at least one genuinely independent pass.

**Plan**:
1. Take the 34-query IRR sample already used
2. Use `claude-3-5-sonnet-20241022` as the second rater with a fully documented prompt
   (prompt will be committed to `eval/external_queries/irr_prompt.txt`)
3. Compute Cohen's κ between the author's Pass 1 labels and Claude's labels
4. Report as "LLM second-rater agreement" — honest about it being a different model, not a human
5. If budget allows, note that human annotation via MTurk/Prolific would be the gold standard

**Cost estimate**: ~$0.02–0.05.
**Requires**: `ANTHROPIC_API_KEY`.

---

## 4. External Benchmark Anchor: FinanceBench

**Confirmed**: FinanceBench by Islam et al. (arXiv:2311.11944), Apache 2.0 license,
150 open-source samples at `patronus-ai/financebench` on GitHub.

**The honest mapping problem**: FinanceBench is a document-QA benchmark (SEC filings).
Our system uses structured fixture data (4 tickers), not SEC PDFs. The tool interfaces
do NOT map directly. We cannot evaluate our system on FinanceBench at face value.

**What we CAN do honestly**:
1. Download the 150 FinanceBench samples
2. Identify any questions that could be answered by our 4 fixture tickers
   (NVDA, TSLA, JPM, XOM) and our tool schemas
3. Report: "X of 150 FinanceBench questions are in-scope for our tool set; our
   system answers Y correctly."
4. Acknowledge: "Our fixture data does not contain the full SEC filing detail needed
   for most FinanceBench questions. This comparison is bounded and partial."

**Implementation**: `experiments/financebench_slice.py` — download dataset, filter to
in-scope questions, run S1, evaluate manually (the system cannot auto-grade against
FinanceBench gold answers without humans in the loop).

**Cost**: Free (rule planner) + human review time.

---

## 5. Ablations

| Ablation ID | What changes | Control | Why |
|-------------|-------------|---------|-----|
| A1 | Tool cap: 4 (default) vs. ∞ (unlimited) | S1 rule planner | Does cap cause real misses? |
| A2 | Grounding: formatter vs. LLM synthesis | Same planner (S1) | The core synthesis claim |
| A3 | Sentiment: ML (DistilBERT) vs. keyword | get_ratings output | Is ML scoring worth it? |
| A4 | Planner: rule vs. LLM | Same synthesizer (formatter) | Planner contribution |
| A5 | Search: FAISS semantic vs. TF-IDF | search_news only | Does FAISS help over TF-IDF? |

A5 is runnable offline (no API cost) by setting `NEWS_SEARCH_ENGINE=tfidf`.
A1 is runnable offline.
A2 requires LLM API for the synthesis half.
A3 requires DistilBERT weights (`python3 classifier/train.py` first).
A4 requires LLM API for the planner half.

---

## 6. File Structure & Reproducibility

```
experiments/
├── run_all.sh                   # One-command reproduce (sequential, no parallelism)
├── config.py                    # SEED=42, model names, temperatures, all constants
├── systems/
│   ├── s1_rule_planner.py       # S1: rule planner + formatter
│   ├── s2_llm_planner.py        # S2: Groq planner + formatter
│   ├── s3_no_tools.py           # S3: single-shot LLM, no tools
│   ├── s4_call_all.py           # S4: always call all 4 tools
│   ├── s5_react.py              # S5: ReAct-style loop
│   └── s6_llm_synth.py          # S6: Groq planner + Groq synthesis
├── metrics/
│   ├── tool_selection.py        # Precision/recall/F1, exact match, abstention
│   ├── synthesis_quality.py     # Citation validity, numeric faithfulness, unsupported rate
│   └── latency.py               # 30-run timing harness
├── ablations/
│   ├── a1_tool_cap.py
│   ├── a2_synthesis_mode.py
│   ├── a3_sentiment_mode.py
│   ├── a4_planner_type.py
│   └── a5_search_engine.py
├── react_agent.py               # ReAct loop implementation
└── financebench_slice.py        # FinanceBench in-scope analysis

results/
├── s1_rule_planner/raw/         # One JSON per query
├── s2_llm_planner/raw/
├── ... (one dir per system)
├── ablations/
└── tables/                      # Generated by generate_tables.py

eval/
└── external_queries/
    ├── README.md                # Model, prompt, date
    ├── irr_prompt.txt           # Claude second-rater prompt (committed)
    ├── external_queries.jsonl   # 50 externally generated queries + labels
    └── claude_irr_labels.jsonl  # Claude's labels for 34-query sample
```

---

## 7. Execution Order & API Cost Summary

| Step | Runs offline? | Requires key | Est. cost | Approval needed |
|------|--------------|--------------|-----------|-----------------|
| S1 full eval (47 queries) | ✅ Yes | — | $0 | Auto-approved |
| S4 call-all (47 queries) | ✅ Yes | — | $0 | Auto-approved |
| A1 tool cap ablation | ✅ Yes | — | $0 | Auto-approved |
| A3 sentiment ablation | ✅ After train | — | $0 | Auto-approved |
| A5 search engine ablation | ✅ Yes | — | $0 | Auto-approved |
| S2 LLM planner (47+dev) | ❌ | GROQ_API_KEY | ~$0 (free tier) | **Needs approval** |
| S3 no-tools (47 queries) | ❌ | GROQ_API_KEY | ~$0 (free tier) | **Needs approval** |
| S5 ReAct (47 queries) | ❌ | GROQ_API_KEY | ~$0.05–0.15 | **Needs approval** |
| S6 LLM synthesis (47 queries) | ❌ | GROQ_API_KEY | ~$0.05–0.15 | **Needs approval** |
| External query gen (50) | ❌ | ANTHROPIC_API_KEY | ~$0.05–0.10 | **Needs approval** |
| Claude IRR second rater (34) | ❌ | ANTHROPIC_API_KEY | ~$0.02–0.05 | **Needs approval** |
| FinanceBench slice | ✅ (S1) | — | $0 | Auto-approved |

**Total worst-case cost**: ~$0.30–0.50 for all LLM experiments.
**Groq free tier**: $0 if within rate limits (likely for 47 queries).

---

## 8. Tables to Generate

All generated by script from `results/`, not hand-typed.

**Table 1**: Main results — tool-selection P/R/F1, exact match, abstention accuracy, tool calls/query (S1–S5 on 47-query test set)

**Table 2**: Synthesis quality — citation validity %, numeric faithfulness %, unsupported-claim rate %, abstention correctness (S1 formatter vs. S6 LLM synthesis, same planner)

**Table 3**: Latency & cost — mean±std ms cold/warm, tool calls, est. $/1000 queries

**Table 4**: Ablation results — each ablation vs. its control on dev set (not test)

**Table 5**: External anchor — FinanceBench in-scope slice results (partial, honest)

---

## 9. Figures to Generate

**Figure 1**: Tool-set F1 bar chart with 95% CI error bars — S1 through S5

**Figure 2**: Synthesis quality grouped bar — citation validity vs. numeric faithfulness for formatter vs. LLM paths

**Figure 3**: Latency violin plots — cold vs. warm, offline vs. online systems

**Figure 4**: Confusion matrix — predicted vs. gold tool set for S1 and S2

---

## 10. What to Do If Results Are Weak

Pre-commit policy:
- If S2 (LLM planner) does NOT outperform S1 (rule planner): report the result, update
  audit.md ("LLM planner did not improve over rule-based on the test set"), and frame the
  paper contribution as the benchmark + synthesis design pattern, not planner comparison.
- If LLM synthesis (S6) has similar or worse numeric faithfulness than formatter (S1):
  report it and strengthen the "grounded-by-construction" claim.
- If call-all-tools (S4) achieves similar F1 to rule planner: report it — this is evidence
  that our fixture data is too narrow (4 tickers, no ticker conflicts) to differentiate planners.
- If FinanceBench slice shows 0 answerable questions: report it honestly.

---

## Approval Request

**Ready to run without any API calls** (free, offline):
- S1, S4 full eval on 47-query test set
- A1 (tool cap), A5 (search engine) ablations
- FinanceBench slice (download + filter, S1 eval)
- Metric harness development (tool_selection.py, synthesis_quality.py, latency.py)
- results/ directory structure, generate_tables.py scaffold

**Waiting for go-ahead before ANY API call**:
- S2, S3, S5, S6 (require GROQ_API_KEY)
- External query generation + Claude IRR rater (require ANTHROPIC_API_KEY)
- A2, A4 ablations (require LLM API)

**Questions for the user before starting**:
1. Do you have a GROQ_API_KEY available? (Free at console.groq.com)
2. Do you have an ANTHROPIC_API_KEY for external query generation and Claude IRR?
3. Should I start with the free/offline experiments immediately, or wait for full approval?
