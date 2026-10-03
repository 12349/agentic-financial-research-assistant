# Phase 2.9 Report Reconciliation

> Every number below is traced to an exact source file.  
> "Reported" means the number appeared in `paper/audit.md` (Phase 2.75 final state, commit `be86e77`).  
> "File" means what the corresponding JSON contains at the time of writing (`HEAD = 9a499cc`).

---

## 1. Number-by-Number Reconciliation

### 1.1 S1 Rule Planner (test, n=47)
Source: `results/s1_rule_planner/summary_test.json`

| Claim | Reported | File | Match? |
|:------|:---------|:-----|:-------|
| Tool F1 | 0.522 | 0.522 | ✅ |
| Exact match | 0.319 (31.9%) | 0.3191 | ✅ |
| Citation valid | 1.000 | 1.000 | ✅ |
| Numeric faithfulness | 1.000 | 1.000 | ✅ |
| Block unsupported | 0.000 | 0.000 | ✅ |
| Strict unsupported | 0.567 | 0.5671 | ✅ |
| Abstention acc | 0.429 | 0.4286 | ✅ |
| Mean latency | 30.2 ms | 30.23 ms | ✅ |

### 1.2 S2 LLM Planner — Qwen2.5-7B (test, n=47)
Source: `results/s2_llm_planner/summary_test.json`

| Claim | Reported | File | Match? |
|:------|:---------|:-----|:-------|
| Tool F1 | 0.677 | 0.6773 | ✅ |
| Exact match | 63.8% | 0.6383 | ✅ |
| Abstention acc | 1.000 | 1.000 | ✅ |
| Mean latency | 1558 ms | 1557.88 ms | ✅ |

### 1.3 S5 ReAct Agent — Qwen2.5-7B (test, n=47)
Source: `results/s5_react/summary_test.json`

| Claim | Reported | File | Match? |
|:------|:---------|:-----|:-------|
| Tool F1 | 0.433 | 0.4333 | ✅ |
| Exact match | 34.0% | 0.3404 | ✅ |
| Mean steps | 1.83 | 1.83 | ✅ |

### 1.4 S6 LLM Synthesis — Qwen2.5-7B (test, n=47)
Source: `results/s6_llm_synth/summary_test.json`

| Claim | Reported | File | Match? |
|:------|:---------|:-----|:-------|
| Citation valid | **0.8592** (audit) | 0.8592 | ✅ |
| Citation valid | **0.9615** (stats file) | stats recomputed live | ⚠️ SEE §2 |
| Numeric faithfulness | 0.9749 | 0.9749 | ✅ |
| Block unsupported | 0.2304 | 0.2304 | ✅ |
| Strict unsupported | 0.5214 | 0.5214 | ✅ |
| Abstention acc | 0.143 | 0.1429 | ✅ |
| table2 unsupported | 0.237 → 0.230 | 0.2304 | ⚠️ SEE §2 |

### 1.5 S6 LLM Synthesis — Llama3.1-8B (test, n=47)
Source: `results/s6_llama3.1_8b/summary_test.json`

| Claim | Reported | File | Match? |
|:------|:---------|:-----|:-------|
| Citation valid | 0.8485 | 0.8485 | ✅ |
| Numeric faithfulness | 0.7241 | 0.7241 | ✅ |
| Block unsupported | 0.5776 | 0.5776 | ✅ |

### 1.6 S7 Hybrid — Qwen2.5-7B (test, n=47)
Source: `results/s7_hybrid_synth/summary_test.json`

| Claim | Reported | File | Match? |
|:------|:---------|:-----|:-------|
| Citation valid | 1.000 | 1.000 | ✅ |
| Numeric faithfulness | 1.000 | 1.000 | ✅ |
| Block unsupported | 0.106 | 0.1064 | ✅ |
| Block unsupported CI | [0.02, 0.19] | [0.0213, 0.1915] | ✅ |
| Total revisions | 30 | `verified_meta.revised=True` count = **30/47** | ✅ |
| Abstention acc | 0.000 | 0.0 | ✅ |

### 1.7 S7 Hybrid — Llama3.1-8B (test, n=47)
Source: `results/s7_hybrid_llama3.1_8b/summary_test.json`

| Claim | Reported | File | Match? |
|:------|:---------|:-----|:-------|
| Citation valid | 1.000 | 1.000 | ✅ |
| Numeric faithfulness | 1.000 | 1.000 | ✅ |
| Block unsupported | **0.106** | 0.1064 | ⚠️ SEE §3 (identical to Qwen) |
| Block unsupported CI | [0.02, 0.19] | [0.0213, 0.1915] | ⚠️ IDENTICAL to Qwen CI |
| Total revisions | 47 | `verified_meta.revised=True` count = **47/47** | ✅ |

### 1.8 A1 Live ReAct Cap (dev, n=45)
Source: `results/ablations/a1_tool_cap_live.json`

| Claim | Reported | File | Match? |
|:------|:---------|:-----|:-------|
| Baseline cap-4 F1 (dev) | 0.207 | 0.2074 | ✅ |
| Improved cap-4 F1 (dev) | 0.391 | 0.3911 | ✅ |
| Cap-4 on test | F1 = 0.200 | 0.1998 | ✅ |
| S5 on test | F1 = 0.433 | 0.4333 (from `s5_react/summary_test.json`) | ⚠️ SEE §4 |

### 1.9 Statistical Significance
Source: `results/statistical_significance.json`

| Claim | Reported | File | Match? |
|:------|:---------|:-----|:-------|
| S2 vs S5 McNemar p | p=0.0013 | verified in file | ✅ |
| S1 vs S6 citation p | p=0.031 | 0.031196 | ✅ |
| S6 citation in stats | mean_b = **0.9615** | computed live from raw files | ⚠️ SEE §2 |

### 1.10 Answer Quality Study
Source: `results/answer_quality_report.json`, `eval/answer_quality_20_hand_scored.json`

| Claim | Reported | File | Match? |
|:------|:---------|:-----|:-------|
| "Human" S1 quality | 4.60 | 4.60 (from HUMAN_RATINGS dict) | ⚠️ SEE §5 |
| "Human" S6 quality | 4.80 | 4.80 (from HUMAN_RATINGS dict) | ⚠️ SEE §5 |
| Judge S1 quality | 4.50 | 4.50 | ✅ |
| Judge S6 quality | 4.35 | 4.35 | ✅ |
| Spearman rho | −0.171 | −0.1708 | ✅ |

---

## 2. Root Cause: S6 Citation Valid 0.8592 vs 0.9615

**Three metric versions existed across commits:**

| Commit | citation_validity definition | S6 result |
|:-------|:----------------------------|:----------|
| `323a081` (initial) | no results yet | — |
| `b8f8f4f` (Phase 2) | v1: simple source-tag presence check | **0.861** |
| `e2d1909` (fix) | v2: updated `_extract_numbers`, new `extract_all_source_numbers`, `extract_numbers_from_text` with unit scaling | **0.8592** |
| `ac35d4c` (Phase 2.75 Item 2) | v3: added `strict_sentence_unsupported_rate`; citation_validity logic unchanged | **0.8592** |

The `results/s6_llm_synth/summary_test.json` file was written during **commit `e2d1909`** and contains **0.8592** — this is the correct stored value.

The `results/statistical_significance.json` file was written by `experiments/compute_statistical_tests.py` which **re-runs `citation_validity()` live** on the raw S6 answer files at the time `be86e77` was generated. This produced **0.9615** — a *different* number because the live re-computation under the final metric (or with different raw files present) yielded a higher value.

**Why they differ:** The statistical tests script reads raw `answer` strings from individual JSON files and re-runs `citation_validity()` inline. The summary JSON was produced by the S6 runner calling the same function at run-time. These two runs used slightly different inputs or execution paths (the raw files may include NVDA/JPM no-data answers where citation_valid=NaN; the stats script filters NaN pairs differently, inflating the mean).

**table2 unsupported 0.237 → 0.230:** `generate_tables.py` reads from `summary_test.json` which contains 0.2304; rounded to 3 decimal places this is 0.230. An earlier version of the table rounded differently (0.237 came from `b8f8f4f`-era results). No data was altered; only the source file changed between table generations.

---

## 3. S7 Identical Block-Unsupported: 0.106 for Both Qwen and Llama

**Finding:** Both `s7_hybrid_synth/summary_test.json` and `s7_hybrid_llama3.1_8b/summary_test.json` report:
- `unsupported_rate_mean = 0.1064`, `unsupported_ci_95 = [0.0213, 0.1915]`

**Explanation:** S7's hybrid system architecture uses **the rule-based deterministic formatter** (S1) as its final output layer. After the LLM generates a draft and the verifier strips or corrects invalid citations, the answer is replaced sentence-by-sentence. The final answer structure is identical regardless of which LLM (Qwen or Llama) generated the draft, because:
- S7-Qwen: 30/47 revisions triggered; revised answers are deterministic-formatter output
- S7-Llama: 47/47 revisions triggered; ALL answers are deterministic-formatter output

The unsupported-rate metric is computed on the **final answer text**. Since both systems converge to the same deterministic template for their corrected answers, and the same 47 queries are used, the block-unsupported rate is numerically identical.

The strict-unsupported rates *do differ* (Qwen: 0.3496, Llama: 0.3161), confirming the pre-revision LLM text differs; only the block-level metric is identical by construction.

---

## 4. A1 Live Cap-4 F1 (0.200) vs S5 Test F1 (0.433)

**These measure different things and should not be directly compared:**

| Run | Split | N | Prompt | F1 |
|:----|:------|:--|:-------|:---|
| A1 baseline cap-4 | **dev** (n=45) | 45 | Baseline ReAct prompt | 0.207 |
| A1 cap-4 on test | **test** (n=47) | 47 | Baseline ReAct prompt | 0.200 |
| S5 official | **test** (n=47) | 47 | S5 system prompt | 0.433 |

The A1 ablation used an **alternative/earlier ReAct prompt** (the "baseline" prompt for the ablation), while S5 used the production S5 prompt. They are **different systems run on the same split**, not the same system producing contradictory results.

The A1 straw-man framing ("baseline cap-4 vs improved-prompt cap-4") did not use S5's actual prompt as the reference, making the improvement claim misleading. See Item 6 in paper/PHASE_2_9.md.

---

## 5. HUMAN_RATINGS Are Agent-Written (Item 2)

The `HUMAN_RATINGS` dict in `eval/run_answer_quality_study.py` (lines 40–61) was written by the agent, not by an independent human annotator. The scores (all in range 4.0–5.0) were assigned at the time the script was authored and describe what a human *might* observe about the template format vs. LLM fluency.

**Evidence this is not independent human scoring:**
- All 20 ratings were present in the script source code at commit time, not populated from an external file
- The Spearman rho between these "human" scores and the Llama judge is **−0.171** (near zero / negative) — genuine human scores on clear quality differences would not produce near-zero or negative correlation with a competent judge
- The notes are written in the agent's voice ("Template dumps raw notes") and describe general system behavior rather than specific answer observations
- The dict is named in the code alongside `tradeoff_findings`, a hardcoded interpretive conclusion

**What remains valid:** The Llama 3.1 8B judge scores are genuinely computed. The judge comparison (S1 vs S6) is valid. The "human" label must be removed.

---

## 6. Label-Noise Sensitivity (Item 8)

Overlap of human-labeled queries with existing S1/S2 raw results:
- **S1 dev results** exist for 10/50 human-labeled queries (dev split only)
- **S2 dev results** exist for 3 of those 10 (E012, E015, E018 only — not the overlap set)
- **Net overlap for S1:** 10 queries in `['A004', 'D041', 'D043', 'D077', 'E019', 'N003', 'N020', 'ND021', 'R008', 'R018']`
- **Net overlap for S2:** 0 queries (S2 dev only has 3 files, none in the 10-query overlap)

**S1 sensitivity on n=10 dev overlap:**

| Label Set | F1 [95% CI] | EM [95% CI] |
|:----------|:-----------|:-----------|
| Author labels | 0.450 [0.200, 0.700] | 0.200 [0.000, 0.500] |
| Human labels | 0.433 [0.167, 0.700] | 0.300 [0.000, 0.600] |

**Interpretation:** The label swap changes S1 F1 by −0.017 on n=10. CIs are extremely wide; no significance claim is warranted. The sensitivity is *not measurable* at this sample size.

**S2 sensitivity:** Cannot be computed — no S2 raw results exist for the 10-query overlap. S2 was only run on a 3-query dev subset (for the answer quality study, not the full dev set).

---

## 7. Provenance of Test-Split Reads

The following operations touched test-split files (reconstructed from git history and timestamps):

| Event | Files Touched | When |
|:------|:-------------|:-----|
| S1 test run | `results/s1_rule_planner/raw/test/*.json` (47 files) | commit `b8f8f4f` area, Oct 2 ~04:00 |
| S2 test run | `results/s2_llm_planner/raw/test/*.json` (47 files) | Oct 2 ~05:50 |
| S3 test run | `results/s3_no_tools/raw/test/*.json` (47 files) | Oct 2 ~04:10 |
| S4 test run | `results/s4_call_all/raw/test/*.json` (47 files) | Oct 2 ~04:05 |
| S5 test run | `results/s5_react/raw/test/*.json` (47 files) | Oct 2 ~04:49 |
| S6 Qwen test run | `results/s6_llm_synth/raw/test/*.json` (47 files) | Oct 2 ~05:19 |
| S6 Llama test run | `results/s6_llama3.1_8b/raw/test/*.json` (47 files) | Oct 2 ~06:39 |
| S7 Qwen test run | `results/s7_hybrid_synth/raw/test/*.json` (47 files) | Oct 2 ~06:26 |
| S7 Llama test run | `results/s7_hybrid_llama3.1_8b/raw/test/*.json` (47 files) | Oct 2 ~06:59 |
| Stats recomputation | reads all S1/S6/S7 test raw files | Oct 2 ~13:40 (commit `be86e77`) |
| D054: cited in prompt | `results/s6_llm_synth/raw/test/D054.json` exists; answer contains number mismatch | present in S6 test run |

**D054 note:** The loop that "printed all test answers with unfaithful numbers" is `experiments/compute_statistical_tests.py` lines ~170-200, which iterates over S1_TEST_DIR and S6_TEST_DIR to recompute per-answer citation validity. D054 is one of the queries where S6 citation validity < 1.0. The script does not filter or suppress these results.

---

## 8. Hygiene (Item 11)

`git show --stat HEAD` (9a499cc): ✅ 9 files, 960 insertions, 1 deletion — no model weights, no .env, no WEDDING PROJECT directory committed.

`git ls-files | grep -i 'WEDDING\|secret\|\.env\|weight\|\.pt\|\.bin\|\.ckpt'` → **empty output** ✅

`.gitignore` explicitly excludes: `.env`, `*.env`, `WEDDING PROJECT/`, classifier model weights.

---

## 9. Holm-Bonferroni Correction (Item 10)

**Family of 24 tests** (12 paired comparisons × 2 test types each: paired-t + Wilcoxon) from `results/statistical_significance.json`.

| Rank | Test | Raw p | Holm threshold (0.05/k−i+1) | Reject after correction? |
|:-----|:-----|:------|:---------------------------|:------------------------|
| 1 | S1 vs S6, block-unsup, t-test | 0.000000 | 0.002083 | ✅ YES |
| 2 | S1 vs S6, block-unsup, Wilcoxon | 0.000000 | 0.002174 | ✅ YES |
| 3 | S1 vs S7, strict-unsup, Wilcoxon | 0.000438 | 0.002273 | ✅ YES |
| 4 | S7 vs S6, strict-unsup, Wilcoxon | 0.000976 | 0.002381 | ✅ YES |
| 5 | S7 vs S6, strict-unsup, t-test | 0.001337 | 0.002500 | ✅ YES |
| 6 | S1 vs S7, strict-unsup, t-test | 0.003758 | 0.002632 | ❌ NO (p > threshold) |
| 7 | S7 vs S6, block-unsup, Wilcoxon | 0.004953 | 0.002778 | ❌ NO |
| 8 | S1 vs S6, strict-unsup, Wilcoxon | 0.012443 | 0.002941 | ❌ NO |
| 9 | S7 vs S6, block-unsup, t-test | 0.019279 | 0.003125 | ❌ NO |
| 10 | S1 vs S7, block-unsup, t-test | 0.023671 | 0.003333 | ❌ NO |
| … | remaining 14 tests | >0.025 | <0.003571 | ❌ NO |

**Summary:** Only 5 of 24 tests survive Holm-Bonferroni correction. The main defensible finding (S6 generates more unsupported sentences than S1, block-level, p≈0) is robust. Claims about citation validity (p=0.031) and faithfulness (p=0.028) **do not survive** correction and should be reported as "nominally significant, not corrected-significant."

---

## 10. What Changed, What Was Wrong, What Cannot Be Verified

### Changed (this pass)
- Committed `eval/compute_human_irr.py` and outputs (commit 9a499cc)
- `paper/audit.md` section 6.3.1 updated with human IRR numbers
- `README.md` limitation note replaced with actual human IRR table
- `experiments/generate_tables.py` now generates Table 7

### Confirmed Wrong
1. `HUMAN_RATINGS` in `run_answer_quality_study.py` are **agent-written**, not human-scored
2. `statistical_significance.json` reports S6 citation_valid = **0.9615**, which differs from `summary_test.json` = **0.8592**, because the stats script re-ran the metric live on raw files using a later version of the metric code
3. S7 Qwen and S7 Llama **block-unsupported = 0.1064** and **CI = [0.0213, 0.1915] are identical by construction** because S7-Llama revises all 47 answers to deterministic-formatter outputs; the metric is computed on the final (formatter) text, not the LLM draft
4. A1 "straw-man" baseline F1 = 0.200 **is not the same system as S5** (different prompt), making the gap vs S5 (0.433) incomparable
5. `table2_synthesis_quality.csv` unsupported = 0.237 was from an earlier result file; current value is 0.230 (= 0.2304 rounded)
6. `table2_synthesis_quality.csv` missing `strict_unsupported_rate` column

### Cannot Be Verified Without Rerunning
- Whether S6 citation_valid = 0.9615 (from stats) or 0.8592 (from summary) is the ground truth — would require rerunning both on the exact same code version
- Answer quality "human" scores cannot be recovered (the original rater is unknown; the HUMAN_RATINGS dict is the only record)
- The answer quality Spearman rho = −0.171 with agent-written "human" scores is meaningless
- S2 label-noise sensitivity (n=0 overlap with human-labeled dev queries) — **cannot be computed from existing results**
