# PROVENANCE.md

> Phase 2.95 test-file access log and data provenance.
> All numbers in the paper must be copied from files listed here.
> Every result file produced by the Phase 2.95 pipeline includes
> `git_commit` and `metric_blob_hash` fields written by
> `experiments/systems/common.py:save_system_results()` at run time.

---

## 1. Pipeline Freeze Guarantee (Phase 2.95 Item 1)

`scripts/run_all.sh` refuses to execute if `git status --porcelain` is non-empty.
Every summary JSON written by `save_system_results()` contains:

```json
{
  "git_commit": "<HEAD SHA at run time>",
  "metric_blob_hash": "<git hash-object of synthesis_quality.py at run time>"
}
```

These are stamped by code, never by hand.
The Phase 2.9 hand-injected `metric_blob_hash` fields have been removed from all
pre-existing summary JSONs (they were injected post-hoc into files that were not
produced under a clean tree). This removal is documented in commit `3aa7844` +
subsequent revert in Phase 2.95.

---

## 2. Phase 2.9 Gate Script Disclosure (Item 9)

In Phase 2.9, `experiments/abstention_gate.py` called `evaluate_on_no_data(ROOT / "eval" / "queries_test.jsonl")`.
This function read and printed test-split `no_data` query texts and per-query gate decisions.

**Test queries inspected by the gate script:**
- ND003: "What is Amazon's forward revenue guidance?"
- ND004: "Any news about Alphabet's antitrust case?"
- ND009: "What dividend does Exxon Mobil pay?"
- ND024: "Give me Google's analyst ratings and price targets."
- ND025: "What is the options flow on NVDA calls expiring next Friday?"
- ND028: "Show me JPMorgan's credit default swap spreads."
- ND031: "What's the news on Coinbase's regulatory battles?"

The gate was NOT fitted to these queries (it used only a pre-specified KNOWN_TICKERS
and COMPANY_NAMES set). However, the gate designer (the agent) observed these test
query texts before the overnight rerun in Phase 2.95.

**Consequence:** The Phase 2.95 abstention analysis (Item 8) relies on DEV-only
development, with no further inspection of test no_data queries. The 7 test queries
listed above are considered contaminated for any gate-design purpose.
The gate itself is reported as "0/7, below S1 baseline 3/7" and NOT applied.

---

## 3. Test-Split File Access Log

### Phase 2.75 original runs (commit `be86e77`, Oct 2 2026)

| System | Raw files | Timestamp |
|:-------|:----------|:----------|
| S1 test | `results/s1_rule_planner/raw/test/*.json` (47 files) | Oct 2 ~04:00 |
| S2 Qwen test | `results/s2_llm_planner/raw/test/*.json` (47 files) | Oct 2 ~05:50 |
| S3 test | `results/s3_no_tools/raw/test/*.json` (47 files) | Oct 2 ~04:10 |
| S4 test | `results/s4_call_all/raw/test/*.json` (47 files) | Oct 2 ~04:05 |
| S5 Qwen test | `results/s5_react/raw/test/*.json` (47 files) | Oct 2 ~04:49 |
| S6 Qwen test | `results/s6_llm_synth/raw/test/*.json` (47 files) | Oct 2 ~05:19 |
| S6 Llama test | `results/s6_llama3.1_8b/raw/test/*.json` (47 files) | Oct 2 ~06:39 |
| S7 Qwen test | `results/s7_hybrid_synth/raw/test/*.json` (47 files) | Oct 2 ~06:26 |
| S7 Llama test | `results/s7_hybrid_llama3.1_8b/raw/test/*.json` (47 files) | Oct 2 ~06:59 |
| Stats recompute | reads S1/S6/S7 test raw files | Oct 2 ~13:40 (commit `be86e77`) |

### Phase 2.9 (commit `3aa7844`, Oct 2 2026 22:59)

| Event | Files Read | Why |
|:------|:-----------|:----|
| Reconciliation | `summary_test.json` for all systems | Read-only, not written |
| Metric blob injection | All `summary_test.json` files | Modified (hash injected by hand — **retracted in Phase 2.95**) |
| Gate script | `eval/queries_test.jsonl` (7 no_data rows read and printed) | See §2 above |
| Label-noise | `results/s1_rule_planner/raw/dev/*.json` (10 files) | Read-only |

### Phase 2.95 (overnight rerun — to be populated after run)

After `bash scripts/run_all.sh` completes, every summary JSON will include
`git_commit` and `metric_blob_hash`. This log will be updated with the run
timestamp and commit hash.

---

## 4. Split Assignment (frozen at commit `9a499cc`)

- `eval/queries_train.jsonl`: 138 queries (not used for evaluation)
- `eval/queries_dev.jsonl`: 45 queries (used for DEV-only development)
- `eval/queries_test.jsonl`: 47 queries (held-out test split)

Gold labels: **never modified after commit `9a499cc`** (human IRR commit).
The 50 human-labeled queries in `eval/human_annotation_sample_filled.csv`
span train+dev splits. They are used only for label-noise sensitivity analysis
and IRR; they are NOT evaluation data.

---

## 5. Phase 2.9 Errors Corrected in Phase 2.95

| Error | Phase 2.9 Action | Phase 2.95 Correction |
|:------|:----------------|:---------------------|
| Metric blob hash hand-injected into summary JSONs | Added `metric_blob_hash` field manually | Removed; now auto-stamped by `save_system_results()` at run time |
| S7 stratum missing | "Explained" identical block-unsupported by reasoning | Added `synthesis_stratum` field to every S7 raw answer; field computed by code, not inferred post-hoc |
| A1 straw-man not verified | Documented as limitation | `a1_cap_via_s5.py` reruns cap=4 vs cap=unlimited using production S5 code and reports whether cap reproduces S5 F1 |
| S2 label-noise sensitivity not computable | Noted as limitation | `label_noise_sensitivity.py` runs S2 on all 50 human-labeled queries fresh |
| Stats family defined after computing | Family items enumerated during reconciliation | `experiments/stats_family.json` pre-declares k=26 comparisons before any computation |
| S5 hardcoded model | `PRIMARY_MODEL` only | Added `--model` argument to `s5_react.py` |
| Gate script inspected test queries | Gate developed on DEV; test inspection undisclosed | Disclosed here in §2; gate result is "0/7 below S1 baseline", not applied |
