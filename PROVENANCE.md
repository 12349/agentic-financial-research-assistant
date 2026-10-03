# PROVENANCE.md

> Single authoritative provenance file (merged from `results/PROVENANCE.md` in Phase 2.95).
> The duplicate at `results/PROVENANCE.md` was deleted in the same commit as this merge.
> All numbers in the paper must be copied from result files listed here.
> Every result file produced by the Phase 2.95 pipeline includes `git_commit`
> and `metric_blob_hash` written by `experiments/systems/common.py:save_system_results()`.

---

## 1. Commit Hash Registry

All reported numbers postdate the OpenMP and tokenization stability fixes in `tools/search_news.py`.

| Milestone | Commit | Date | Description |
|:----------|:-------|:-----|:------------|
| Initial clean commit | `aa751b8` | 2026-08-27 | Repository baseline before Phase 1 audit |
| Eval set & IRR audit | `d1e52f0` | 2026-10-01 | 229-query evaluation set with gold label self-consistency audit |
| Phase 1 research audit | `4e6a190` | 2026-10-01 | Comprehensive architectural and empirical audit (`paper/audit.md`) |
| Phase 2 implementation | `b8f8f4f` | 2026-10-01 | OpenMP fixes, Ollama integration, systems S1–S6, ablations A1–A5 |
| Phase 2.5 metric fixes | `e2d1909` | 2026-10-02 | Corrected synthesis metrics, label leakage fixes, guided IRR, test isolation |
| Phase 2.75 final results | `be86e77` | 2026-10-02 | Full benchmark run, all summary JSONs produced |
| Human IRR (gold labels frozen) | `9a499cc` | 2026-10-02 | Human IRR, gold labels never modified after this commit |
| Phase 2.9 integrity pass | `3aa7844` | 2026-10-02 | Reconciliation, AGENT_RATINGS rename, metric blob injection (retracted) |
| Phase 2.9 blind CSV fix | `4b50076` | 2026-10-02 | Added missing `eval/blind_human_scoring_sheet.csv` |
| Phase 2.95 freeze infrastructure | `f0f2a90` | 2026-10-02 | Auto-stamping, S7 stratum, independent checker, stats family |
| Phase 2.95 runtime fix | `ec7a539` | 2026-10-03 | Corrected runtime estimates, model-ordered pipeline, RESUME mode |
| **Phase 2.95 pre-run state** | **HEAD** | 2026-10-03 | All infrastructure committed; run begins from clean tree |

---

## 2. Pipeline Freeze Guarantee (Phase 2.95 Item 1)

`scripts/run_all.sh` refuses to execute if `git status --porcelain` is non-empty.
Every summary JSON written by `save_system_results()` contains:

```json
{
  "git_commit": "<HEAD SHA at run time>",
  "metric_blob_hash": "<git hash-object of synthesis_quality.py at run time>"
}
```

Stamped by code at write time, never injected by hand.
Phase 2.9 hand-injected fields were removed in Phase 2.95 (commit `ec7a539`).

---

## 3. Test-Split File Access Log

### Phase 2.75 runs (commit `be86e77`, Oct 2 2026)

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
| Stats recompute | Reads S1/S6/S7 test raw files | Oct 2 ~13:40 (commit `be86e77`) |

### Phase 2.5 metric debugging (commit `e2d1909`)

During metric audit, the following **12 test-split files** were inspected to diagnose
implausible faithfulness (0.081) and unsupported rate (0.613) in S1:
`A002`, `A011`, `A014`, `A021`, `A026`, `A027`, `A031`, `D011`, `D012`, `D022`, `D054`, `E002`
(all under `results/s1_rule_planner/raw/test/`).

No prompts or model weights were tuned on these outputs; the fixes were strictly to the
metric evaluator (`experiments/metrics/synthesis_quality.py`).

### Phase 2.9 (commit `3aa7844`, Oct 2 2026 22:59)

| Event | Files read | Why |
|:------|:-----------|:----|
| Reconciliation | `summary_test.json` for all systems | Read-only |
| Metric blob injection | All `summary_test.json` | Modified (retracted in Phase 2.95) |
| Gate script | `eval/queries_test.jsonl` (7 no_data rows) | See §4 below |
| Label-noise | `results/s1_rule_planner/raw/dev/*.json` (10 files) | Read-only |

### Phase 2.95 (overnight run — commit `ec7a539` and above)

After `bash scripts/run_all.sh` completes, every summary JSON will include
`git_commit` and `metric_blob_hash`. This log will be updated with run timestamp.

---

## 4. Phase 2.9 Gate Script Disclosure

In Phase 2.9, `experiments/abstention_gate.py` read and printed test-split `no_data`
query texts before the overnight rerun.

**Test queries inspected by gate script (7 queries):**
- ND003: "What is Amazon's forward revenue guidance?"
- ND004: "Any news about Alphabet's antitrust case?"
- ND009: "What dividend does Exxon Mobil pay?"
- ND024: "Give me Google's analyst ratings and price targets."
- ND025: "What is the options flow on NVDA calls expiring next Friday?"
- ND028: "Show me JPMorgan's credit default swap spreads."
- ND031: "What's the news on Coinbase's regulatory battles?"

Gate was NOT fitted to these queries (used pre-specified `KNOWN_TICKERS` only).
These 7 queries are contaminated for any future gate-design purpose.
Gate result: 0/7 abstention accuracy (below S1 baseline 3/7). Gate NOT applied.

---

## 5. Split Assignment (frozen at commit `9a499cc`)

| File | n | Role |
|:-----|:-:|:-----|
| `eval/queries_train.jsonl` | 138 | Not used for evaluation |
| `eval/queries_dev.jsonl` | 45 | DEV-only development |
| `eval/queries_test.jsonl` | 47 | Held-out test split |

Gold labels: **never modified after `9a499cc`**.
The 50 human-labeled queries in `eval/human_annotation_sample_filled.csv`
span train+dev only. Used only for label-noise sensitivity and IRR.

---

## 6. Prompt Iteration History

### Planner prompts

| Iteration | Target | Description | Outcome |
|:----------|:-------|:------------|:--------|
| 1 | Groq/Claude freeform | Markdown with code fences | Parse failures on local 7B models |
| 2 | qwen2.5:7b-instruct JSON | `Output ONLY a raw JSON array` | 97.9% parse success (46/47 test) |
| 3 | ReAct (S5) | Thought-Action-Observation loop | Early-stopping pathology (mean 1.38 steps) |

### Synthesizer prompts

| Iteration | System | Outcome |
|:----------|:-------|:--------|
| 1 | S1 deterministic formatter | 100% citation, 100% faithfulness, 9.8 ms |
| 2 | S6 LLM synthesis | 85.9% citation, 23.0% unsupported, 9.6 s |
| 3 | S7 hybrid (LLM + verifier) | 100% citation (via formatter fallback), 10.6% block-unsupported |

### Labeling prompts

| Iteration | κ | Notes |
|:----------|:--|:------|
| Unprompted baseline | 0.2507 | High subjectivity |
| Formal guidelines | 0.3419 | Codified single/dual/ambiguous/no_data |

---

## 7. S5 Code Invariance (Phase 2.5 → Phase 2.75)

After the initial S5 test run (commit `b8f8f4f`), `s5_react.py` was modified
in commit `1a6102a` only to add `max_steps` parameter for the A1 ablation.

```diff
-def run_react_query(query: str, client: OllamaClient) -> dict:
+def run_react_query(query: str, client: OllamaClient, max_steps: int = REACT_MAX_STEPS) -> dict:
```

`run_s5()` calls `run_react_query(query_text, client)` with no `max_steps` argument,
so the default `REACT_MAX_STEPS=4` is used — identical to the original function.
Reported S5 test numbers (`F1=0.4333`, `EM=0.3404`) are invariant to this change.

---

## 8. Phase 2.9 Errors Corrected in Phase 2.95

| Error | Phase 2.9 | Phase 2.95 Correction |
|:------|:----------|:----------------------|
| Metric blob hash hand-injected | Added manually post-hoc | Removed; auto-stamped by `save_system_results()` |
| S7 stratum missing | Explained by prose reasoning | `synthesis_stratum` field written by code per-answer |
| A1 straw-man unverified | Documented as limitation | `a1_cap_via_s5.py` reruns with production S5 code |
| S2 label-noise not runnable | Noted as limitation | `label_noise_sensitivity.py` runs S2 on all 50 |
| Stats family defined post-hoc | k=26 after computing | `stats_family.json` k=13, declared before rerun |
| S5 hardcoded model | `PRIMARY_MODEL` only | `--model` arg added to `s5_react.py` |
| Gate script inspected test queries | Undisclosed | Disclosed above in §4 |
| Runtime estimate wrong (55h) | Confused ms with seconds | Corrected to 3.3h from logged latencies |
