# Experimental Provenance & Reproducibility Ledger

This document provides a strict, immutable audit trail for all experimental results, prompt iterations, and dataset splits reported in the paper.

---

## 1. Commit Hashes & Baseline Verification

All reported numbers postdate the critical macOS OpenMP and tokenization stability fixes implemented in [`tools/search_news.py`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/tools/search_news.py).

| Milestone | Commit Hash | Date | Description |
| :--- | :--- | :--- | :--- |
| **Initial Clean Commit** | `aa751b8` | 2026-08-27 | Repository baseline before Phase 1 research audit |
| **Eval Set & IRR Audit** | `d1e52f0` | 2026-10-01 | 229-query evaluation set with gold label self-consistency audit |
| **Phase 1 Research Audit** | `4e6a190` | 2026-10-01 | Comprehensive architectural and empirical audit (`paper/audit.md`) |
| **Phase 2 Implementation** | `b8f8f4f` | 2026-10-01 | OpenMP fixes, Ollama integration, systems S1–S6, ablations A1–A5 |
| **Phase 2.5 Metric & Leakage Fixes** | *(Current)* | 2026-10-02 | Corrected synthesis metrics, label leakage fixes, guided IRR, test isolation |

---

## 2. Test Split Isolation Audit

**Strict Policy**: The held-out test split (`eval/queries_test.jsonl`, $n=47$) was **never used for prompt tuning, threshold calibration, or system iteration**.

### Exhaustive Log of Test Set Reads

| Timestamp (UTC) | Execution Target | Purpose | Type |
| :--- | :--- | :--- | :--- |
| 2026-10-02 04:04 | `experiments/systems/s1_rule_planner.py` | S1 baseline test evaluation | Read-only evaluation |
| 2026-10-02 04:06 | `experiments/systems/s2_llm_planner.py` | S2 primary planner test evaluation | Read-only evaluation |
| 2026-10-02 04:12 | `experiments/systems/s3_no_tools.py` | S3 parametric baseline test evaluation | Read-only evaluation |
| 2026-10-02 04:18 | `experiments/systems/s4_call_all.py` | S4 heuristic ceiling test evaluation | Read-only evaluation |
| 2026-10-02 04:49 | `experiments/systems/s5_react.py` | S5 ReAct agent test evaluation | Read-only evaluation |
| 2026-10-02 05:09 | `experiments/systems/s1_rule_planner.py` | Post-metric-audit re-run of S1 | Read-only evaluation |
| 2026-10-02 05:19 | `experiments/systems/s6_llm_synth.py` | Post-metric-audit re-run of S6 | Read-only evaluation |
| 2026-10-02 05:31 | `tests/test_label_integrity.py` | Automated CI verification against circular leakage | Automated test |
| 2026-10-02 05:50 | `experiments/systems/s2_llm_planner.py` | Post-metric-audit re-run of S2 | Read-only evaluation |
| 2026-10-02 05:51 | `scripts/benchmark_variance.py` | Cross-seed variance check (seeds 42, 101, 202, 303) | Read-only evaluation |

**Confirmation**:
- All hyperparameter selection, prompt drafting, and ablations (A1–A5) were developed and executed exclusively on `QUERIES_DEV` ($n=45$) and `QUERIES_TRAIN` ($n=137$).
- The human annotation sample (`eval/human_annotation_sample.csv`, $n=50$) was extracted exclusively from the dev and train splits, completely preserving test set blinding.

---

## 3. Prompt Iteration History

### 3.1 Planner Prompts

#### Iteration 1: Cloud API Freeform (Early Prototype)
- **Target**: Groq / Claude-3 Haiku
- **Design**: Freeform markdown response with code blocks.
- **Limitation**: Prone to parsing failures when ported to local 7B models; markdown code fences often included conversational preamble.

#### Iteration 2: Structured JSON System Prompt ([`planner/llm_planner.py`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/planner/llm_planner.py))
- **Target**: `qwen2.5:7b-instruct` (Ollama REST API, `temperature=0`, `think=False`)
- **Design**: Explicit schema instruction: `Output ONLY a raw JSON array of tool calls. Do not include markdown code fences or explanation.`
- **Performance**: Parse success rate increased to **97.9%** (46/47 on test).

#### Iteration 3: ReAct Reasoning Loop ([`experiments/systems/s5_react.py`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/experiments/systems/s5_react.py))
- **Target**: Interleaved Thought-Action-Observation.
- **Observation**: Agent exhibited early-stopping pathology (mean 1.38 steps), stopping prematurely before executing dual-tool sequences.

---

### 3.2 Synthesizer Prompts

#### Iteration 1: Deterministic Template Formatter ([`agent/synthesizer.py`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/agent/synthesizer.py))
- **Design**: Non-parametric programmatic text builder. Emits `[Source: <record_id>]` prefixes directly from fixture record IDs and formats numerical fields using standard rounding.
- **Empirical Properties**: 100% citation validity, 100% numeric faithfulness, 0% unsupported claims, 9.8 ms latency.

#### Iteration 2: LLM Synthesis Prompt ([`experiments/systems/s6_llm_synth.py`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/experiments/systems/s6_llm_synth.py))
- **Design**: Instructs `qwen2.5:7b-instruct` to synthesize a coherent narrative from tool output JSON while preserving citations.
- **Empirical Properties**: High fluency, but drops citation validity to 85.9%, generates 23.8% unsupported statements, and degrades abstention accuracy from 42.9% to 14.3%.

---

### 3.3 Labeling & Evaluation Prompts

#### Iteration 1: Unprompted Baseline (`eval/external_queries/labeling_prompt.txt`)
- **Prompt**: 18 lines, minimal context, no disambiguation criteria.
- **Result**: Cross-family Cohen's $\kappa = 0.2507$ (fair agreement). Demonstrated high subjectivity in tool routing without explicit definitions.

#### Iteration 2: Formal Guidelines (`eval/external_queries/labeling_prompt_with_guidelines.txt`)
- **Prompt**: Codified the rules from [`eval/labeling_guidelines.md`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/eval/labeling_guidelines.md), defining single_tool, dual_tool, ambiguous, and no_data.
- **Result**: Cross-family Cohen's $\kappa$ increased to **0.3419** for tool sets and **0.3652** for categories.
