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

### 2.1 Full Disclosure: Metric Debugging Inspection of Test Split Outputs
During the Phase 2.5 metric audit (Item 1), the deterministic formatter's implausible initial metrics ($0.081$ faithfulness, $0.613$ unsupported rate) were investigated by printing and inspecting the following **12 test-split result files**:
1. [`results/s1_rule_planner/raw/test/A002.json`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/results/s1_rule_planner/raw/test/A002.json)
2. [`results/s1_rule_planner/raw/test/A011.json`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/results/s1_rule_planner/raw/test/A011.json)
3. [`results/s1_rule_planner/raw/test/A014.json`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/results/s1_rule_planner/raw/test/A014.json)
4. [`results/s1_rule_planner/raw/test/A021.json`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/results/s1_rule_planner/raw/test/A021.json)
5. [`results/s1_rule_planner/raw/test/A026.json`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/results/s1_rule_planner/raw/test/A026.json)
6. [`results/s1_rule_planner/raw/test/A027.json`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/results/s1_rule_planner/raw/test/A027.json)
7. [`results/s1_rule_planner/raw/test/A031.json`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/results/s1_rule_planner/raw/test/A031.json)
8. [`results/s1_rule_planner/raw/test/D011.json`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/results/s1_rule_planner/raw/test/D011.json)
9. [`results/s1_rule_planner/raw/test/D012.json`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/results/s1_rule_planner/raw/test/D012.json)
10. [`results/s1_rule_planner/raw/test/D022.json`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/results/s1_rule_planner/raw/test/D022.json)
11. [`results/s1_rule_planner/raw/test/D054.json`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/results/s1_rule_planner/raw/test/D054.json) *(inspected to diagnose rounding of `5.76%` to `+5.8%`)*
12. [`results/s1_rule_planner/raw/test/E002.json`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/results/s1_rule_planner/raw/test/E002.json) *(inspected to diagnose rounding of `-1.25%` to `-1.2%`)*

**Methodological Implication**: Although no system prompts or model weights were tuned on test (the changes were strictly bug fixes in the metric evaluator [`experiments/metrics/synthesis_quality.py`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/experiments/metrics/synthesis_quality.py)), inspecting test outputs means the metric code had visibility into test data. To validate that the corrected metric does not overfit to these instances, an independent 30-answer validation set constructed entirely from DEV outputs and synthetic corruptions is evaluated in Section 2 of Phase 2.75.

---

### 2.2 ReAct Agent (`s5_react.py`) Code Modifications & Test Invariance Proof
Following the initial test run of S5 in commit `b8f8f4f`, [`experiments/systems/s5_react.py`](file:///Volumes/Johnys%20Extreme%20Pro/Johny's%20MiniX/Downloads/Multi-Agent%20Financial%20Research%20Assistant%20—%20Full%20Project%20Spec/experiments/systems/s5_react.py) was modified in commit `1a6102a` solely to support the A1 tool-cap ablation parameter (`max_steps`).

**Exact Git Diff (`git diff b8f8f4f 1a6102a -- experiments/systems/s5_react.py`)**:
```diff
--- a/experiments/systems/s5_react.py
+++ b/experiments/systems/s5_react.py
@@ -68,7 +68,7 @@ def parse_action(text: str) -> Optional[tuple[str, dict]]:
     return tool_name, {}
 
 
-def run_react_query(query: str, client: OllamaClient) -> dict:
+def run_react_query(query: str, client: OllamaClient, max_steps: int = REACT_MAX_STEPS) -> dict:
     """Run multi-step ReAct agent on a single query."""
     history = f"Question: {query}\n"
     tools_called = []
@@ -81,7 +81,7 @@ def run_react_query(query: str, client: OllamaClient) -> dict:
 
     t0 = time.perf_counter()
 
-    for step in range(REACT_MAX_STEPS):
+    for step in range(max_steps):
         steps += 1
         prompt = f"{_REACT_SYSTEM_PROMPT}\n\n{history}Thought:"
         resp = client.generate(prompt, num_predict=256)
```

**Proof of Invariance for Reported S5 Test Numbers**:
1. In Python, default parameter evaluation binds `max_steps = REACT_MAX_STEPS = 4` when `run_react_query` is invoked without the optional argument.
2. In `run_s5()` (which executes the test evaluation), line 170 calls `res = run_react_query(query_text, client)` with no `max_steps` passed, executing identically to the original function.
3. The prompt template, system prompt, parser, and stopping conditions (`_FINAL_ANSWER_RE`) remain 100% byte-for-byte identical between commits `b8f8f4f` and `1a6102a`.
4. The reported test numbers in `results/s5_react/summary_test.json` are identical to the original run because the execution path for cap=4 is completely unchanged.

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
