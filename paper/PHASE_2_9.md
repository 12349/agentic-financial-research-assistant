# Phase 2.9 — Integrity & Reconciliation Pass

> Saved verbatim from the session prompt on 2026-10-02 so any new session can continue.

## Context

- Phase 2.75 reports contained errors:
  - Answer-quality "human" ratings were written by the agent (HUMAN_RATINGS dict)
  - Answer-quality and metric-validation numbers in the report don't match the logs
  - S6 test citation validity moved from 0.859 to 0.9615 after metric edits
  - S7 Qwen and Llama block-unsupported numbers are identical
  - Live A1 ReAct baseline (F1 0.200) does not match reported S5 (F1 0.433)
- A table regeneration changed S6 unsupported rate 0.237 → 0.230, confirming results
  were produced under different metric versions
- Human IRR is complete (commit 9a499cc). Do NOT change any gold labels.

---

## Items (in order)

### 1. REPORT_RECONCILIATION
Create `results/REPORT_RECONCILIATION.md`: list every number in the Phase 2.75
final report, its source file, and whether it matches. Correct `paper/audit.md`
to match the files. Explain why two answer-quality runs differ.

### 2. HUMAN_RATINGS → AGENT_RATINGS
Rename HUMAN_RATINGS to AGENT_RATINGS everywhere, relabel outputs and docs,
delete the hardcoded tradeoff_findings string. Generate a blind shuffled CSV
(system names hidden, no judge scores) for a real human to score.

### 3. METRIC VALIDATION REBUILD
Rebuild the metric validation set with independently written expected verdicts
(not computed by the metric under test). Drop claims of R^2 or "hand-annotated"
unless true. Produce a CSV for human verification of real S1/S6 outputs.

### 4. METRIC VERSIONING
Diff `synthesis_quality.py` across the commits between S6 citation validity
0.8592 and 0.9615 and explain. Freeze the metric, write its git hash into every
results JSON, and recompute S1, S6, S7 on test for both model families under
that single version. Regenerate all tables from those results only.

### 5. S7 INVESTIGATION
Diff `s7_hybrid_synth.py` between the Qwen and Llama test runs and rerun both
with final code; explain the identical block-unsupported number and CI; report
counts of fully pruned answers and fallbacks to the deterministic formatter;
run S7 on dev with answer quality (relevance/completeness) to show pruning cost;
add an independently implemented checker and manually audit 20 S7 answers;
remove "tuned on DEV" unless true.

### 6. A1/S5 FIX
Make the ablation call the real S5 implementation with a cap parameter so
cap=4 reproduces F1 0.433. Redo the improved-prompt comparison against that.
Retract the straw-man claim until it holds.

### 7. ABSTENTION GATE
Add a relevance gate (ticker/entity match plus retrieval score threshold)
developed on DEV only. Evaluate S1+gate and S7+gate on test abstention and
in-scope answer quality. Report n=7 results descriptively.

### 8. LABEL-NOISE SENSITIVITY
On the 50 human-labeled queries (train/dev only), report S1 and S2 tool-selection
F1 and exact match under author labels and under human labels, with CIs. This is
a sensitivity analysis only. Do not modify gold labels, and do not tune anything
on these 50. Also report, from `results/human_irr/disagreements_analysis.md`,
the actual disagreeing queries with both labels and the human's notes, not only
the agent's grouping.

### 9. PROVENANCE
Rebuild the list of test-split files read from logs and shell history (include
D054 and the loop that printed all test answers with unfaithful numbers).

### 10. STATS
Holm-Bonferroni correction across the family of paired tests; report raw and
adjusted p-values.

### 11. HYGIENE
Run `git show --stat HEAD`; confirm no secrets, .env files, model weights, or
unrelated folders (e.g. WEDDING PROJECT/) were committed.

### 12. SAVE SPEC
Save this prompt text to `paper/PHASE_2_9.md` and commit it, so a new session
can read it.

---

## Guiding Rules

- Every reported number must be copied from a result file with its path.
- Never put interpretive conclusions into scripts.
- Report: what changed, what was wrong, what still can't be verified.
