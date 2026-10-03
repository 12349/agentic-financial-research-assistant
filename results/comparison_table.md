# Old vs New Comparison Table

> Phase 2.95 rerun commit: `ec7a53963dc1a2e26153c75de83ce1c3bd58ffe1`
> Pre-freeze results lack `git_commit`; marked as `pre-freeze`.

| System | Split | Status | git_commit | F1 / Cit.Valid | EM / Unsup.Block | Latency ms | Notes |
|:-------|:------|:-------|:-----------|:---------------|:-----------------|:-----------|:------|
| S1 rule-planner | test | 📦 OLD | `pre-freeze` | 0.522 | 0.3191 | 30.23 | |
| S2 Qwen planner | test | 📦 OLD | `pre-freeze` | 0.6773 | 0.6383 | 1557.88 | |
| S4 call-all | test | 📦 OLD | `pre-freeze` | 0.5157 | 0.0638 | 22.72 | |
| S5 ReAct Qwen | test | 📦 OLD | `pre-freeze` | 0.4333 | 0.3404 | 17140.25 | |
| S5 ReAct Llama | test | ❌ NOT_RUN | `?` | ? | ? | ? | |
| S6 LLM synth Qwen | test | 📦 OLD | `pre-freeze` | 0.8592 | 0.2304 | 9642.93 | |
| S6 LLM synth Llama | test | 📦 OLD | `pre-freeze` | 0.8485 | 0.5776 | 10383.03 | |
| S7 Hybrid Qwen | test | 📦 OLD | `pre-freeze` | 1.0 | 0.1064 | 14685.0 | |
| S7 Hybrid Llama | test | 📦 OLD | `pre-freeze` | 1.0 | 0.1064 | 18964.0 | |
| S2 Llama planner | test | 📦 OLD | `pre-freeze` | 0.6142 | 0.2553 | 2845.59 | |
