# External Query Dataset (EXPLORATORY)

> [!WARNING]
> **Exploratory Status**: All external-query evaluation results in this directory and `results/external_queries/` are explicitly designated as **exploratory** until independent human ground-truth labels are collected and merged.

## Methodological Disclosures
1. **Identical Model for Generation and Labeling**: Both query generation (`generation_prompt.txt`) and automated labeling (`labeling_prompt_with_guidelines.txt`) were performed by the **same model family**: `llama3.1:8b` (Meta, Q4_K_M quantization, temperature 0.0). Consequently, inter-rater agreement between the generator and labeler reflects model self-consistency rather than independent human consensus.
2. **Author vs. Model Agreement**: When evaluated against independent author annotations following strict labeling guidelines, the Llama 3.1 8B labels show moderate agreement ($\kappa = 0.3419$ on guided pass; baseline without guidelines $\kappa = 0.2507$).
3. **Human Validation Loader**: When external human annotations are available in CSV format, execute:
   ```bash
   python3 -m eval.external_queries.merge_human_labels --csv path/to/human_labels.csv
   ```
   This merges the human labels and recomputes Cohen's $\kappa$ across all three pairs (Human vs. Author, Human vs. Llama, Author vs. Llama).

## Generation Details
- **Generator Model**: `llama3.1:8b`
- **Labeler Model**: `llama3.1:8b` (identical model family)
- **Quantization**: Q4_K_M
- **Temperature**: 0.0
- **Seed**: 42
- **Hardware**: Apple M4 (Mac mini 2024, 16 GB unified memory, macOS 27.0)
