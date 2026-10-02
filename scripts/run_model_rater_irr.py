"""
scripts/run_model_rater_irr.py

Runs the second-rater label pass using llama3.1:8b on the 34-query IRR sample.
Computes Cohen's kappa between author labels (Pass 1) and llama3.1:8b labels.
Reports honestly as "LLM second-rater (llama3.1:8b) agreement", NOT human IRR.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from experiments.config import RESULTS_DIR, SECOND_MODEL
from experiments.metrics.tool_selection import bootstrap_ci
from experiments.ollama_client import rater_client


def cohen_kappa(rater1: list[str], rater2: list[str]) -> float:
    """Compute Cohen's kappa for two raters with categorical ratings."""
    assert len(rater1) == len(rater2)
    n = len(rater1)
    if n == 0:
        return 0.0

    categories = sorted(set(rater1) | set(rater2))
    po = sum(1 for a, b in zip(rater1, rater2) if a == b) / n

    c1 = {c: rater1.count(c) / n for c in categories}
    c2 = {c: rater2.count(c) / n for c in categories}
    pe = sum(c1[c] * c2[c] for c in categories)

    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def run_llama_irr():
    eval_dir = Path("eval")
    with open(eval_dir / "queries.jsonl", "r", encoding="utf-8") as f:
        all_queries = [json.loads(line) for line in f]

    # Replicate exact sample (n=34, seed=999)
    rng = random.Random(999)
    sample_size = max(1, int(len(all_queries) * 0.15))
    sample_indices = sorted(rng.sample(range(len(all_queries)), sample_size))
    sample = [all_queries[i] for i in sample_indices]

    prompt_template = Path("eval/external_queries/labeling_prompt.txt").read_text(encoding="utf-8")
    client = rater_client(model=SECOND_MODEL)

    print(f"Running LLM rater pass with {SECOND_MODEL} on n={len(sample)} queries...")

    author_tool_sets = []
    author_categories = []
    llama_tool_sets = []
    llama_categories = []
    records = []

    for i, q in enumerate(sample):
        query_text = q["query"]
        prompt = prompt_template.replace("{{QUERY}}", query_text)
        resp = client.generate(prompt)

        # Parse response
        text = resp.response.strip()
        tools = []
        cat = "single_tool"
        rationale = ""

        try:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                tools = parsed.get("gold_tools", [])
                cat = parsed.get("category", "single_tool")
                rationale = parsed.get("rationale", "")
        except Exception:
            pass

        # Normalize tools
        tools = sorted(set(tools))
        gold_tools = sorted(set(q.get("gold_tools", [])))
        gold_cat = q.get("category", "single_tool")

        author_tool_sets.append(",".join(gold_tools))
        author_categories.append(gold_cat)
        llama_tool_sets.append(",".join(tools))
        llama_categories.append(cat)

        rec = {
            "query_id": q.get("id"),
            "query": query_text,
            "author_tools": gold_tools,
            "author_category": gold_cat,
            "llama_tools": tools,
            "llama_category": cat,
            "llama_rationale": rationale,
            "match_tools": gold_tools == tools,
            "match_category": gold_cat == cat,
        }
        records.append(rec)
        print(f"  [{i+1}/{len(sample)}] Match: tools={rec['match_tools']}, cat={rec['match_category']}")

    kappa_tools = cohen_kappa(author_tool_sets, llama_tool_sets)
    kappa_cat = cohen_kappa(author_categories, llama_categories)
    exact_tools_rate = sum(1 for r in records if r["match_tools"]) / len(records)
    exact_cat_rate = sum(1 for r in records if r["match_category"]) / len(records)

    summary = {
        "rater_model": SECOND_MODEL,
        "n_sample": len(sample),
        "sampling_seed": 999,
        "exact_tool_set_match_rate": round(exact_tools_rate, 4),
        "exact_category_match_rate": round(exact_cat_rate, 4),
        "cohens_kappa_tool_sets": round(kappa_tools, 4),
        "cohens_kappa_categories": round(kappa_cat, 4),
        "stated_limitation": (
            "This is a second-rater pass conducted with an independent LLM family (llama3.1:8b), "
            "not an independent human rater."
        ),
    }

    out_file = Path("eval/external_queries/llama_irr_labels.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    summary_file = Path(RESULTS_DIR) / "external_queries" / "llama_irr_agreement.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved LLM rater results to {out_file}")
    print(f"Summary saved to {summary_file}")
    print(f"Tool-set Kappa:  {kappa_tools:.4f}")
    print(f"Category Kappa:  {kappa_cat:.4f}")
    print(f"Exact Tools Acc: {exact_tools_rate:.4f}")


if __name__ == "__main__":
    run_llama_irr()
