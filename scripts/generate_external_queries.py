"""
scripts/generate_external_queries.py

Generates 50 external queries using llama3.1:8b (different model family from Qwen)
to test domain generalization and reduce prompt-author co-adaptation.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from experiments.config import SECOND_MODEL
from experiments.ollama_client import OllamaClient


def generate_queries():
    prompt_file = Path("eval/external_queries/generation_prompt.txt")
    raw_out = Path("eval/external_queries/external_queries_raw.jsonl")
    final_out = Path("eval/external_queries/external_queries.jsonl")

    prompt = prompt_file.read_text(encoding="utf-8")
    client = OllamaClient(model=SECOND_MODEL, think=False, num_predict=2048)

    print(f"Generating queries using {SECOND_MODEL}...")
    resp = client.generate(prompt)

    # Extract JSON array
    text = resp.response.strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    queries = []
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                queries = [q for q in parsed if isinstance(q, str)]
        except Exception:
            pass

    if not queries:
        # Fallback: extract line by line
        lines = [line.strip().strip('",-[]1234567890. ') for line in text.split("\n")]
        queries = [l for l in lines if len(l) > 15 and "?" in l]

    print(f"Generated {len(queries)} raw queries.")

    # Save raw
    with open(raw_out, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps({"query": q}) + "\n")

    # Annotate with rule-based rubric for initial labels (reviewed per plan)
    from planner.rule_based import plan
    records = []
    for i, q in enumerate(queries[:50]):
        pl = plan(q)
        tools = [c["tool"] for c in pl]
        if not tools:
            cat = "no_data"
        elif len(tools) == 1:
            cat = "single_tool"
        elif len(tools) == 2:
            cat = "dual_tool"
        else:
            cat = "ambiguous"

        rec = {
            "id": f"ext_{i+1:03d}",
            "query": q,
            "gold_tools": tools,
            "category": cat,
            "generator": SECOND_MODEL,
        }
        records.append(rec)

    with open(final_out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(records)} external queries to {final_out}")


if __name__ == "__main__":
    generate_queries()
