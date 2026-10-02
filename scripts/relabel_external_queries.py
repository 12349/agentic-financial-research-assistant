"""
scripts/relabel_external_queries.py

Independently labels the external queries dataset using llama3.1:8b
with explicit guidelines from eval/labeling_guidelines.md.
Completely removes any circular dependency on planner.rule_based.plan().
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from experiments.config import SECOND_MODEL
from experiments.ollama_client import OllamaClient

VALID_TOOLS = {"get_earnings", "get_guidance", "get_ratings", "search_news"}
VALID_CATEGORIES = {"single_tool", "dual_tool", "ambiguous", "no_data"}


def relabel_queries():
    prompt_template = Path("eval/external_queries/labeling_prompt_with_guidelines.txt").read_text(encoding="utf-8")
    ext_file = Path("eval/external_queries/external_queries.jsonl")

    queries = []
    with open(ext_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    print(f"Relabeling {len(queries)} external queries independently with {SECOND_MODEL}...")
    client = OllamaClient(model=SECOND_MODEL, temperature=0.0, think=False, num_predict=256)

    updated_records = []
    for idx, q_rec in enumerate(queries, 1):
        q_text = q_rec["query"]
        prompt = prompt_template.replace("{{QUERY}}", q_text)
        resp = client.generate(prompt)

        text = resp.response.strip()
        gold_tools = []
        category = "single_tool"
        rationale = ""

        try:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                raw_tools = parsed.get("gold_tools", [])
                gold_tools = sorted([t for t in raw_tools if t in VALID_TOOLS])
                raw_cat = parsed.get("category", "").lower().strip()
                if raw_cat in VALID_CATEGORIES:
                    category = raw_cat
                rationale = parsed.get("rationale", "")
        except Exception as e:
            print(f"Warning parsing [{idx}]: {e}")

        # Post-validation per guidelines: if category is no_data, gold_tools MUST be []
        if category == "no_data":
            gold_tools = []
        elif not gold_tools and category != "no_data":
            category = "no_data"

        new_rec = {
            "id": q_rec["id"],
            "query": q_text,
            "gold_tools": gold_tools,
            "category": category,
            "generator": SECOND_MODEL,
            "labeler": f"{SECOND_MODEL} (independent guidelines pass)",
            "rationale": rationale,
        }
        updated_records.append(new_rec)
        print(f"  [{idx}/{len(queries)}] {category}: {gold_tools} | {q_text[:60]}")

    with open(ext_file, "w", encoding="utf-8") as f:
        for r in updated_records:
            f.write(json.dumps(r) + "\n")

    print(f"\nSuccessfully wrote {len(updated_records)} independently labeled queries to {ext_file}.")


if __name__ == "__main__":
    relabel_queries()
