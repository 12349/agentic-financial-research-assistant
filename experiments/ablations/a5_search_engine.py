"""
experiments/ablations/a5_search_engine.py

Ablation 5: Search Engine (FAISS Semantic Embeddings vs. TF-IDF).
Compares retrieval results on news-seeking queries in the dev split.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from experiments.config import QUERIES_DEV, RESULTS_DIR
from experiments.systems.common import load_queries
from tools.search_news import search_news


def run_a5() -> dict:
    queries = load_queries(QUERIES_DEV)
    news_queries = [q for q in queries if "search_news" in q.get("gold_tools", [])]

    semantic_scores = []
    tfidf_scores = []
    top_1_agreements = 0

    for q in news_queries:
        query_text = q["query"]

        # 1. Semantic (FAISS)
        os.environ["NEWS_SEARCH_ENGINE"] = "semantic"
        res_sem = search_news(query_text)
        articles_sem = res_sem.get("results", [])

        # 2. TF-IDF
        os.environ["NEWS_SEARCH_ENGINE"] = "tfidf"
        res_tfidf = search_news(query_text)
        articles_tfidf = res_tfidf.get("results", [])

        if articles_sem:
            semantic_scores.append(articles_sem[0].get("score", 0.0))
        if articles_tfidf:
            tfidf_scores.append(articles_tfidf[0].get("score", 0.0))

        if articles_sem and articles_tfidf:
            if articles_sem[0].get("id") == articles_tfidf[0].get("id"):
                top_1_agreements += 1

    # Reset env
    os.environ["NEWS_SEARCH_ENGINE"] = "semantic"

    summary = {
        "ablation": "A5_search_engine",
        "split": "dev",
        "n_news_queries": len(news_queries),
        "mean_top1_semantic_score": round(sum(semantic_scores) / len(semantic_scores), 4) if semantic_scores else 0.0,
        "mean_top1_tfidf_score": round(sum(tfidf_scores) / len(tfidf_scores), 4) if tfidf_scores else 0.0,
        "top1_article_agreement_rate": round(top_1_agreements / len(news_queries), 4) if news_queries else 0.0,
    }

    out_file = Path(RESULTS_DIR) / "ablations" / "a5_search_engine.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"A5 results saved to: {out_file}")
    print(f"Top-1 Agreement Rate: {summary['top1_article_agreement_rate']}")
    print(f"Mean Semantic Score:  {summary['mean_top1_semantic_score']}")
    print(f"Mean TF-IDF Score:    {summary['mean_top1_tfidf_score']}")
    return summary


if __name__ == "__main__":
    run_a5()
