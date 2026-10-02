"""
experiments/ablations/a3_sentiment_mode.py

Ablation 3: Sentiment Scoring Mode (DistilBERT vs Keyword Bucket).
Evaluates the impact of ML sentiment scoring on analyst ratings commentary
across all ratings queries in the dev set.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.config import QUERIES_DEV, RESULTS_DIR
from experiments.systems.common import load_queries
from tools.get_ratings import _get_ratings_offline, _load_fixtures


def run_a3() -> dict:
    queries = load_queries(QUERIES_DEV)
    ratings_queries = [q for q in queries if "get_ratings" in q.get("gold_tools", [])]

    # Evaluate across all 4 tickers in fixture
    tickers = ["NVDA", "TSLA", "JPM", "XOM"]
    comparison = {}

    for t in tickers:
        # Full ML mode
        ml_res = _get_ratings_offline(t, limit=10)
        ml_ratings = ml_res.get("ratings", [])

        # Keyword-only comparison
        keyword_sentiments = []
        ml_sentiments = []
        confs = []

        sentiment_map = {
            "buy": "bullish", "outperform": "bullish", "overweight": "bullish",
            "sell": "bearish", "underperform": "bearish",
            "neutral": "neutral", "hold": "neutral",
        }

        for r in ml_ratings:
            kw = sentiment_map.get(r.get("rating", "").lower(), "neutral")
            keyword_sentiments.append(kw)
            ml = r.get("ml_sentiment", kw)
            ml_sentiments.append(ml)
            if r.get("ml_confidence"):
                confs.append(r["ml_confidence"])

        agreements = sum(1 for m, k in zip(ml_sentiments, keyword_sentiments) if m == k)
        comparison[t] = {
            "total_ratings": len(ml_ratings),
            "ml_scored_count": len(confs),
            "mean_confidence": round(sum(confs) / len(confs), 3) if confs else 0.0,
            "keyword_vs_ml_agreement": round(agreements / len(ml_ratings), 3) if ml_ratings else 1.0,
            "ml_consensus": ml_res.get("summary", {}).get("consensus"),
        }

    summary = {
        "ablation": "A3_sentiment_mode",
        "split": "dev",
        "n_ratings_queries": len(ratings_queries),
        "per_ticker": comparison,
    }

    out_file = Path(RESULTS_DIR) / "ablations" / "a3_sentiment_mode.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"A3 results saved to: {out_file}")
    for t, data in comparison.items():
        print(f"  {t}: Agreement={data['keyword_vs_ml_agreement']}, Mean Conf={data['mean_confidence']}")
    return summary


if __name__ == "__main__":
    run_a3()
