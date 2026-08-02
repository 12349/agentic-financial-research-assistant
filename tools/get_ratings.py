"""
tools/get_ratings.py

Returns the most recent analyst ratings for a given ticker.

Offline mode (default):  loads data/ratings_fixtures.json
Online mode:             calls the financial data API when FINANCIAL_DATA_API_KEY is set

Sentiment scoring (two-layer):
  1. ML classifier (classifier/predict.py) — scores analyst notes with fine-tuned
     DistilBERT when model weights are present in classifier/model/.
     Reports per-rating ml_sentiment + ml_confidence.
  2. Keyword bucket (fallback) — maps rating action (Buy/Sell/Hold) to
     bullish/bearish/neutral. Always available, zero-dependency.

Returns a dict with:
  ticker, ratings (list), summary (bull/bear/neutral counts), mode
Each rating has a 'source_ref': {ticker, channel, record_id}
"""

import os
import json
import time
from pathlib import Path
from collections import Counter

_DATA_DIR = Path(__file__).parent.parent / "data"
_FIXTURES_PATH = _DATA_DIR / "ratings_fixtures.json"


def _ml_score_notes(ratings: list[dict]) -> list[dict]:
    """
    Attempt to score each rating's analyst notes with the fine-tuned classifier.
    Adds 'ml_sentiment' and 'ml_confidence' fields in-place.
    Safe no-op if classifier/model/ weights are absent.
    """
    try:
        from classifier.predict import classify_batch  # noqa: PLC0415
        notes = [r.get("notes", "") or "" for r in ratings]
        # Only call model if there are non-empty notes
        non_empty_indices = [i for i, n in enumerate(notes) if n.strip()]
        if not non_empty_indices:
            return ratings
        non_empty_texts = [notes[i] for i in non_empty_indices]
        predictions = classify_batch(non_empty_texts)
        for idx, pred in zip(non_empty_indices, predictions):
            if pred is not None:
                label, conf = pred
                ratings[idx]["ml_sentiment"]   = label
                ratings[idx]["ml_confidence"]  = round(conf, 3)
    except Exception:  # noqa: BLE001
        # Classifier unavailable or errored — proceed with keyword fallback only
        pass
    return ratings


def _load_fixtures() -> list[dict]:
    with open(_FIXTURES_PATH, "r") as f:
        return json.load(f)


def _get_ratings_offline(ticker: str, limit: int) -> dict:
    all_ratings = _load_fixtures()
    ticker_upper = ticker.upper()
    ratings = [r for r in all_ratings if r["ticker"].upper() == ticker_upper]

    if not ratings:
        return {
            "ticker": ticker_upper,
            "ratings": [],
            "summary": None,
            "error": f"No ratings data found for {ticker_upper}",
        }

    # Sort by date descending
    ratings_sorted = sorted(ratings, key=lambda r: r["date"], reverse=True)[:limit]

    # Add source_ref to each
    for r in ratings_sorted:
        r["source_ref"] = {
            "ticker": r["ticker"],
            "channel": "analyst_ratings",
            "record_id": r["id"],
        }

    # ML classifier: score analyst notes text (no-op if model absent)
    ratings_sorted = _ml_score_notes(ratings_sorted)

    # Keyword bucket: maps rating action to sentiment (always available)
    # Used as primary for ratings with no ML score, or when model absent
    sentiment_map = {
        "buy": "bullish", "outperform": "bullish", "overweight": "bullish",
        "strong buy": "bullish",
        "sell": "bearish", "underperform": "bearish", "underweight": "bearish",
        "neutral": "neutral", "hold": "neutral", "equal-weight": "neutral",
    }
    # Use ML sentiment when available for notes-bearing ratings; keyword bucket otherwise
    sentiments = []
    ml_count = 0
    for r in ratings_sorted:
        if r.get("ml_sentiment"):
            sentiments.append(r["ml_sentiment"])
            ml_count += 1
        else:
            sentiments.append(sentiment_map.get(r["rating"].lower(), "neutral"))

    counts = Counter(sentiments)
    total = len(sentiments)
    consensus = max(counts, key=counts.get) if counts else "neutral"

    summary = {
        "total_ratings": total,
        "bullish": counts.get("bullish", 0),
        "neutral": counts.get("neutral", 0),
        "bearish": counts.get("bearish", 0),
        "consensus": consensus,
        "ml_scored_count": ml_count,   # ratings scored by trained classifier
        "avg_price_target": round(
            sum(r["price_target"] for r in ratings_sorted if r.get("price_target"))
            / max(1, sum(1 for r in ratings_sorted if r.get("price_target"))),
            2,
        ),
    }

    return {
        "ticker": ticker_upper,
        "ratings": ratings_sorted,
        "summary": summary,
    }


# Set RATINGS_API_URL (and FINANCIAL_DATA_API_KEY) to point at your data provider.
# Example: RATINGS_API_URL=https://api.yourprovider.com/v2.1/calendar/ratings
_RATINGS_API_URL = os.environ.get("RATINGS_API_URL", "")


def _get_ratings_online(ticker: str, limit: int) -> dict:
    """Call the financial data API for analyst ratings. Falls back to offline on any error."""
    try:
        import requests  # noqa: PLC0415
        api_key = os.environ["FINANCIAL_DATA_API_KEY"]
        params = {
            "token": api_key,
            "parameters[tickers]": ticker.upper(),
            "parameters[pageSize]": limit,
        }
        resp = requests.get(
            _RATINGS_API_URL,
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json()
        items = raw.get("ratings", raw) if isinstance(raw, dict) else raw

        ratings = []
        for item in items[:limit]:
            rating = {
                "id": str(item.get("id", "")),
                "ticker": ticker.upper(),
                "analyst_firm": item.get("analyst", ""),
                "analyst_name": item.get("analyst_name", ""),
                "rating": item.get("rating_current", ""),
                "prior_rating": item.get("rating_prior", ""),
                "price_target": item.get("pt_current"),
                "prior_price_target": item.get("pt_prior"),
                "action": item.get("action_company", ""),
                "date": item.get("date", ""),
                "notes": "",
                "source_ref": {
                    "ticker": ticker.upper(),
                    "channel": "analyst_ratings",
                    "record_id": str(item.get("id", "")),
                },
            }
            ratings.append(rating)

        if not ratings:
            return _get_ratings_offline(ticker, limit)

        sentiments = []
        sentiment_map = {
            "buy": "bullish", "outperform": "bullish", "overweight": "bullish",
            "sell": "bearish", "underperform": "bearish",
            "neutral": "neutral", "hold": "neutral",
        }
        for r in ratings:
            sentiments.append(sentiment_map.get(r["rating"].lower(), "neutral"))
        counts = Counter(sentiments)
        consensus = max(counts, key=counts.get) if counts else "neutral"
        pts = [r["price_target"] for r in ratings if r.get("price_target")]

        return {
            "ticker": ticker.upper(),
            "ratings": ratings,
            "summary": {
                "total_ratings": len(ratings),
                "bullish": counts.get("bullish", 0),
                "neutral": counts.get("neutral", 0),
                "bearish": counts.get("bearish", 0),
                "consensus": consensus,
                "avg_price_target": round(sum(pts) / len(pts), 2) if pts else None,
            },
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[get_ratings] Online fetch failed ({exc}), falling back to offline.")
        return _get_ratings_offline(ticker, limit)


def get_ratings(ticker: str, limit: int = 5) -> dict:
    """
    Get the most recent analyst ratings for a ticker.

    Args:
        ticker: Stock ticker symbol (e.g. 'NVDA').
        limit:  Maximum number of ratings to return (default 5).

    Returns:
        {
          "ticker":    str,
          "ratings":   list of rating dicts (with source_ref),
          "summary":   {total_ratings, bullish, neutral, bearish, consensus, avg_price_target},
          "mode":      "offline" | "online",
          "tool":      "get_ratings",
          "latency_ms": float,
        }
    """
    t0 = time.time()
    mode = "online" if os.environ.get("FINANCIAL_DATA_API_KEY") else "offline"

    if mode == "online":
        result = _get_ratings_online(ticker, limit)
    else:
        result = _get_ratings_offline(ticker, limit)

    result["mode"] = mode
    result["tool"] = "get_ratings"
    result["latency_ms"] = round((time.time() - t0) * 1000, 1)
    return result


if __name__ == "__main__":
    out = get_ratings("TSLA")
    print(json.dumps(out, indent=2))
