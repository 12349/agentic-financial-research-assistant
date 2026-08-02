"""
tools/search_news.py

Searches financial news articles using TF-IDF similarity.

Offline mode (default):  loads data/news_fixtures.json
Online mode:             calls the financial data API when FINANCIAL_DATA_API_KEY is set

Returns a list of dicts, each with keys:
  id, ticker, headline, body, published_at, source, url, score
  plus a 'source_ref' dict: {ticker, channel, record_id}
"""

import os
import json
import math
import time
import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path to fixture file (relative to project root)
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent.parent / "data"
_FIXTURES_PATH = _DATA_DIR / "news_fixtures.json"


# ---------------------------------------------------------------------------
# TF-IDF helpers (pure Python, no dependencies)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _build_idf(corpus: list[list[str]]) -> dict[str, float]:
    """Compute inverse document frequency for each token across the corpus."""
    N = len(corpus)
    df: dict[str, int] = {}
    for doc_tokens in corpus:
        for tok in set(doc_tokens):
            df[tok] = df.get(tok, 0) + 1
    return {tok: math.log((N + 1) / (count + 1)) + 1 for tok, count in df.items()}


def _tf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Compute TF-IDF vector for a list of tokens."""
    tf: dict[str, int] = {}
    for tok in tokens:
        tf[tok] = tf.get(tok, 0) + 1
    return {tok: (count / len(tokens)) * idf.get(tok, 1.0) for tok, count in tf.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    dot = sum(a.get(k, 0.0) * v for k, v in b.items())
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Offline search (TF-IDF over fixtures)
# ---------------------------------------------------------------------------

def _load_fixtures() -> list[dict]:
    with open(_FIXTURES_PATH, "r") as f:
        return json.load(f)


def _search_offline(query: str, ticker: Optional[str], top_k: int) -> list[dict]:
    articles = _load_fixtures()

    # Filter by ticker if specified
    if ticker:
        ticker_upper = ticker.upper()
        articles = [a for a in articles if a["ticker"].upper() == ticker_upper]

    if not articles:
        return []

    # Build corpus from headline + body
    corpus_texts = [a["headline"] + " " + a["body"] for a in articles]
    corpus_tokens = [_tokenize(t) for t in corpus_texts]
    idf = _build_idf(corpus_tokens)

    query_vec = _tf_vector(_tokenize(query), idf)
    scored = []
    for i, (article, tokens) in enumerate(zip(articles, corpus_tokens)):
        doc_vec = _tf_vector(tokens, idf)
        score = _cosine(query_vec, doc_vec)
        scored.append((score, article))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, article in scored[:top_k]:
        result = dict(article)
        result["score"] = round(score, 4)
        result["source_ref"] = {
            "ticker": article["ticker"],
            "channel": "news",
            "record_id": article["id"],
        }
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Online search (financial data API)
# ---------------------------------------------------------------------------

# Set NEWS_API_URL (and FINANCIAL_DATA_API_KEY) to point at your data provider.
# Example: NEWS_API_URL=https://api.yourprovider.com/v2/news
_NEWS_API_URL = os.environ.get("NEWS_API_URL", "")


def _search_online(query: str, ticker: Optional[str], top_k: int) -> list[dict]:
    """Call the financial data news API. Falls back to offline on any error."""
    try:
        import requests  # noqa: PLC0415
        api_key = os.environ["FINANCIAL_DATA_API_KEY"]
        params = {
            "token": api_key,
            "pageSize": top_k,
            "tickers": ticker.upper() if ticker else None,
            "searchFields": "all",
            "q": query,
        }
        params = {k: v for k, v in params.items() if v is not None}
        resp = requests.get(
            _NEWS_API_URL,
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json()
        articles = raw if isinstance(raw, list) else raw.get("data", [])
        results = []
        for item in articles[:top_k]:
            result = {
                "id": str(item.get("id", "")),
                "ticker": ticker or (item.get("stocks", [{}])[0].get("name", "") if item.get("stocks") else ""),
                "headline": item.get("title", ""),
                "body": item.get("body", item.get("teaser", "")),
                "published_at": item.get("created", ""),
                "source": item.get("source", "Financial News"),
                "url": item.get("url", ""),
                "score": 1.0,
                "source_ref": {
                    "ticker": ticker or "",
                    "channel": "news",
                    "record_id": str(item.get("id", "")),
                },
            }
            results.append(result)
        return results
    except Exception as exc:  # noqa: BLE001
        # Degrade gracefully: return offline results
        print(f"[search_news] Online fetch failed ({exc}), falling back to offline.")
        return _search_offline(query, ticker, top_k)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_news(query: str, ticker: Optional[str] = None, top_k: int = 3) -> dict:
    """
    Search financial news articles.

    Args:
        query:  Natural-language search query.
        ticker: Optional ticker symbol to filter results.
        top_k:  Maximum number of results to return (default 3).

    Returns:
        {
          "results":   list of article dicts (with source_ref),
          "mode":      "offline" | "online",
          "tool":      "search_news",
          "query":     original query,
          "ticker":    ticker filter (or None),
        }
    """
    t0 = time.time()
    mode = "online" if os.environ.get("FINANCIAL_DATA_API_KEY") else "offline"

    if mode == "online":
        results = _search_online(query, ticker, top_k)
    else:
        results = _search_offline(query, ticker, top_k)

    return {
        "results": results,
        "mode": mode,
        "tool": "search_news",
        "query": query,
        "ticker": ticker,
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    out = search_news("data center revenue performance", ticker="NVDA")
    print(json.dumps(out, indent=2))
