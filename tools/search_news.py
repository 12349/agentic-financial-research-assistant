"""
tools/search_news.py

Searches financial news articles using vector embeddings (all-MiniLM-L6-v2 + FAISS IndexFlatIP)
with automated fallback to TF-IDF similarity.

Offline mode (default):  loads data/news_fixtures.json into in-memory FAISS index
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
from typing import Optional, Tuple

# Suppress the HuggingFace Hub "unauthenticated requests" warning that appears on
# terminals without HF_TOKEN set. This is output suppression only — model loading
# behaviour is unchanged. Set the env var before any huggingface_hub imports fire.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Path to fixture file (relative to project root)
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent.parent / "data"
_FIXTURES_PATH = _DATA_DIR / "news_fixtures.json"

# Threshold for semantic cosine similarity (range: -1.0 to 1.0; typically 0.0 to 1.0)
DEFAULT_SIMILARITY_THRESHOLD = float(os.environ.get("NEWS_SIMILARITY_THRESHOLD", "0.30"))
# Threshold for TF-IDF cosine similarity fallback
DEFAULT_TFIDF_THRESHOLD = float(os.environ.get("NEWS_TFIDF_THRESHOLD", "0.05"))


# ---------------------------------------------------------------------------
# Fixture loading helper
# ---------------------------------------------------------------------------

def _load_fixtures() -> list[dict]:
    with open(_FIXTURES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# In-Memory Semantic Index (FAISS IndexFlatIP + all-MiniLM-L6-v2)
# ---------------------------------------------------------------------------

class _SemanticNewsIndex:
    """
    In-memory vector store built once at startup/import time.
    Uses SentenceTransformer ('all-MiniLM-L6-v2') and faiss.IndexFlatIP.
    Normalized embeddings ensure inner product equals cosine similarity.
    """

    def __init__(self):
        self._initialized = False
        self._model = None
        self._index = None
        self._articles = []
        self._available = False
        self._init_error = None

    def ensure_initialized(self):
        if self._initialized:
            return

        engine_config = os.environ.get("NEWS_SEARCH_ENGINE", "semantic").lower()
        if engine_config == "tfidf":
            self._available = False
            self._init_error = "NEWS_SEARCH_ENGINE configured to tfidf"
            self._initialized = True
            return

        try:
            import numpy as np
            import faiss
            # Suppress the HuggingFace Hub "unauthenticated requests" warning that
            # appears on terminals without HF_TOKEN set. Model loading is unchanged.
            import logging as _logging
            _logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)
            from sentence_transformers import SentenceTransformer

            articles = _load_fixtures()
            if not articles:
                self._articles = []
                self._index = None
                self._available = True
                self._initialized = True
                return

            corpus_texts = [f"{a['headline']}. {a['body']}" for a in articles]
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(
                corpus_texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)

            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)

            self._model = model
            self._index = index
            self._articles = articles
            self._available = True
            self._initialized = True
        except Exception as exc:
            self._available = False
            self._init_error = str(exc)
            self._initialized = True
            print(f"[search_news] Semantic search initialization unavailable ({exc}). Using TF-IDF fallback.")

    @property
    def is_available(self) -> bool:
        self.ensure_initialized()
        return self._available


# Global singleton instance
_semantic_index = _SemanticNewsIndex()


def _search_semantic(query: str, ticker: Optional[str], top_k: int, threshold: float) -> list[dict]:
    """Execute vector similarity search over FAISS in-memory index."""
    import numpy as np

    idx = _semantic_index
    if not idx.is_available or idx._index is None or not idx._articles:
        return []

    # Encode query with L2-normalization for cosine similarity
    query_vec = idx._model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    # Search all indexed documents
    k_candidates = len(idx._articles)
    distances, indices = idx._index.search(query_vec, k_candidates)

    ticker_upper = ticker.upper() if ticker else None
    results = []

    for score, doc_idx in zip(distances[0], indices[0]):
        if doc_idx < 0 or doc_idx >= len(idx._articles):
            continue

        article = idx._articles[doc_idx]

        # Apply ticker filter if specified
        if ticker_upper and article["ticker"].upper() != ticker_upper:
            continue

        # Apply similarity threshold guardrail
        if float(score) < threshold:
            continue

        result = dict(article)
        result["score"] = round(float(score), 4)
        result["source_ref"] = {
            "ticker": article["ticker"],
            "channel": "news",
            "record_id": article["id"],
        }
        results.append(result)

        if len(results) >= top_k:
            break

    return results


# ---------------------------------------------------------------------------
# TF-IDF Fallback helpers (pure Python, zero external dependencies)
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


def _search_tfidf(query: str, ticker: Optional[str], top_k: int, threshold: float) -> list[dict]:
    """Fallback offline search using pure-Python TF-IDF."""
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
    for article, tokens in zip(articles, corpus_tokens):
        doc_vec = _tf_vector(tokens, idf)
        score = _cosine(query_vec, doc_vec)
        if score >= threshold:
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
# Offline Search Dispatcher
# ---------------------------------------------------------------------------

def _search_offline(query: str, ticker: Optional[str], top_k: int) -> Tuple[list[dict], str]:
    if _semantic_index.is_available:
        return _search_semantic(query, ticker, top_k, DEFAULT_SIMILARITY_THRESHOLD), "semantic"
    return _search_tfidf(query, ticker, top_k, DEFAULT_TFIDF_THRESHOLD), "tfidf"


# ---------------------------------------------------------------------------
# Online search (financial data API)
# ---------------------------------------------------------------------------

_NEWS_API_URL = os.environ.get("NEWS_API_URL", "")


def _search_online(query: str, ticker: Optional[str], top_k: int) -> Tuple[list[dict], str]:
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
        return results, "live_api"
    except Exception as exc:  # noqa: BLE001
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
          "results":    list of article dicts (with source_ref),
          "mode":       "offline" | "online",
          "engine":     "semantic" | "tfidf" | "live_api",
          "tool":       "search_news",
          "query":      original query,
          "ticker":     ticker filter (or None),
          "latency_ms": elapsed time in milliseconds,
        }
    """
    t0 = time.time()
    mode = "online" if os.environ.get("FINANCIAL_DATA_API_KEY") else "offline"

    if mode == "online":
        results, engine = _search_online(query, ticker, top_k)
    else:
        results, engine = _search_offline(query, ticker, top_k)

    return {
        "results": results,
        "mode": mode,
        "engine": engine,
        "tool": "search_news",
        "query": query,
        "ticker": ticker,
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }


# Pre-initialize semantic index at module import time
_semantic_index.ensure_initialized()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    out = search_news("data center revenue performance", ticker="NVDA")
    print(json.dumps(out, indent=2))
