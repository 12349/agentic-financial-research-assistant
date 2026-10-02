"""
experiments/ollama_client.py

Thin wrapper around the Ollama REST API.

CRITICAL: This module uses the REST API directly (not `ollama run` CLI).
The CLI triggers unbounded extended thinking mode on qwen3.6, inflating
latency by 5-30x for structured JSON tasks. The REST API allows think=False.

All calls use:
  - temperature = 0       (deterministic)
  - seed        = 42      (reproducible; Ollama honours this for most models)
  - think       = False   (disables chain-of-thought for qwen3.6)
  - num_predict = varies  (capped; prevents runaway generation)

Usage:
    from experiments.ollama_client import OllamaClient
    client = OllamaClient(model="qwen3.6:latest")
    result = client.generate("Output JSON only: ...")
    print(result.response)          # str
    print(result.tok_per_sec)       # float
    print(result.elapsed_ms)        # int
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from experiments.config import (
    OLLAMA_BASE_URL,
    OLLAMA_SEED,
    OLLAMA_TEMPERATURE,
    PRIMARY_MODEL,
    SECOND_MODEL,
)


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class OllamaResponse:
    response: str               # Generated text (thinking tokens stripped)
    prompt_tokens: int
    gen_tokens: int
    elapsed_ms: int
    tok_per_sec: float
    model: str
    done: bool
    raw: dict                   # Full Ollama JSON response, for audit


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Calls the Ollama REST API /api/generate endpoint.

    Parameters
    ----------
    model : str
        Ollama model name. Must already be pulled locally.
    base_url : str
        Ollama server URL (default: http://localhost:11434).
    temperature : float
        Sampling temperature. Use 0 for deterministic output.
    seed : int
        Random seed. Passed to Ollama options; honoured for reproducibility.
    think : bool
        If False (default), disables extended thinking mode on qwen3.x models.
        MUST be False for structured JSON tasks — thinking mode generates
        thousands of reasoning tokens before output, taking minutes per call.
    num_predict : int
        Hard cap on generated tokens. 256 is sufficient for tool plans;
        1024 for synthesis.
    timeout : int
        HTTP timeout in seconds. Use ≥300 for first cold-start load.
    """

    def __init__(
        self,
        model: str = PRIMARY_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = OLLAMA_TEMPERATURE,
        seed: int = OLLAMA_SEED,
        think: bool = False,
        num_predict: int = 256,
        timeout: int = 300,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.think = think
        self.num_predict = num_predict
        self.timeout = timeout
        self._endpoint = f"{self.base_url}/api/generate"

    def generate(self, prompt: str, num_predict: Optional[int] = None) -> OllamaResponse:
        """
        Send a generation request and return an OllamaResponse.

        Raises
        ------
        RuntimeError
            If the HTTP request fails or Ollama returns a non-200 status.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": self.think,     # False = disable extended thinking
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_predict": num_predict if num_predict is not None else self.num_predict,
            },
        }

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw_bytes = resp.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama request failed: {exc}. "
                f"Is Ollama running at {self.base_url}? "
                f"Run: ollama serve"
            ) from exc

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        try:
            raw = json.loads(raw_bytes)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Could not parse Ollama response JSON: {exc}") from exc

        if "error" in raw:
            raise RuntimeError(f"Ollama returned error: {raw['error']}")

        response_text = raw.get("response", "").strip()
        prompt_tokens = raw.get("prompt_eval_count", 0)
        gen_tokens = raw.get("eval_count", 0)
        eval_ns = raw.get("eval_duration", 1)
        tok_per_sec = round(gen_tokens / (eval_ns / 1e9), 1) if eval_ns else 0.0

        return OllamaResponse(
            response=response_text,
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            elapsed_ms=elapsed_ms,
            tok_per_sec=tok_per_sec,
            model=self.model,
            done=raw.get("done", True),
            raw=raw,
        )

    def is_available(self) -> bool:
        """Return True if the Ollama server is reachable and the model is loaded."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            model_names = [m.get("name", "") for m in data.get("models", [])]
            # Match on prefix (e.g. "qwen3.6" matches "qwen3.6:latest")
            short = self.model.split(":")[0]
            return any(short in name for name in model_names)
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def planner_client(model: str = PRIMARY_MODEL) -> OllamaClient:
    """Client for tool-plan generation. Short output, think=False."""
    return OllamaClient(model=model, think=False, num_predict=256)


def synthesizer_client(model: str = PRIMARY_MODEL) -> OllamaClient:
    """Client for answer synthesis. Longer output, think=False."""
    return OllamaClient(model=model, think=False, num_predict=1024)


def rater_client(model: str = SECOND_MODEL) -> OllamaClient:
    """Client for second-rater label pass (llama3.1:8b). think=False."""
    return OllamaClient(model=model, think=False, num_predict=256)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else PRIMARY_MODEL
    print(f"Smoke-testing: {model}")
    print(f"Endpoint:      {OLLAMA_BASE_URL}/api/generate")
    print(f"think=False, temperature=0, seed=42, num_predict=128")
    print()

    client = OllamaClient(model=model, num_predict=128)

    if not client.is_available():
        print(f"ERROR: Model '{model}' not found in Ollama. Run: ollama pull {model}")
        sys.exit(1)

    prompt = (
        "Output ONLY a raw JSON array, no explanation, no markdown. "
        "Query: 'What is the current analyst sentiment on Tesla stock?' "
        "Tools: search_news, get_ratings, get_guidance, get_earnings. "
        "Return: [{\"tool\": \"<name>\", \"args\": {\"ticker\": \"<ticker>\"}}]"
    )

    result = client.generate(prompt, num_predict=128)

    print(f"Response:     {result.response[:200]}")
    print(f"Elapsed:      {result.elapsed_ms}ms")
    print(f"Speed:        {result.tok_per_sec} tok/s")
    print(f"Tokens:       prompt={result.prompt_tokens}  gen={result.gen_tokens}")

    try:
        parsed = json.loads(result.response)
        assert isinstance(parsed, list)
        print(f"JSON valid:   ✅  {parsed}")
    except Exception:
        print(f"JSON valid:   ⚠️  Not clean JSON (parser fallback will handle)")
