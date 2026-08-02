"""
classifier/predict.py

Inference wrapper for the fine-tuned DistilBERT sentiment classifier.

Usage:
    from classifier.predict import classify_sentiment
    label, confidence = classify_sentiment("Raised PT on strong demand.")
    # Returns e.g. ("bullish", 0.94)

Falls back to None if model weights are not present (classifier/model/ is gitignored).
Caller should handle None by falling back to rule-based keyword bucketing.
"""

import os
from typing import Optional

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

# Lazy-loaded globals — loaded once on first call
_model     = None
_tokenizer = None
_device    = None

# Label mapping matching train.py
_ID_TO_LABEL = {0: "bearish", 1: "neutral", 2: "bullish"}


def _is_model_available() -> bool:
    """Check if saved model weights exist."""
    return os.path.isfile(os.path.join(_MODEL_DIR, "config.json"))


def _load_model():
    """Lazy-load model and tokenizer on first inference call."""
    global _model, _tokenizer, _device
    if _model is not None:
        return  # already loaded

    import torch  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
    )

    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = DistilBertTokenizerFast.from_pretrained(_MODEL_DIR)
    _model     = DistilBertForSequenceClassification.from_pretrained(_MODEL_DIR)
    _model.to(_device)
    _model.eval()


def classify_sentiment(text: str) -> Optional[tuple[str, float]]:
    """
    Classify financial text sentiment.

    Args:
        text: Analyst note, headline, or rating commentary (plain text)

    Returns:
        (label, confidence) where label is "bullish" | "neutral" | "bearish"
        Returns None if model weights are not available (fallback to rules).
    """
    if not _is_model_available():
        return None

    try:
        _load_model()
        import torch  # noqa: PLC0415
        enc = _tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(_device)
        attention_mask = enc["attention_mask"].to(_device)

        with torch.no_grad():
            logits = _model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs  = torch.softmax(logits, dim=-1)[0]
            pred   = int(probs.argmax().item())
            conf   = float(probs[pred].item())

        return _ID_TO_LABEL[pred], conf

    except Exception as exc:  # noqa: BLE001
        print(f"[classifier/predict] Inference failed ({exc}), returning None for fallback.")
        return None


def classify_batch(texts: list[str]) -> list[Optional[tuple[str, float]]]:
    """
    Classify a batch of texts. More efficient than calling classify_sentiment in a loop.

    Returns list of (label, confidence) or None per item.
    """
    if not _is_model_available():
        return [None] * len(texts)

    try:
        _load_model()
        import torch  # noqa: PLC0415
        enc = _tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(_device)
        attention_mask = enc["attention_mask"].to(_device)

        with torch.no_grad():
            logits = _model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs  = torch.softmax(logits, dim=-1)
            preds  = probs.argmax(dim=-1).tolist()
            confs  = probs.max(dim=-1).values.tolist()

        return [(_ID_TO_LABEL[p], c) for p, c in zip(preds, confs)]

    except Exception as exc:  # noqa: BLE001
        print(f"[classifier/predict] Batch inference failed ({exc}), returning None list.")
        return [None] * len(texts)


if __name__ == "__main__":
    # Quick smoke test
    test_cases = [
        "Raised PT on data center demand strength and Blackwell ramp confirmation.",
        "Cautious on near-term EV competition; autonomous timeline uncertain.",
        "No change to rating; monitoring macro environment.",
        "Strong capital markets rebound; credit quality better than feared.",
        "Oil price uncertainty from OPEC policy caps upside.",
    ]
    print(f"Model available: {_is_model_available()}")
    print()
    for text in test_cases:
        result = classify_sentiment(text)
        if result:
            label, conf = result
            print(f"  [{label:8s} {conf:.2f}]  {text[:70]}")
        else:
            print(f"  [NO MODEL]  {text[:70]}")
