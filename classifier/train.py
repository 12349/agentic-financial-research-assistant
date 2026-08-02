"""
classifier/train.py

Fine-tunes DistilBERT on the Twitter Financial News Sentiment dataset
(zeroshot/twitter-financial-news-sentiment, 9,543 samples) supplemented
with manually labeled fixture headlines from our own project data.

Dataset: https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment
Labels:  0=bearish | 1=bullish | 2=neutral

Run with: PYTHONPATH=. python3 classifier/train.py

Outputs:
  classifier/model/          — saved model weights + tokenizer
  classifier/training_results.md — real precision/recall/F1 metrics
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_results.md")

# Label mapping: Twitter Financial News Sentiment → our domain labels
# TFNS: 0=bearish, 1=bullish, 2=neutral
TFNS_TO_LABEL = {0: "bearish", 1: "bullish", 2: "neutral"}
LABEL_TO_ID  = {"bearish": 0, "neutral": 1, "bullish": 2}
ID_TO_LABEL  = {v: k for k, v in LABEL_TO_ID.items()}

# ---------------------------------------------------------------------------
# Supplemental labeled examples from our own fixture headlines + analyst notes
# Labeled by hand: these are the "our data" contribution on top of FPB
# ---------------------------------------------------------------------------
FIXTURE_EXAMPLES = [
    # Bullish analyst notes
    ("Raised PT on data center demand strength and Blackwell ramp confirmation.", "bullish"),
    ("Sees Blackwell cycle as multi-year AI infrastructure tailwind.", "bullish"),
    ("Supply constraints easing; data center demand exceeds prior estimates.", "bullish"),
    ("Q4 guidance above Street; raises FY2026 estimates by 15%.", "bullish"),
    ("Tesla AI and FSD monetization opportunity undervalued; Cybercab a long-term catalyst.", "bullish"),
    ("Full self-driving revenue model gaining credibility post-Cybercab reveal.", "bullish"),
    ("Best-in-class franchise; NII guidance raise signals sustained earnings power.", "bullish"),
    ("Investment banking recovery and NII strength drive estimate upgrades.", "bullish"),
    ("Strong capital markets rebound; credit quality better than feared.", "bullish"),
    ("Pioneer integration tracking ahead of plan; structural cost advantages intact.", "bullish"),
    ("Nvidia Data Center Revenue Hits Record $22.6B in Q3 FY2025, Surpassing Analyst Expectations", "bullish"),
    ("Tesla Q3 Deliveries Beat Expectations at 462,890 Vehicles", "bullish"),
    ("JPMorgan Q3 Net Income Rises 35% on Higher Interest Income", "bullish"),
    ("JPMorgan Raises Full-Year Net Interest Income Guidance to $92.5B", "bullish"),
    ("Nvidia Q4 Guidance Tops Estimates; Analysts Raise Price Targets Across the Board", "bullish"),
    # Bearish analyst notes
    ("Cautious on near-term EV competition; autonomous timeline uncertain.", "bearish"),
    ("Oil price uncertainty from OPEC policy caps upside; disciplined capex a positive.", "bearish"),
    ("Exxon Mobil Warns OPEC Meeting Outcome Could Pressure Oil Prices Further", "bearish"),
    ("OPEC Postpones Production Increases as Oil Prices Slip; Exxon Benefits from Capital Discipline", "bearish"),
    ("CEO Dimon warned of geopolitical risks and potential credit deterioration ahead.", "bearish"),
    # Neutral notes
    ("Maintains price target; monitoring macro environment.", "neutral"),
    ("No change to rating; await next quarter results.", "neutral"),
    ("Exxon Q3 Earnings Beat on Pioneer Synergies; OPEC Uncertainty Clouds Q4 Outlook", "neutral"),
    ("OPEC Uncertainty Clouds Q4 Outlook for energy sector.", "neutral"),
    ("Nvidia Blackwell GPU Demand Described as 'Insane' as Supply Constraints Ease", "bullish"),
]


def load_financial_phrasebank():
    """Load Twitter Financial News Sentiment dataset (parquet-native, no legacy script)."""
    from datasets import load_dataset  # noqa: PLC0415
    print("Loading Twitter Financial News Sentiment dataset...")
    # zeroshot/twitter-financial-news-sentiment loads from parquet natively
    # 9,543 training samples — larger and more headline-like than FPB sentences_allagree
    ds    = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
    texts  = [row["text"] for row in ds]
    labels = [TFNS_TO_LABEL[row["label"]] for row in ds]
    print(f"  Loaded {len(texts)} samples")
    return texts, labels


def build_dataset(texts, labels):
    """Add fixture examples and return combined dataset."""
    fixture_texts   = [t for t, _ in FIXTURE_EXAMPLES]
    fixture_labels  = [l for _, l in FIXTURE_EXAMPLES]
    all_texts  = texts  + fixture_texts
    all_labels = labels + fixture_labels
    print(f"  + {len(fixture_texts)} fixture examples → total {len(all_texts)} samples")
    return all_texts, all_labels


def train_and_evaluate(all_texts, all_labels):
    """Fine-tune DistilBERT and return metrics."""
    from sklearn.model_selection import train_test_split  # noqa: PLC0415
    from sklearn.metrics import (  # noqa: PLC0415
        classification_report, precision_recall_fscore_support
    )
    import torch  # noqa: PLC0415
    from torch.utils.data import Dataset, DataLoader  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    from torch.optim import AdamW  # noqa: PLC0415

    # --- Train / test split (stratified, 20% test) ---
    label_ids = [LABEL_TO_ID[l] for l in all_labels]
    X_train, X_test, y_train, y_test = train_test_split(
        all_texts, label_ids, test_size=0.2, random_state=42, stratify=label_ids
    )
    print(f"\nSplit: {len(X_train)} train / {len(X_test)} test")
    print(f"  Train distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Test  distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # --- Tokenizer ---
    print("\nLoading tokenizer: distilbert-base-uncased")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # --- Dataset class ---
    class FinDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            enc = tokenizer(
                texts, truncation=True, padding=True,
                max_length=max_len, return_tensors="pt"
            )
            self.input_ids      = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.labels         = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids":      self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels":         self.labels[idx],
            }

    print("Tokenizing...")
    train_ds = FinDataset(X_train, y_train, tokenizer)
    test_ds  = FinDataset(X_test,  y_test,  tokenizer)

    BATCH_SIZE = 32
    EPOCHS     = 3
    LR         = 2e-5

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Model ---
    print("\nLoading DistilBERT for sequence classification (3 classes)...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3
    )
    model.to(device)

    total_steps = len(train_loader) * EPOCHS
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    # --- Training loop ---
    print(f"\nFine-tuning for {EPOCHS} epochs...")
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        elapsed  = round(time.time() - t_start, 1)
        print(f"  Epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  elapsed={elapsed}s")

    # --- Evaluation ---
    print("\nEvaluating on held-out test set...")
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(labels.cpu().tolist())

    label_names = ["bearish", "neutral", "bullish"]
    report = classification_report(
        all_true, all_preds,
        target_names=label_names,
        output_dict=True,
    )
    report_str = classification_report(
        all_true, all_preds,
        target_names=label_names,
    )
    print("\nClassification Report:")
    print(report_str)

    train_time = round(time.time() - t_start, 1)

    # --- Save model ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"\nSaving model to {MODEL_DIR}/ ...")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("  Saved.")

    return report, report_str, train_time, len(X_train), len(X_test)


def write_results(report, report_str, train_time, n_train, n_test, n_total):
    """Write training_results.md with real metrics."""
    w_avg = report["weighted avg"]
    macro = report["macro avg"]

    md = f"""# Sentiment Classifier — Training Results

**Model:** DistilBERT (`distilbert-base-uncased`), fine-tuned for sequence classification  
**Task:** 3-class financial sentiment — `bullish` / `neutral` / `bearish`  
**Dataset:** Twitter Financial News Sentiment (zeroshot/twitter-financial-news-sentiment) + {len(FIXTURE_EXAMPLES)} hand-labeled fixture headlines  
**Total samples:** {n_total} ({n_train} train / {n_test} test, 80/20 stratified split)  
**Training:** 3 epochs, batch size 32, lr=2e-5, AdamW + linear warmup, CPU  
**Training time:** {train_time}s

---

## Per-class metrics (held-out test set)

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| bearish | {report['bearish']['precision']:.3f} | {report['bearish']['recall']:.3f} | {report['bearish']['f1-score']:.3f} | {int(report['bearish']['support'])} |
| neutral | {report['neutral']['precision']:.3f} | {report['neutral']['recall']:.3f} | {report['neutral']['f1-score']:.3f} | {int(report['neutral']['support'])} |
| bullish | {report['bullish']['precision']:.3f} | {report['bullish']['recall']:.3f} | {report['bullish']['f1-score']:.3f} | {int(report['bullish']['support'])} |

## Aggregate metrics

| Metric | Weighted avg | Macro avg |
|--------|-------------|-----------|
| Precision | {w_avg['precision']:.3f} | {macro['precision']:.3f} |
| Recall    | {w_avg['recall']:.3f} | {macro['recall']:.3f} |
| F1-score  | {w_avg['f1-score']:.3f} | {macro['f1-score']:.3f} |

## Full sklearn classification report

```
{report_str}
```

## Notes

- The Twitter Financial News Sentiment dataset (`zeroshot/twitter-financial-news-sentiment`) contains
  9,543 short financial headlines labeled bearish/bullish/neutral by finance professionals.
  It is headline-length text — well-matched to our use case of classifying analyst commentary notes.
- The classifier is used to score analyst commentary notes (free-text) in `tools/get_ratings.py`,
  supplementing the structured rating field (Buy/Sell/Hold) with NLP-based sentiment on analyst rationale.
- Rule-based keyword bucketing remains the fallback if model weights are not present (`classifier/model/`).
- To reproduce: `PYTHONPATH=. python3 classifier/train.py`  
  Model weights are gitignored (large); re-generate locally before running online mode.
"""
    with open(RESULTS_PATH, "w") as f:
        f.write(md)
    print(f"\nTraining results written to {RESULTS_PATH}")


if __name__ == "__main__":
    texts, labels     = load_financial_phrasebank()
    all_texts, all_labels = build_dataset(texts, labels)
    n_total           = len(all_texts)
    report, report_str, train_time, n_train, n_test = train_and_evaluate(all_texts, all_labels)
    write_results(report, report_str, train_time, n_train, n_test, n_total)
    print("\nDone. Run `python3 classifier/evaluate.py` to re-evaluate a saved model.")
