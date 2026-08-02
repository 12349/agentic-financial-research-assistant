"""
classifier/evaluate.py

Re-evaluates the saved DistilBERT model on the held-out test set.
Use this to verify a freshly re-trained model after running train.py.

Run with: PYTHONPATH=. python3 classifier/evaluate.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

LABEL_TO_ID = {"bearish": 0, "neutral": 1, "bullish": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
TFNS_TO_LABEL = {0: "bearish", 1: "bullish", 2: "neutral"}


def evaluate():
    if not os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
        print(f"ERROR: No model found at {MODEL_DIR}/")
        print("Run `PYTHONPATH=. python3 classifier/train.py` first.")
        sys.exit(1)

    import torch  # noqa: PLC0415
    from datasets import load_dataset  # noqa: PLC0415
    from sklearn.metrics import classification_report  # noqa: PLC0415
    from sklearn.model_selection import train_test_split  # noqa: PLC0415
    from torch.utils.data import Dataset, DataLoader  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
    )

    # Load same dataset used in training (to reproduce the split)
    print("Loading dataset...")
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
    texts  = [row["text"] for row in ds]
    labels = [TFNS_TO_LABEL[row["label"]] for row in ds]

    # Fixture examples (same as train.py)
    FIXTURE_EXAMPLES = [
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
        ("Cautious on near-term EV competition; autonomous timeline uncertain.", "bearish"),
        ("Oil price uncertainty from OPEC policy caps upside; disciplined capex a positive.", "bearish"),
        ("Exxon Mobil Warns OPEC Meeting Outcome Could Pressure Oil Prices Further", "bearish"),
        ("OPEC Postpones Production Increases as Oil Prices Slip; Exxon Benefits from Capital Discipline", "bearish"),
        ("CEO Dimon warned of geopolitical risks and potential credit deterioration ahead.", "bearish"),
        ("Maintains price target; monitoring macro environment.", "neutral"),
        ("No change to rating; await next quarter results.", "neutral"),
        ("Exxon Q3 Earnings Beat on Pioneer Synergies; OPEC Uncertainty Clouds Q4 Outlook", "neutral"),
        ("OPEC Uncertainty Clouds Q4 Outlook for energy sector.", "neutral"),
        ("Nvidia Blackwell GPU Demand Described as 'Insane' as Supply Constraints Ease", "bullish"),
    ]
    texts  += [t for t, _ in FIXTURE_EXAMPLES]
    labels += [l for _, l in FIXTURE_EXAMPLES]

    # Reproduce exact same test split (same random_state=42)
    label_ids = [LABEL_TO_ID[l] for l in labels]
    _, X_test, _, y_test = train_test_split(
        texts, label_ids, test_size=0.2, random_state=42, stratify=label_ids
    )
    print(f"Evaluating on {len(X_test)} held-out test samples...")

    # Load saved model
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model     = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    class FinDataset(Dataset):
        def __init__(self, texts, labels, tokenizer):
            enc = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
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

    test_ds     = FinDataset(X_test, y_test, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=64)

    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            preds  = model(input_ids=ids, attention_mask=mask).logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(labels.cpu().tolist())

    report = classification_report(all_true, all_preds, target_names=["bearish", "neutral", "bullish"])
    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    evaluate()
