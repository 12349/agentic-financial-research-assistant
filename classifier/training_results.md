# Sentiment Classifier — Training Results

**Model:** DistilBERT (`distilbert-base-uncased`), fine-tuned for sequence classification  
**Task:** 3-class financial sentiment — `bullish` / `neutral` / `bearish`  
**Dataset:** Twitter Financial News Sentiment (zeroshot/twitter-financial-news-sentiment) + 25 hand-labeled fixture headlines  
**Total samples:** 9568 (7654 train / 1914 test, 80/20 stratified split)  
**Training:** 3 epochs, batch size 32, lr=2e-5, AdamW + linear warmup, CPU  
**Training time:** 541.4s

---

## Per-class metrics (held-out test set)

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| bearish | 0.702 | 0.751 | 0.726 | 289 |
| neutral | 0.900 | 0.899 | 0.899 | 1237 |
| bullish | 0.808 | 0.768 | 0.787 | 388 |

## Aggregate metrics

| Metric | Weighted avg | Macro avg |
|--------|-------------|-----------|
| Precision | 0.851 | 0.803 |
| Recall    | 0.850 | 0.806 |
| F1-score  | 0.850 | 0.804 |

## Full sklearn classification report

```
              precision    recall  f1-score   support

     bearish       0.70      0.75      0.73       289
     neutral       0.90      0.90      0.90      1237
     bullish       0.81      0.77      0.79       388

    accuracy                           0.85      1914
   macro avg       0.80      0.81      0.80      1914
weighted avg       0.85      0.85      0.85      1914

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
