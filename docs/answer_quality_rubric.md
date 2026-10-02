# Answer Quality Rubric: Relevance & Completeness

This document defines the formal rubric for evaluating financial research answers produced by the assistant systems (e.g. S1 Deterministic Formatter vs. S6 LLM Synthesizer vs. S7 Hybrid Verifier).

To ensure objective evaluation and avoid self-evaluation bias:
1. **Independent Model Judge**: All automated LLM-as-a-judge scores on the DEV split are evaluated using **Llama 3.1 8B** (`llama3.1:8b`), a strictly different model family from the answering synthesizer (**Qwen 2.5 7B**).
2. **Judge Validation**: A 20-answer subset is independently hand-scored by human annotators using this exact rubric to compute human-judge agreement (MAE and Spearman correlation).
3. **Split Blinding**: Answer quality tuning and validation are conducted exclusively on the `DEV` split ($n=45$).

---

## 1. Rubric Dimensions

### Dimension A: Query Relevance (1 – 5)
Measures whether the response directly addresses the specific user intent, entity, and question asked, without tangential commentary or unrelated retrieved noise.

| Score | Description | Behavioral Anchors |
| :--- | :--- | :--- |
| **1 (Irrelevant)** | Completely fails to address query | Discusses wrong company, answers unrelated financial metric, or fabricates irrelevant prose. |
| **2 (Marginally Relevant)** | Weak overlap with query intent | Mentions the correct company name but fails to answer the question, or responds with tangential news articles. |
| **3 (Moderately Relevant)** | Answers core question with notable noise | Provides answer to the core question but includes substantial unrelated background or tangential commentary. |
| **4 (Relevant & Focused)** | Directly addresses the query | Focuses cleanly on the requested entities and questions; minor extraneous detail. |
| **5 (Highly Relevant & Precise)** | Perfectly targeted to intent | Directly, concisely, and precisely answers the exact query requested. If the query asks for a specific quarter, only that quarter is addressed. If query cannot be answered, cleanly abstains. |

---

### Dimension B: Information Completeness / Fact Coverage (1 – 5)
Measures whether the response contains all essential financial facts, metrics, comparisons, and nuances required to provide an institutional-grade answer based on available evidence.

| Score | Description | Behavioral Anchors |
| :--- | :--- | :--- |
| **1 (Incomplete / Empty)** | Contains no useful financial facts | Fails to report any requested metric, revenue, EPS, rating, or price target when data was available. |
| **2 (Severely Deficient)** | Missing primary figures | Mentions whether a company beat or missed but omits the actual financial numbers (e.g., revenue/EPS values). |
| **3 (Partially Complete)** | Covers primary figures, misses details | Reports top-line numbers (e.g., actual revenue) but omits estimates, consensus comparisons, or guidance. |
| **4 (Substantially Complete)** | Covers all core required metrics | Reports actuals vs. estimates, percentage surprises, and key management commentary/notes. |
| **5 (Comprehensive Coverage)** | Institutional completeness | Reports actual vs. consensus, beat/miss magnitude, guidance changes, and drivers/synergies without omission. On unanswerable queries (`no_data`), explicitly explains the absence of data. |

---

## 2. Combined Score
The overall **Answer Quality Score** is the arithmetic mean:
$$\text{Quality Score} = \frac{\text{Relevance} + \text{Completeness}}{2} \in [1.0, 5.0]$$

---

## 3. Judge Prompt Template (Llama 3.1 8B)

```text
You are an expert financial research evaluator. Your task is to score the quality of an AI research assistant's answer on two criteria: Query Relevance (1-5) and Information Completeness (1-5).

User Query: {query}
Retrieved Tool Records: {evidence_summary}
Assistant Answer: {answer}

Scoring Rubric:
- Relevance (1-5): Does the answer directly address the specific entity, topic, and intent of the user query? (1 = completely irrelevant/wrong company, 5 = perfectly focused and targeted).
- Completeness (1-5): Does the answer provide the necessary financial figures, beats/misses, and context to satisfy the query based on the evidence? On queries with no relevant data, honest abstention scores 5, whereas hallucinating facts scores 1. (1 = missing all figures, 5 = comprehensive institutional coverage).

Respond ONLY with valid JSON in this exact format:
{
  "relevance": <int 1-5>,
  "completeness": <int 1-5>,
  "rationale": "<1-2 sentence justification>"
}
```
