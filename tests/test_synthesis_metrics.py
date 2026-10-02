"""
tests/test_synthesis_metrics.py

Unit tests for synthesis quality metrics:
- Known-good grounded answers
- Known-bad hallucinated numbers
- Invalid/invented citations
- Uncited claims
- Abstention edge cases
"""

import math
import pytest
from experiments.metrics.synthesis_quality import (
    abstention_correctness,
    citation_validity,
    numeric_faithfulness,
    unsupported_claim_rate,
)

TOOL_OUTPUTS = [
    {
        "tool": "get_earnings",
        "ticker": "NVDA",
        "earnings": [
            {
                "period": "Q3 FY2025",
                "revenue_actual": 35082000000.0,
                "revenue_estimate": 33160000000.0,
                "eps_actual": 0.81,
                "eps_estimate": 0.75,
                "revenue_surprise_pct": 5.8,
                "source_ref": {"record_id": "earnings-nvda-001"},
            }
        ],
    },
    {
        "tool": "search_news",
        "ticker": "NVDA",
        "results": [
            {
                "id": "news-nvda-001",
                "headline": "Nvidia Data Center Revenue Hits Record $22.6B in Q3",
                "body": "Nvidia delivered record data center revenue of $22.6 billion, up 112% year-over-year.",
                "source_ref": {"record_id": "news-nvda-001"},
            }
        ],
    },
]


def test_known_good_answer():
    """Formatter-style answer: numbers and citations are directly from source."""
    answer = (
        "[Source: earnings-nvda-001] NVDA Q3 FY2025 earnings: Revenue $35.1B vs estimate $33.2B. "
        "EPS $0.81 vs estimate $0.75 (+5.8%).\n\n"
        "[Source: news-nvda-001] Nvidia Data Center Revenue Hits Record $22.6B in Q3\n"
        "Nvidia delivered record data center revenue of $22.6 billion, up 112% year-over-year."
    )

    cit = citation_validity(answer, TOOL_OUTPUTS)
    assert cit["valid_fraction"] == 1.0
    assert len(cit["invalid_ids"]) == 0

    faith = numeric_faithfulness(answer, TOOL_OUTPUTS)
    assert faith["faithful_fraction"] == 1.0
    assert len(faith["unfaithful_values"]) == 0

    unsup = unsupported_claim_rate(answer)
    assert unsup["unsupported_rate"] == 0.0


def test_known_bad_hallucinated_numbers():
    """Answer contains invented numbers ($99.9B, $4.99)."""
    answer = (
        "[Source: earnings-nvda-001] NVDA revenue was $99.9B and EPS was $4.99."
    )
    faith = numeric_faithfulness(answer, TOOL_OUTPUTS)
    assert faith["faithful_fraction"] == 0.0
    assert 99.9 in faith["unfaithful_values"]
    assert 4.99 in faith["unfaithful_values"]


def test_known_bad_fabricated_citation():
    """Answer cites non-existent source record ID."""
    answer = (
        "[Source: fake-source-999] NVDA reported strong data center demand."
    )
    cit = citation_validity(answer, TOOL_OUTPUTS)
    assert cit["valid_fraction"] == 0.0
    assert "fake-source-999" in cit["invalid_ids"]


def test_uncited_claims():
    """Free text with no citations should have a high unsupported claim rate."""
    answer = (
        "Nvidia is doing very well in the market. "
        "Analysts expect massive stock appreciation next year. "
        "The supply chain issues have completely resolved."
    )
    unsup = unsupported_claim_rate(answer)
    assert unsup["unsupported_rate"] == 1.0


def test_abstention_correct():
    """Correct abstention when no data is available."""
    answer = "No data was found to answer this query.\n\nNo news data was available for this query."
    abst = abstention_correctness(answer, "no_data")
    assert abst["applicable"] is True
    assert abst["correct"] is True


def test_abstention_failed():
    """Failure to abstain when given no_data query."""
    answer = "[Source: news-nvda-001] Nvidia announced record earnings for the quarter."
    abst = abstention_correctness(answer, "no_data")
    assert abst["applicable"] is True
    assert abst["correct"] is False


def test_abstention_not_applicable():
    """Abstention should not apply to in-scope queries."""
    abst = abstention_correctness("Some answer", "single_tool")
    assert abst["applicable"] is False
