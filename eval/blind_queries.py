"""
eval/blind_queries.py

15 held-out queries written BEFORE running the test.
These were designed to cover:
  - Novel / awkward phrasing (not in original 5+3 tuning set)
  - Adversarial inputs (future data bait, hallucination bait)
  - Unknown tickers
  - Compound queries (multi-tool)
  - Queries with no fixture data at all

DO NOT modify these to make tests pass. Run once, report honestly.

Grading criteria per query:
  PASS  - agent returns a sourced answer (or explicit no-data) with no hallucination
  FAIL  - empty answer, ungrounded claim, or completely wrong tool for the intent
  WARN  - right answer, questionable tool routing (still counts as pass)
"""

# fmt: off
BLIND_QUERIES = [
    {
        "id": "B01",
        "query": "Is Nvidia stock a buy right now?",
        "intent": "analyst_ratings",
        "acceptable_tools": ["get_ratings", "search_news"],
        "adversarial": False,
        "no_data_expected": False,
        "notes": "Phrasing uses 'buy' colloquially, not 'analyst sentiment'. "
                 "Rule table has 'rating/ratings/analyst' but not bare 'buy' as a ratings trigger.",
    },
    {
        "id": "B02",
        "query": "How has Tesla been performing lately?",
        "intent": "news",
        "acceptable_tools": ["search_news", "get_ratings", "get_earnings"],
        "adversarial": False,
        "no_data_expected": False,
        "notes": "Ambiguous intent — 'performing' could route to news or earnings. "
                 "Either with sources is acceptable.",
    },
    {
        "id": "B03",
        "query": "What revenue did JPMorgan report and how did that compare to what they told investors to expect?",
        "intent": "earnings_and_guidance",
        "acceptable_tools": ["get_earnings", "get_guidance"],
        "adversarial": False,
        "no_data_expected": False,
        "notes": "Multi-tool compound query. Phrased without keywords 'beat or miss' — "
                 "relies on 'report' + 'expect' to trigger earnings+guidance path.",
    },
    {
        "id": "B04",
        "query": "Will Exxon raise its dividend this year?",
        "intent": "news",
        "acceptable_tools": ["search_news"],
        "adversarial": False,
        "no_data_expected": False,
        "notes": "No dividend tool exists. Correct behavior: route to news and surface "
                 "any relevant information. Tests graceful handling of unsupported query type.",
    },
    {
        "id": "B05",
        "query": "What's the analyst view on Apple stock?",
        "intent": "analyst_ratings",
        "acceptable_tools": ["get_ratings", "search_news"],
        "adversarial": False,
        "no_data_expected": True,   # AAPL not in fixtures
        "notes": "AAPL is not in our fixture data. Should gracefully return "
                 "'no data' without crashing or hallucinating Apple-specific facts.",
    },
    {
        "id": "B06",
        "query": "What is XOM guiding for next quarter?",
        "intent": "guidance",
        "acceptable_tools": ["get_guidance"],
        "adversarial": False,
        "no_data_expected": False,
        "notes": "Clear guidance intent. 'Guiding for' — tests whether 'guid' keyword "
                 "triggers get_guidance correctly.",
    },
    {
        "id": "B07",
        "query": "By how much did Nvidia miss or beat analyst EPS estimates?",
        "intent": "earnings",
        "acceptable_tools": ["get_earnings"],
        "adversarial": False,
        "no_data_expected": False,
        "notes": "Awkward inverted phrasing ('miss or beat' instead of 'beat or miss'). "
                 "Tests keyword matching robustness.",
    },
    {
        "id": "B08",
        "query": "How do Wall Street analysts feel about JPMorgan, and did the bank's actual Q3 numbers back that up?",
        "intent": "ratings_and_earnings",
        "acceptable_tools": ["get_ratings", "get_earnings"],
        "adversarial": False,
        "no_data_expected": False,
        "notes": "Multi-tool compound: analyst sentiment + earnings actuals. "
                 "Requires two separate tool calls.",
    },
    {
        "id": "B09",
        "query": "What's happening broadly with electric vehicle stocks right now?",
        "intent": "news",
        "acceptable_tools": ["search_news"],
        "adversarial": False,
        "no_data_expected": False,
        "notes": "No specific ticker. Broad sector query. Agent should search news "
                 "without a ticker filter. May find Tesla-related results.",
    },
    {
        "id": "B10",
        "query": "What will Nvidia earn per share next quarter?",
        "intent": "adversarial_future",
        "acceptable_tools": ["search_news", "get_earnings", "get_guidance"],
        "adversarial": True,
        "no_data_expected": True,
        "notes": "Future EPS data does not exist in any fixture. "
                 "PASS only if answer explicitly says no data or uses guidance/news without "
                 "fabricating specific future EPS figures.",
    },
    {
        "id": "B11",
        "query": "Has Tesla changed its guidance upward or downward in recent months?",
        "intent": "guidance",
        "acceptable_tools": ["get_guidance", "search_news"],
        "adversarial": False,
        "no_data_expected": False,
        "notes": "Guidance intent but phrased as direction ('upward/downward'). "
                 "No TSLA guidance record exists — should return no-data gracefully.",
    },
    {
        "id": "B12",
        "query": "Show me recent analyst upgrades for JPM",
        "intent": "analyst_ratings",
        "acceptable_tools": ["get_ratings"],
        "adversarial": False,
        "no_data_expected": False,
        "notes": "'Upgrades' keyword is in rule table. Should route to get_ratings. "
                 "Tests case where intent is very clear.",
    },
    {
        "id": "B13",
        "query": "How exposed is Exxon to crude oil price volatility?",
        "intent": "news",
        "acceptable_tools": ["search_news"],
        "adversarial": False,
        "no_data_expected": False,
        "notes": "Narrative/risk question — no structured data tool can answer this. "
                 "Correct route: search_news. Tests default fallback behavior.",
    },
    {
        "id": "B14",
        "query": "Is TSLA outperforming revenue consensus?",
        "intent": "earnings",
        "acceptable_tools": ["get_earnings"],
        "adversarial": False,
        "no_data_expected": False,
        "notes": "'Outperform' keyword is in rule table under RATINGS, not earnings. "
                 "This is a known potential routing error — may incorrectly call get_ratings.",
    },
    {
        "id": "B15",
        "query": "What forward guidance has ZZZZ management provided to investors?",
        "intent": "guidance",
        "acceptable_tools": ["get_guidance", "search_news"],
        "adversarial": True,
        "no_data_expected": True,
        "notes": "Double adversarial: unknown ticker AND guidance-specific intent. "
                 "Should return no data without crashing.",
    },
]
# fmt: on
