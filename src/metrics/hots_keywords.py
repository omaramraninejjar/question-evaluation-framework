"""
Higher-Order Thinking Skills (HOTS) Keyword Score metric (Cognitive Demand).

Reference:
    Bloom, B. S. (Ed.). (1956). Taxonomy of Educational Objectives:
    The Classification of Educational Goals. Handbook I: Cognitive Domain.
    David McKay.

    Thomas, A., & Thorne, G. (2010). How to Increase Higher Order Thinking.
    Center for Development and Learning.

    Brookhart, S. M. (2010). How to Assess Higher-Order Thinking Skills in
    Your Classroom. ASCD.

What it measures:
    Ratio of Higher-Order Thinking Skills (HOTS) keywords to the total
    content-word vocabulary of the question.

        HOTS keywords signal cognitive operations at Bloom's Levels 4–6:
        analyze, evaluate, critique, compare, synthesize, justify, design …

        LOTS (Lower-Order Thinking Skills) keywords signal Levels 1–2:
        define, list, name, identify, recall, describe, state …

        hots_ratio = HOTS_count / (HOTS_count + LOTS_count + ε)

    A question with zero signal words defaults to a balanced score of 0.5
    (insufficient information to classify).

Score:
    score = hots_ratio ∈ [0.0, 1.0]
    Higher score → more HOTS keywords → higher cognitive demand.
    Returns 0.5 when neither HOTS nor LOTS keywords are found.
    flag_below default 0.3 (question is dominated by recall keywords).

Dependency:
    None — keyword list, no external packages required.
"""

from __future__ import annotations
import re
import logging
from src.models import MetricResult
from src.metrics.base_readability import BaseReadabilityMetric

logger = logging.getLogger(__name__)

_HOTS_KEYWORDS = frozenset({
    # Bloom Level 4 — Analyze
    "analyze", "analyse", "differentiate", "distinguish", "compare", "contrast",
    "categorize", "categorise", "classify", "examine", "investigate", "inspect",
    "deconstruct", "breakdown", "separate", "discriminate", "experiment",
    # Bloom Level 5 — Evaluate
    "evaluate", "assess", "appraise", "argue", "critique", "criticize", "criticise",
    "judge", "justify", "support", "defend", "challenge", "rank", "prioritize",
    "prioritise", "weigh", "recommend", "conclude", "verify",
    # Bloom Level 6 — Create
    "design", "create", "construct", "formulate", "generate", "produce",
    "hypothesize", "hypothesise", "invent", "compose", "develop", "synthesize",
    "synthesise", "integrate", "devise", "propose", "plan", "organize", "organise",
})

_LOTS_KEYWORDS = frozenset({
    # Bloom Level 1 — Remember
    "define", "list", "name", "identify", "recall", "recognize", "recognise",
    "state", "repeat", "match", "label", "memorize", "memorise", "locate",
    "find", "recite", "retrieve",
    # Bloom Level 2 — Understand
    "describe", "explain", "summarize", "summarise", "paraphrase", "indicate",
    "report", "select", "translate", "outline", "review",
})


class HOTSKeywordsMetric(BaseReadabilityMetric):
    """
    HOTS keyword ratio: proportion of thinking-signal keywords that are
    higher-order (Bloom L4–L6) rather than lower-order (Bloom L1–L2).

    Higher score → more HOTS-aligned vocabulary → greater cognitive demand.
    Returns 0.5 when no signal keywords found.

    Args:
        flag_below : Score threshold below which flagged=True (default 0.3).
    """

    name = "hots_keywords"
    description = (
        "HOTS keyword ratio (higher-order / total thinking-signal words) "
        "(higher score = more higher-order thinking vocabulary)."
    )

    def __init__(self, flag_below: float = 0.3):
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        words = re.findall(r"[a-zA-Z]+", text.lower())
        n_hots = sum(1 for w in words if w in _HOTS_KEYWORDS)
        n_lots = sum(1 for w in words if w in _LOTS_KEYWORDS)
        total = n_hots + n_lots
        if total == 0:
            return -1.0  # sentinel for "no signal" — handled in compute()
        return n_hots / total

    def _normalize(self, raw: float) -> float:
        return raw if raw >= 0 else 0.5

    def compute(self, question, context):
        if not question.text.strip():
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                rationale="Question text is empty.",
                flagged=True,
                metadata={"reason": "empty_text"},
            )
        words = re.findall(r"[a-zA-Z]+", question.text.lower())
        n_hots = sum(1 for w in words if w in _HOTS_KEYWORDS)
        n_lots = sum(1 for w in words if w in _LOTS_KEYWORDS)
        total = n_hots + n_lots

        if total == 0:
            return MetricResult(
                metric_name=self.name,
                score=0.5,
                rationale="No HOTS or LOTS signal keywords found — score is neutral (0.5).",
                flagged=False,
                metadata={"n_hots": 0, "n_lots": 0, "hots_ratio": 0.5, "no_signal": True},
            )

        ratio = n_hots / total
        return MetricResult(
            metric_name=self.name,
            score=ratio,
            rationale=(
                f"{n_hots} HOTS keyword(s) and {n_lots} LOTS keyword(s) detected "
                f"(HOTS ratio={ratio:.4f})."
            ),
            flagged=ratio < self.flag_below,
            metadata={"n_hots": n_hots, "n_lots": n_lots, "hots_ratio": ratio},
        )

    def _build_result(self, score: float, raw: float) -> MetricResult:  # pragma: no cover
        raise NotImplementedError
