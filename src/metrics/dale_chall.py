"""
Dale-Chall Readability Formula metric.

Original paper:
    Dale, E., & Chall, J. S. (1948). A formula for predicting readability.
    Educational Research Bulletin, 27(1), 11–20, 28.

    Revised formula:
    Chall, J. S., & Dale, E. (1995). Readability revisited: The new
    Dale-Chall readability formula. Brookline Books.

What it measures:
    A grade-level readability formula that counts the proportion of
    "unfamiliar" words — words not on the Dale list of ~3,000 words known
    by most 4th graders:

        score = 0.1579 × (unfamiliar_words / words × 100)
                + 0.0496 × (words / sentences)
                [+ 3.6365 if unfamiliar_words/words > 5%]

    Unlike syllable-based formulas (Flesch, FK, Gunning), Dale-Chall uses
    an explicit vocabulary difficulty list, making it well-suited for
    educational text where specific terminology matters.

    Approximate grade interpretations:
        < 5.0   4th grade and below (very accessible)
        5.0–5.9 5th–6th grade
        6.0–6.9 7th–8th grade
        7.0–7.9 9th–10th grade
        8.0–8.9 11th–12th grade
        9.0–9.9 College level
       ≥ 10.0   College graduate

Score normalisation:
    score = max(0.0, 1.0 − raw / 10.0)
    Higher score → lower Dale-Chall value → more familiar vocabulary.

Dependency:
    textstat >= 0.7   (pip install textstat)
"""

from __future__ import annotations
import logging
from src.models import MetricResult
from src.metrics.base_readability import BaseReadabilityMetric

logger = logging.getLogger(__name__)

try:
    import textstat as _textstat
    _TEXTSTAT_AVAILABLE = True
except ImportError:
    _textstat = None  # type: ignore[assignment]
    _TEXTSTAT_AVAILABLE = False


class DaleChallMetric(BaseReadabilityMetric):
    """
    Dale-Chall Readability score for question.text.

    Args:
        flag_below : Score threshold below which flagged=True.
            Default 0.2 (Dale-Chall > 8.0 → 11th-grade+ vocabulary).
    """

    name = "dale_chall"
    description = (
        "Dale-Chall Readability: vocabulary-list-based grade estimate "
        "(normalised — higher score = more familiar vocabulary)."
    )

    def __init__(self, flag_below: float = 0.2):
        if not _TEXTSTAT_AVAILABLE:
            raise ImportError(
                "textstat is required for DaleChallMetric.\n"
                "pip install textstat"
            )
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        return float(_textstat.dale_chall_readability_score(text))

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - max(0.0, raw) / 10.0)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        labels = [
            (0.51, "4th grade or below"),
            (0.41, "5th–6th grade"),
            (0.31, "7th–8th grade"),
            (0.21, "9th–10th grade"),
            (0.11, "11th–12th grade"),
            (0.01, "College level"),
            (0.00, "College graduate"),
        ]
        label = next(l for t, l in labels if score >= t)
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=f"Dale-Chall {raw:.2f} ({label}); normalised score {score:.4f}.",
            flagged=score < self.flag_below,
            metadata={"raw_value": raw, "grade_label": label},
        )
