"""
Flesch Reading Ease metric.

Original paper:
    Flesch, R. (1948). A new readability yardstick.
    Journal of Applied Psychology, 32(3), 221–233.
    https://doi.org/10.1037/h0057532

What it measures:
    A formula-based estimate of how easy a text is to read, computed from
    average sentence length and average number of syllables per word:

        score = 206.835
                - 1.015  × (words / sentences)
                - 84.6   × (syllables / words)

    The scale runs from 0 to 100; higher values indicate easier text.
    Approximate interpretations:
        90–100  Very Easy   (5th grade)
        80–90   Easy        (6th grade)
        70–80   Fairly Easy (7th grade)
        60–70   Standard    (8th–9th grade)
        50–60   Fairly Difficult (10th–12th grade)
        30–50   Difficult   (college level)
         0–30   Very Confusing (college graduate)

Score normalisation:
    Raw value is divided by 100 to map into [0.0, 1.0].
    Scores above 100 (possible on very short, simple text) are clamped to 1.0.
    Negative scores (extremely complex text) are clamped to 0.0.

Use in this framework:
    Higher score → more readable question.
    flag_below default = 0.3 (Flesch < 30 → Very Confusing).

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


class FleschReadingEaseMetric(BaseReadabilityMetric):
    """
    Flesch Reading Ease score for question.text.

    Args:
        flag_below : Score threshold below which flagged=True.
            Default 0.3 (Flesch ease < 30 → Very Confusing).
    """

    name = "flesch_ease"
    description = "Flesch Reading Ease: formula-based readability (higher = easier, [0, 1])."

    def __init__(self, flag_below: float = 0.3):
        if not _TEXTSTAT_AVAILABLE:
            raise ImportError(
                "textstat is required for FleschReadingEaseMetric.\n"
                "pip install textstat"
            )
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        return float(_textstat.flesch_reading_ease(text))

    def _normalize(self, raw: float) -> float:
        return min(1.0, max(0.0, raw / 100.0))

    def _build_result(self, score: float, raw: float) -> MetricResult:
        labels = [
            (0.90, "Very Easy"),
            (0.80, "Easy"),
            (0.70, "Fairly Easy"),
            (0.60, "Standard"),
            (0.50, "Fairly Difficult"),
            (0.30, "Difficult"),
            (0.00, "Very Confusing"),
        ]
        label = next(l for t, l in labels if score >= t)
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=f"Flesch Reading Ease {raw:.1f} ({label}); normalised score {score:.4f}.",
            flagged=score < self.flag_below,
            metadata={"raw_value": raw, "label": label},
        )
