"""
Coleman-Liau Index metric.

Original paper:
    Coleman, M., & Liau, T. L. (1975). A computer readability formula designed
    for machine scoring. Journal of Applied Psychology, 60(2), 283–284.
    https://doi.org/10.1037/h0076540

What it measures:
    A US grade-level readability estimate based on characters rather than
    syllables:

        grade = 0.0588 × L − 0.296 × S − 15.8

    where L = average number of letters per 100 words
          S = average number of sentences per 100 words.

    Because it relies on character counts, Coleman-Liau is more stable than
    syllable-based formulas (Flesch, FK, Gunning) on short texts such as
    single-sentence questions.

Score normalisation:
    score = max(0.0, 1.0 − grade / max_grade)
    Default max_grade = 18.0.
    Higher score → lower grade → more accessible text.

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


class ColemanLiauMetric(BaseReadabilityMetric):
    """
    Coleman-Liau Index for question.text.

    Args:
        max_grade  : Grade level that maps to score 0.0 (default 18.0).
        flag_below : Score threshold below which flagged=True.
    """

    name = "coleman_liau"
    description = (
        "Coleman-Liau Index: character-based grade estimate "
        "(normalised — higher score = more accessible)."
    )

    def __init__(self, max_grade: float = 18.0, flag_below: float = 0.2):
        if not _TEXTSTAT_AVAILABLE:
            raise ImportError(
                "textstat is required for ColemanLiauMetric.\n"
                "pip install textstat"
            )
        self.max_grade = max_grade
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        return float(_textstat.coleman_liau_index(text))

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - max(0.0, raw) / self.max_grade)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Coleman-Liau {raw:.1f} "
                f"(normalised {score:.4f}; max_grade={self.max_grade})."
            ),
            flagged=score < self.flag_below,
            metadata={"raw_grade": raw, "max_grade": self.max_grade},
        )
