"""
Linsear Write Formula metric.

Original reference:
    Linsear Write formula (developed by the United States Air Force,
    exact origin undated; widely attributed to ~1971).
    Described in: DuBay, W. H. (2004). The Principles of Readability.
    Costa Mesa, CA: Impact Information, pp. 35–36.
    https://files.eric.ed.gov/fulltext/ED490073.pdf

What it measures:
    A US grade-level estimate that distinguishes easy words (≤ 2 syllables)
    from hard words (> 2 syllables):

        raw_score = (easy_words × 1 + hard_words × 3) / sentence_count
        grade     = raw_score / 2      if raw_score > 20
                  = (raw_score − 1) / 2  otherwise

    It was developed specifically for US Air Force technical manuals and
    performs well on instructional texts of comparable register.

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


class LinsearWriteMetric(BaseReadabilityMetric):
    """
    Linsear Write Formula grade-level estimate for question.text.

    Args:
        max_grade  : Grade level that maps to score 0.0 (default 18.0).
        flag_below : Score threshold below which flagged=True.
    """

    name = "linsear_write"
    description = (
        "Linsear Write Formula: syllable-category grade estimate "
        "(normalised — higher score = more accessible)."
    )

    def __init__(self, max_grade: float = 18.0, flag_below: float = 0.2):
        if not _TEXTSTAT_AVAILABLE:
            raise ImportError(
                "textstat is required for LinsearWriteMetric.\n"
                "pip install textstat"
            )
        self.max_grade = max_grade
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        return float(_textstat.linsear_write_formula(text))

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - max(0.0, raw) / self.max_grade)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Linsear Write {raw:.1f} "
                f"(normalised {score:.4f}; max_grade={self.max_grade})."
            ),
            flagged=score < self.flag_below,
            metadata={"raw_grade": raw, "max_grade": self.max_grade},
        )
