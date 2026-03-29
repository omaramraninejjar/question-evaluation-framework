"""
SMOG (Simple Measure of Gobbledygook) metric.

Original paper:
    McLaughlin, G. H. (1969). SMOG grading — a new readability formula.
    Journal of Reading, 12(8), 639–646.
    https://doi.org/10.1598/JAAL.12.8.2

What it measures:
    A US grade-level estimate based on polysyllabic word counts:

        grade = 3 + sqrt(polysyllable_count × (30 / sentence_count))

    SMOG is designed for texts of at least 30 sentences; on shorter passages
    the formula over-estimates grade level because the 30-sentence normalisation
    factor amplifies sampling noise. textstat returns 0 when sentence_count < 3.

    ⚠ Short-text caveat: single-sentence questions will typically contain fewer
    than 3 sentences, so textstat may return 0.0 or a highly variable estimate.
    Treat SMOG scores on short passages as indicative only.

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


class SMOGMetric(BaseReadabilityMetric):
    """
    SMOG grade-level estimate for question.text.

    Note: SMOG is most reliable for texts with ≥ 30 sentences. For short
    single-sentence questions the score should be treated as indicative.

    Args:
        max_grade  : Grade level that maps to score 0.0 (default 18.0).
        flag_below : Score threshold below which flagged=True.
    """

    name = "smog"
    description = (
        "SMOG Index: polysyllable-based grade estimate "
        "(normalised — higher score = more accessible; "
        "unreliable for texts with fewer than 30 sentences)."
    )

    def __init__(self, max_grade: float = 18.0, flag_below: float = 0.2):
        if not _TEXTSTAT_AVAILABLE:
            raise ImportError(
                "textstat is required for SMOGMetric.\n"
                "pip install textstat"
            )
        self.max_grade = max_grade
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        return float(_textstat.smog_index(text))

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - max(0.0, raw) / self.max_grade)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"SMOG {raw:.1f} "
                f"(normalised {score:.4f}; max_grade={self.max_grade})."
            ),
            flagged=score < self.flag_below,
            metadata={"raw_grade": raw, "max_grade": self.max_grade},
        )
