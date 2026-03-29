"""
Automated Readability Index (ARI) metric.

Original paper:
    Senter, R. J., & Smith, E. A. (1967). Automated readability index.
    Wright-Patterson Air Force Base, Ohio. Technical Report AMRL-TR-66-220.
    https://apps.dtic.mil/sti/citations/AD0667273

What it measures:
    A US grade-level readability estimate based on characters and words:

        grade = 4.71 × (chars / words) + 0.5 × (words / sentences) − 21.43

    ARI uses character counts rather than syllable counts, making it fast
    and well-suited for automated (machine-computed) scoring — hence the name.
    It correlates highly with other grade-level indices (FK, Coleman-Liau).

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


class ARIMetric(BaseReadabilityMetric):
    """
    Automated Readability Index for question.text.

    Args:
        max_grade  : Grade level that maps to score 0.0 (default 18.0).
        flag_below : Score threshold below which flagged=True.
    """

    name = "ari"
    description = (
        "Automated Readability Index: character-based grade estimate "
        "(normalised — higher score = more accessible)."
    )

    def __init__(self, max_grade: float = 18.0, flag_below: float = 0.2):
        if not _TEXTSTAT_AVAILABLE:
            raise ImportError(
                "textstat is required for ARIMetric.\n"
                "pip install textstat"
            )
        self.max_grade = max_grade
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        return float(_textstat.automated_readability_index(text))

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - max(0.0, raw) / self.max_grade)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"ARI {raw:.1f} "
                f"(normalised {score:.4f}; max_grade={self.max_grade})."
            ),
            flagged=score < self.flag_below,
            metadata={"raw_grade": raw, "max_grade": self.max_grade},
        )
