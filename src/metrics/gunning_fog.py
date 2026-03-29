"""
Gunning Fog Index metric.

Original reference:
    Gunning, R. (1952). The technique of clear writing.
    McGraw-Hill. New York.

What it measures:
    A US grade-level readability estimate based on average sentence length
    and the percentage of "complex" words (≥ 3 syllables):

        grade = 0.4 × ((words / sentences) + 100 × (complex_words / words))

    "Fog" refers to how difficult the text is to understand. High complexity
    words drive the score up quickly, making it sensitive to domain-specific
    terminology in educational questions.

    Approximate grade interpretations:
        17+   College graduate level
        13–16 College level
        9–12  High school level
        ≤ 8   Accessible to most adults

Score normalisation:
    score = max(0.0, 1.0 − grade / max_grade)
    Default max_grade = 20.0 (covers the typical ceiling of the index).
    Higher score → lower fog index → clearer question text.

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


class GunningFogMetric(BaseReadabilityMetric):
    """
    Gunning Fog Index for question.text.

    Args:
        max_grade  : Grade level that maps to score 0.0 (default 20.0).
        flag_below : Score threshold below which flagged=True.
    """

    name = "gunning_fog"
    description = (
        "Gunning Fog Index: complex-word-based grade estimate "
        "(normalised — higher score = clearer text)."
    )

    def __init__(self, max_grade: float = 20.0, flag_below: float = 0.2):
        if not _TEXTSTAT_AVAILABLE:
            raise ImportError(
                "textstat is required for GunningFogMetric.\n"
                "pip install textstat"
            )
        self.max_grade = max_grade
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        return float(_textstat.gunning_fog(text))

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - max(0.0, raw) / self.max_grade)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Gunning Fog {raw:.1f} "
                f"(normalised {score:.4f}; max_grade={self.max_grade})."
            ),
            flagged=score < self.flag_below,
            metadata={"raw_grade": raw, "max_grade": self.max_grade},
        )
