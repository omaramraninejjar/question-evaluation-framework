"""
Spache Readability Score metric (Readability).

Reference:
    Spache, G. (1953). A new readability formula for primary-grade reading
    materials. The Elementary School Journal, 53(7), 410–413.
    https://doi.org/10.1086/458513

What it measures:
    Grade-level readability designed for primary-grade (1–3) readers.
    The Spache formula uses sentence length and percentage of "difficult"
    words (words outside a pre-defined list of familiar words for young
    readers). It complements Dale-Chall at the low end of the reading
    spectrum.

    Grade 1–2 → very easy (young readers)
    Grade 3–4 → easy (elementary)
    Grade 5–6 → moderate
    Grade 7+  → difficult (beyond primary scope of the formula)

Score normalisation:
    score = max(0.0, 1.0 − spache_grade / max_grade)
    Default max_grade = 6.0.
    Higher score → lower grade level → easier to read.
    flag_below default 0.3 (grade ≥ 4.2 for max_grade=6).

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


class SpacheScoreMetric(BaseReadabilityMetric):
    """
    Spache readability grade level (primary-reader scale).

    Higher score → lower grade → easier to read for young audiences.

    Args:
        max_grade  : Grade level that maps to score 0.0 (default 6.0).
        flag_below : Score threshold below which flagged=True (default 0.3).
    """

    name = "spache_score"
    description = (
        "Spache readability grade level for primary readers "
        "(higher score = lower grade = easier for young readers)."
    )

    def __init__(self, max_grade: float = 6.0, flag_below: float = 0.3):
        if not _TEXTSTAT_AVAILABLE:
            raise ImportError(
                "textstat is required for SpacheScoreMetric.\n"
                "pip install textstat"
            )
        self.max_grade = max_grade
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        return float(_textstat.spache_readability(text))

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - raw / self.max_grade)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        if raw <= 2.0:
            label = "very easy (grade 1–2)"
        elif raw <= 4.0:
            label = "easy (grade 3–4)"
        elif raw <= 6.0:
            label = "moderate (grade 5–6)"
        else:
            label = "difficult (grade 7+)"
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Spache grade {raw:.2f} — {label} "
                f"(normalised {score:.4f}; max_grade={self.max_grade})."
            ),
            flagged=score < self.flag_below,
            metadata={"spache_grade": raw, "max_grade": self.max_grade, "label": label},
        )
