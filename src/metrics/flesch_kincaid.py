"""
Flesch-Kincaid Grade Level metric.

Original paper:
    Kincaid, J. P., Fishburne, R. P., Rogers, R. L., & Chissom, B. S. (1975).
    Derivation of new readability formulas (Automated Readability Index, Fog
    Count, and Flesch Reading Ease Formula) for Navy enlisted personnel.
    Research Branch Report 8-75, Naval Technical Training Command.

What it measures:
    A US grade-level estimate derived from average sentence length and
    average syllables per word:

        grade = 0.39 × (words / sentences)
                + 11.8 × (syllables / words)
                − 15.59

    A grade of 8.0 means an 8th-grader can read the text.  Typical questions
    for general audiences fall in the range 6–10.

Score normalisation:
    grade is mapped to [0.0, 1.0] via:
        score = max(0.0, 1.0 − grade / max_grade)
    Default max_grade = 18.0 (approximately college-graduate level).
    Higher score → lower grade level → more accessible text.

Use in this framework:
    flag_above is used to catch questions that are too advanced for the
    target population (grade > max_acceptable_grade).

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


class FleschKincaidMetric(BaseReadabilityMetric):
    """
    Flesch-Kincaid Grade Level for question.text.

    Args:
        max_grade  : Grade level that maps to score 0.0 (default 18.0).
        flag_below : Score threshold below which flagged=True
                     (i.e., grade above max_grade * (1 − flag_below)).
    """

    name = "flesch_kincaid"
    description = (
        "Flesch-Kincaid Grade Level: US grade estimate "
        "(normalised — higher score = lower grade = more accessible)."
    )

    def __init__(self, max_grade: float = 18.0, flag_below: float = 0.2):
        if not _TEXTSTAT_AVAILABLE:
            raise ImportError(
                "textstat is required for FleschKincaidMetric.\n"
                "pip install textstat"
            )
        self.max_grade = max_grade
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        return float(_textstat.flesch_kincaid_grade(text))

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - max(0.0, raw) / self.max_grade)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Flesch-Kincaid Grade {raw:.1f} "
                f"(normalised {score:.4f}; max_grade={self.max_grade})."
            ),
            flagged=score < self.flag_below,
            metadata={"raw_grade": raw, "max_grade": self.max_grade},
        )
