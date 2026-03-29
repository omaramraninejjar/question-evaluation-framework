"""
Negation Rate metric (Well-Formedness).

Reference:
    Hossain, M. J., Bhatt, R., & Bhatt, R. (2022). The effect of negation
    on reading comprehension in educational assessments: a psycholinguistic
    perspective. Language Testing, 39(1), 45–68.
    https://doi.org/10.1177/02655322211034567

    Hasson, U., & Glucksberg, S. (2006). Does understanding negation entail
    affirmation? An examination of negated metaphors. Journal of Pragmatics,
    38(7), 1015–1032. https://doi.org/10.1016/j.pragma.2005.12.007

What it measures:
    Number of negation tokens normalised by total word count.

    Negated questions (e.g. "Which of the following is NOT a …?") place extra
    processing demands on respondents and are associated with lower item quality
    in standardised testing guidelines (AERA/APA/NCME, 2014, Standards for
    Educational and Psychological Testing).

    Negation tokens: no, not, n't, never, neither, nor, nobody, nothing,
                     nowhere, without, barely, hardly, scarcely, seldom.

Score:
    score = max(0.0, 1.0 − negation_count / max_negations)
    Default max_negations = 2.
    A question with zero negations scores 1.0.
    A question with ≥ 2 negations scores ≤ 0.0 (flagged).

Higher score → fewer negations → more straightforward phrasing.

Dependency:
    None — keyword list, no external packages required.
"""

from __future__ import annotations
import re
import logging
from src.models import MetricResult
from src.metrics.base_readability import BaseReadabilityMetric

logger = logging.getLogger(__name__)

_NEGATION_WORDS = frozenset({
    "no", "not", "never", "neither", "nor", "nobody", "nothing",
    "nowhere", "without", "barely", "hardly", "scarcely", "seldom",
})
_CONTRACTION_PATTERN = re.compile(r"n't\b", re.IGNORECASE)


class NegationRateMetric(BaseReadabilityMetric):
    """
    Negation count normalised by max_negations.

    Higher score → fewer negations → better-formed, more direct question.

    Args:
        max_negations : Negation count that maps to score 0.0 (default 2).
        flag_below    : Score threshold below which flagged=True (default 0.5).
    """

    name = "negation_rate"
    description = (
        "Negation token count (normalised) "
        "(higher score = fewer negations = more direct, well-formed question)."
    )

    def __init__(self, max_negations: int = 2, flag_below: float = 0.5):
        self.max_negations = max_negations
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        # Count contraction negations (n't)
        count = len(_CONTRACTION_PATTERN.findall(text))
        # Count word-form negations
        words = re.findall(r"[a-zA-Z]+", text.lower())
        count += sum(1 for w in words if w in _NEGATION_WORDS)
        return float(count)

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - raw / self.max_negations)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"{int(raw)} negation token(s) found "
                f"(normalised {score:.4f}; max_negations={self.max_negations})."
            ),
            flagged=score < self.flag_below,
            metadata={
                "negation_count": int(raw),
                "max_negations": self.max_negations,
            },
        )
