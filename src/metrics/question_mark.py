"""
Question Mark metric (Well-Formedness).

Reference:
    American Educational Research Association, American Psychological Association,
    & National Council on Measurement in Education. (2014).
    Standards for Educational and Psychological Testing. AERA.
    Section 4.7: Item writing guidelines — punctuation and formatting.

What it measures:
    Binary indicator: does the question end with a question mark '?'.

    A well-formed interrogative sentence should close with '?'. Missing
    question marks may indicate a stem formatted as a declarative sentence,
    an incomplete sentence, or a copy-paste artefact. This is one of the
    simplest structural well-formedness checks.

Score:
    1.0 if text (stripped of whitespace) ends with '?'
    0.0 otherwise

Higher score → question has proper interrogative punctuation.

Dependency:
    None — pure Python, no external packages required.
"""

from __future__ import annotations
from src.models import Question, EvaluationContext, MetricResult
from src.metrics.base_readability import BaseReadabilityMetric


class QuestionMarkMetric(BaseReadabilityMetric):
    """
    Checks that the question text ends with a question mark.

    Score 1.0 → ends with '?'; 0.0 → does not.

    Args:
        flag_below : Score threshold below which flagged=True (default 0.5,
                     so absence of '?' always triggers a flag).
    """

    name = "question_mark"
    description = (
        "Binary: 1.0 if question ends with '?', else 0.0 "
        "(structural punctuation well-formedness)."
    )

    def __init__(self, flag_below: float = 0.5):
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        return 1.0 if text.strip().endswith("?") else 0.0

    def _normalize(self, raw: float) -> float:
        return raw  # already binary 0/1

    def _build_result(self, score: float, raw: float) -> MetricResult:
        has_mark = bool(raw)
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                "Question ends with '?'."
                if has_mark
                else "Question does not end with '?' — check punctuation."
            ),
            flagged=score < self.flag_below,
            metadata={"ends_with_question_mark": has_mark},
        )
