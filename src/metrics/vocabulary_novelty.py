"""
Vocabulary Novelty metric (Diversity).

Reference:
    Deane, P., & Sheehan, K. (2003). Automatic item generation via frame semantics:
    Natural language generation of mathematical word problems.
    ETS Research Report RR-03-16.

    Gierl, M. J., & Haladyna, T. M. (Eds.). (2013).
    Automatic Item Generation: Theory and Practice.
    Routledge. Chapter 4: Lexical diversity in generated items.

What it measures:
    Fraction of question words that do NOT appear in the course content.

        novelty = |{w ∈ question_words} \ {w ∈ content_words}| / |question_words|

    A question that only reuses words from the course content may be a direct
    surface-level paraphrase with low lexical diversity. A question with novel
    vocabulary requires the respondent to apply or transfer knowledge rather
    than pattern-match to the source text.

    High novelty → richer vocabulary diversification from the source material.
    Low novelty  → question mostly repeats course-content vocabulary.

    Requires course_content in the EvaluationContext. Returns score=0.0 and
    flagged=True when no course_content is provided.

Score:
    score = novelty ratio ∈ [0.0, 1.0]
    Higher score → more vocabulary beyond the course content → more diverse.
    flag_below default 0.2 (very low novelty — nearly all words come from content).

Dependency:
    None — pure Python, no external packages required.
"""

from __future__ import annotations
import re
import logging
from src.models import Question, EvaluationContext, MetricResult
from src.metrics.base import BaseMetric

logger = logging.getLogger(__name__)


def _word_set(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[a-zA-Z]+", text)}


class VocabularyNoveltyMetric(BaseMetric):
    """
    Fraction of question words absent from the course content.

    Higher score → more vocabulary beyond the source material → greater diversity.
    Returns score=0.0 / flagged=True when course_content is missing.

    Args:
        flag_below : Score threshold below which flagged=True (default 0.2).
    """

    name = "vocabulary_novelty"
    description = (
        "Fraction of question words not found in course content "
        "(higher score = more novel vocabulary = greater lexical diversity)."
    )

    def __init__(self, flag_below: float = 0.2):
        self.flag_below = flag_below

    def compute(self, question: Question, context: EvaluationContext) -> MetricResult:
        if not question.text.strip():
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                rationale="Question text is empty.",
                flagged=True,
                metadata={"reason": "empty_text"},
            )

        if not context.course_content:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                rationale="No course_content provided — cannot compute vocabulary novelty.",
                flagged=True,
                metadata={"reason": "no_course_content"},
            )

        question_words = _word_set(question.text)
        content_words  = _word_set(context.course_content)

        if not question_words:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                rationale="No alphabetic words found in question.",
                flagged=True,
                metadata={"reason": "no_words"},
            )

        novel = question_words - content_words
        ratio = len(novel) / len(question_words)

        return MetricResult(
            metric_name=self.name,
            score=ratio,
            rationale=(
                f"{len(novel)}/{len(question_words)} question words absent from "
                f"course content (novelty={ratio:.4f})."
            ),
            flagged=ratio < self.flag_below,
            metadata={
                "novel_word_count": len(novel),
                "question_word_count": len(question_words),
                "novelty_ratio": ratio,
            },
        )
