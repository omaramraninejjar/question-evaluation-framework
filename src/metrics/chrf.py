"""
chrF (Character n-gram F-score) metric.

Original paper:
    Popović (2015). chrF: character n-gram F-score for automatic MT evaluation.
    WMT 2015 (Workshop on Statistical Machine Translation).
    https://aclanthology.org/W15-3049/

    chrF++ extension (adds word unigrams):
    Popović (2017). chrF++: words helping character n-grams.
    WMT 2017.
    https://aclanthology.org/W17-4770/

What it measures:
    F-score over character n-gram overlap between a candidate text and one or
    more reference texts. Unlike BLEU (which operates on word tokens), chrF
    works at the character level, making it more sensitive to morphological
    variants and domain-specific terminology — both common in educational text.
    Scores are in [0.0, 1.0]; higher = greater character-level overlap.

    Setting word_order=2 activates chrF++ mode, which additionally weights
    word bigrams alongside character n-grams.

Use in this framework:
    Curriculum Alignment — references = context.learning_objectives
    Concept Coverage     — references = [context.course_content]

Aggregation over multiple references:
    The candidate is scored against each reference independently;
    the maximum score across references is returned.

Dependency:
    sacrebleu >= 2.0   (pip install sacrebleu)
"""

from __future__ import annotations
import logging
from src.models import Question, EvaluationContext, MetricResult
from src.metrics.base import BaseReferenceMetric

logger = logging.getLogger(__name__)

try:
    from sacrebleu.metrics import CHRF as _CHRF
    _CHRF_AVAILABLE = True
except ImportError:
    _CHRF = None  # type: ignore[assignment,misc]
    _CHRF_AVAILABLE = False


class chrFMetric(BaseReferenceMetric):
    """
    chrF score between question.text and context references.

    Args:
        reference_source : Which context field to use as references.
            "learning_objectives" → context.learning_objectives  (list[str])
            "course_content"      → context.course_content       (str)
        char_order       : Character n-gram order (default 6, as in the paper).
        word_order       : Word n-gram order. 0 = chrF (default), 2 = chrF++.
        beta             : Weighting of recall vs precision (default 2.0,
                           as in the paper — recall-weighted).
        flag_below       : Score threshold below which flagged=True.
    """

    name = "chrf"
    description = (
        "Character n-gram F-score between question text and context references "
        "(chrF / chrF++, max over references)."
    )

    def __init__(
        self,
        reference_source: str = "learning_objectives",
        char_order: int = 6,
        word_order: int = 0,
        beta: float = 2.0,
        flag_below: float = 0.3,
    ):
        if not _CHRF_AVAILABLE:
            raise ImportError(
                "sacrebleu is required for chrFMetric.\n"
                "pip install sacrebleu"
            )
        self.reference_source = reference_source
        self.char_order = char_order
        self.word_order = word_order
        self.beta = beta
        self.flag_below = flag_below
        self._scorer = _CHRF(char_order=char_order, word_order=word_order, beta=beta)

    # ------------------------------------------------------------------

    def compute(self, question: Question, context: EvaluationContext) -> MetricResult:
        references = self._collect_references(context)

        if not references:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                rationale=f"No references found in context.{self.reference_source}.",
                flagged=True,
                metadata={"reason": "no_references", "reference_source": self.reference_source},
            )

        if not question.text.strip():
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                rationale="Question text is empty.",
                flagged=True,
                metadata={"reason": "empty_candidate"},
            )

        # Score against each reference independently; keep the best.
        # sacrebleu sentence_score returns a CHRFScore; .score is in [0, 100].
        best_raw = max(
            self._scorer.sentence_score(question.text, [ref]).score
            for ref in references
        )
        best_score = best_raw / 100.0  # normalise to [0, 1]

        mode = "chrF++" if self.word_order > 0 else "chrF"
        quality = "strong" if best_score >= 0.6 else "moderate" if best_score >= 0.3 else "weak"

        return MetricResult(
            metric_name=self.name,
            score=best_score,
            rationale=(
                f"{mode} {best_score:.4f} ({quality} character-level overlap) "
                f"vs {len(references)} reference(s) from context.{self.reference_source}."
            ),
            flagged=best_score < self.flag_below,
            metadata={
                "reference_source": self.reference_source,
                "n_references": len(references),
                "char_order": self.char_order,
                "word_order": self.word_order,
                "mode": mode,
            },
        )
