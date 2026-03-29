"""
BLEU (Bilingual Evaluation Understudy) metric.

Original paper:
    Papineni et al. (2002). BLEU: a Method for Automatic Evaluation of
    Machine Translation. ACL 2002.
    https://aclanthology.org/P02-1040/

What it measures:
    N-gram overlap between a candidate text and one or more reference texts,
    with a brevity penalty that discourages overly short candidates.
    Scores are in [0.0, 1.0]; higher = greater overlap.

Use in this framework:
    Curriculum Alignment — references = context.learning_objectives
    Concept Coverage     — references = [context.course_content]

Dependency:
    nltk >= 3.8
    On first run: import nltk; nltk.download('punkt_tab')
"""

from __future__ import annotations
import logging
from src.models import Question, EvaluationContext, MetricResult
from src.metrics.base import BaseReferenceMetric

logger = logging.getLogger(__name__)

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


def _tokenize(text: str) -> list[str]:
    if _NLTK_AVAILABLE:
        return word_tokenize(text.lower())
    return text.lower().split()


class BLEUMetric(BaseReferenceMetric):
    """
    Sentence-level BLEU between question.text and context references.

    Args:
        reference_source : Which context field to use as references.
            "learning_objectives" → context.learning_objectives  (list[str])
            "course_content"      → context.course_content       (str)
        weights          : N-gram weights, must sum to 1.0.
            (0.25, 0.25, 0.25, 0.25) = BLEU-4 (default)
            (1.0,)                   = BLEU-1, better for short texts
        flag_below       : Score threshold below which flagged=True.
    """

    name = "bleu"
    description = (
        "N-gram overlap between question text and context references "
        "(NLTK sentence_bleu with smoothing)."
    )

    def __init__(
        self,
        reference_source: str = "learning_objectives",
        weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
        flag_below: float = 0.3,
    ):
        if not _NLTK_AVAILABLE:
            raise ImportError(
                "nltk is required for BLEUMetric.\n"
                "pip install nltk\n"
                "python -c \"import nltk; nltk.download('punkt_tab')\""
            )
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {sum(weights):.4f}")

        self.reference_source = reference_source
        self.weights = weights
        self.flag_below = flag_below
        # Method1 smoothing prevents score collapsing to 0 on short texts
        self._smoother = SmoothingFunction().method1

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

        candidate_tokens = _tokenize(question.text)

        if not candidate_tokens:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                rationale="Question text is empty.",
                flagged=True,
                metadata={"reason": "empty_candidate"},
            )

        score = float(sentence_bleu(
            references=[_tokenize(r) for r in references],
            hypothesis=candidate_tokens,
            weights=self.weights,
            smoothing_function=self._smoother,
        ))

        quality = "strong" if score >= 0.6 else "moderate" if score >= 0.3 else "weak"

        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"BLEU {score:.4f} ({quality} overlap) vs "
                f"{len(references)} reference(s) from context.{self.reference_source}."
            ),
            flagged=score < self.flag_below,
            metadata={
                "reference_source": self.reference_source,
                "n_references": len(references),
                "candidate_tokens": len(candidate_tokens),
                "weights": list(self.weights),
            },
        )

