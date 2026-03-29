"""
METEOR (Metric for Evaluation of Translation with Explicit ORdering) metric.

Original paper:
    Banerjee & Lavie (2005). METEOR: An Automatic Metric for MT Evaluation
    with Improved Correlation with Human Judgments. ACL Workshop on
    Intrinsic and Extrinsic Evaluation Measures for MT and/or Summarization.
    https://aclanthology.org/W05-0909/

What it measures:
    Alignment-based overlap between a candidate text and one or more
    reference texts. Unlike BLEU (precision-only) or ROUGE (recall-only),
    METEOR computes an F-mean that weights recall more heavily (via alpha),
    then applies a fragmentation penalty that punishes non-contiguous matches.
    This makes it more robust to paraphrase and better correlated with human
    judgement on short texts.
    Scores are in [0.0, 1.0]; higher = better alignment.

Key parameters (passed through to NLTK):
    alpha  (default 0.9) — controls precision/recall balance in the F-mean.
                           Higher values weight recall more heavily.
    beta   (default 3.0) — exponent in the fragmentation penalty.
                           Higher values penalise non-contiguous matches more.
    gamma  (default 0.5) — weight of the fragmentation penalty.

Use in this framework:
    Curriculum Alignment — references = context.learning_objectives
    Concept Coverage     — references = [context.course_content]

Aggregation over multiple references:
    NLTK's meteor_score handles multiple references natively and returns
    the best-matching alignment score.

Dependency:
    nltk >= 3.8
    On first run:
        import nltk
        nltk.download('punkt_tab')
        nltk.download('wordnet')
"""

from __future__ import annotations
import logging
from src.models import Question, EvaluationContext, MetricResult
from src.metrics.base import BaseReferenceMetric

logger = logging.getLogger(__name__)

try:
    from nltk.translate.meteor_score import meteor_score as _meteor_score
    from nltk.tokenize import word_tokenize
    _NLTK_AVAILABLE = True
except ImportError:
    _meteor_score = None        # type: ignore[assignment]
    word_tokenize = None        # type: ignore[assignment]
    _NLTK_AVAILABLE = False


def _tokenize(text: str) -> list[str]:
    if _NLTK_AVAILABLE:
        return word_tokenize(text.lower())
    return text.lower().split()


class METEORMetric(BaseReferenceMetric):
    """
    METEOR score between question.text and context references.

    Args:
        reference_source : Which context field to use as references.
            "learning_objectives" → context.learning_objectives  (list[str])
            "course_content"      → context.course_content       (str)
        alpha            : Precision/recall balance in the F-mean.
                           Must be in (0.0, 1.0]. Default 0.9 (recall-heavy).
        beta             : Fragmentation penalty exponent. Default 3.0.
        gamma            : Fragmentation penalty weight. Default 0.5.
        flag_below       : Score threshold below which flagged=True.
    """

    name = "meteor"
    description = (
        "Alignment-based overlap with fragmentation penalty between question "
        "text and context references (NLTK meteor_score)."
    )

    def __init__(
        self,
        reference_source: str = "learning_objectives",
        alpha: float = 0.9,
        beta: float = 3.0,
        gamma: float = 0.5,
        flag_below: float = 0.3,
    ):
        if not _NLTK_AVAILABLE:
            raise ImportError(
                "nltk is required for METEORMetric.\n"
                "pip install nltk\n"
                "python -c \"import nltk; nltk.download('punkt_tab'); nltk.download('wordnet')\""
            )
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0.0, 1.0], got {alpha}")
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma must be in [0.0, 1.0], got {gamma}")

        self.reference_source = reference_source
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.flag_below = flag_below

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

        try:
            score = float(_meteor_score(
                references=[_tokenize(r) for r in references],
                hypothesis=_tokenize(question.text),
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
            ))
        except LookupError:
            raise LookupError(
                "METEOR requires WordNet data.\n"
                "python -c \"import nltk; nltk.download('wordnet')\""
            )

        quality = "strong" if score >= 0.6 else "moderate" if score >= 0.3 else "weak"

        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"METEOR {score:.4f} ({quality} alignment) vs "
                f"{len(references)} reference(s) from context.{self.reference_source}."
            ),
            flagged=score < self.flag_below,
            metadata={
                "reference_source": self.reference_source,
                "n_references": len(references),
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
            },
        )
