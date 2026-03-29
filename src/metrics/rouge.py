"""
ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation — Longest
Common Subsequence) metric.

Original paper:
    Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of
    Summaries. ACL Workshop on Text Summarization Branches Out.
    https://aclanthology.org/W04-1013/

What it measures:
    Longest Common Subsequence (LCS) overlap between a candidate text and
    one or more reference texts. Unlike BLEU, ROUGE-L captures in-order
    word matches that need not be contiguous, making it more robust to
    paraphrasing. F-measure balances precision and recall; recall-weighted
    variants are common when references are known to be concise.
    Scores are in [0.0, 1.0]; higher = greater overlap.

Use in this framework:
    Curriculum Alignment — references = context.learning_objectives
    Concept Coverage     — references = [context.course_content]

Score type:
    "fmeasure"  (default) — harmonic mean of LCS precision and recall
    "precision"           — fraction of candidate tokens in LCS
    "recall"              — fraction of reference tokens in LCS

Aggregation over multiple references:
    Max ROUGE-L across all references (standard practice).

Dependency:
    rouge-score >= 0.1.2   (pip install rouge-score)
"""

from __future__ import annotations
import logging
from src.models import Question, EvaluationContext, MetricResult
from src.metrics.base import BaseReferenceMetric

logger = logging.getLogger(__name__)

try:
    from rouge_score import rouge_scorer as _rouge_scorer_mod
    _ROUGE_AVAILABLE = True
except ImportError:
    _rouge_scorer_mod = None    # type: ignore[assignment]
    _ROUGE_AVAILABLE = False

_VALID_SCORE_TYPES = {"fmeasure", "precision", "recall"}


class ROUGELMetric(BaseReferenceMetric):
    """
    ROUGE-L F-measure (or precision / recall) between question.text
    and context references.

    Args:
        reference_source : Which context field to use as references.
            "learning_objectives" → context.learning_objectives  (list[str])
            "course_content"      → context.course_content       (str)
        score_type       : Which component of the ROUGE-L score to return.
            "fmeasure"  (default), "precision", or "recall"
        use_stemmer      : Apply Porter stemmer before matching.
            Reduces sensitivity to morphological variants (default: True).
        flag_below       : Score threshold below which flagged=True.
    """

    name = "rouge_l"
    description = (
        "LCS-based overlap between question text and context references "
        "(rouge-score RougeL with optional stemming)."
    )

    def __init__(
        self,
        reference_source: str = "learning_objectives",
        score_type: str = "fmeasure",
        use_stemmer: bool = True,
        flag_below: float = 0.3,
    ):
        if not _ROUGE_AVAILABLE:
            raise ImportError(
                "rouge-score is required for ROUGELMetric.\n"
                "pip install rouge-score"
            )
        if score_type not in _VALID_SCORE_TYPES:
            raise ValueError(
                f"score_type must be one of {_VALID_SCORE_TYPES}, got {score_type!r}"
            )

        self.reference_source = reference_source
        self.score_type = score_type
        self.use_stemmer = use_stemmer
        self.flag_below = flag_below
        self._scorer = _rouge_scorer_mod.RougeScorer(
            ["rougeL"], use_stemmer=use_stemmer
        )

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

        # Score against each reference; keep the best (max) result.
        best_score = 0.0
        for ref in references:
            rouge_scores = self._scorer.score(ref, question.text)
            candidate = getattr(rouge_scores["rougeL"], self.score_type)
            if candidate > best_score:
                best_score = candidate

        quality = "strong" if best_score >= 0.6 else "moderate" if best_score >= 0.3 else "weak"

        return MetricResult(
            metric_name=self.name,
            score=best_score,
            rationale=(
                f"ROUGE-L {self.score_type} {best_score:.4f} ({quality} overlap) "
                f"vs {len(references)} reference(s) from context.{self.reference_source}."
            ),
            flagged=best_score < self.flag_below,
            metadata={
                "reference_source": self.reference_source,
                "score_type": self.score_type,
                "n_references": len(references),
                "use_stemmer": self.use_stemmer,
            },
        )

