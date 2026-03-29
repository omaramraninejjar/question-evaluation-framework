"""
BLEURT metric.

Original paper:
    Sellam et al. (2020). BLEURT: Learning Robust Metrics for Text Generation.
    ACL 2020.
    https://arxiv.org/abs/2004.04696

What it measures:
    Learned semantic similarity between a candidate text and one or more
    reference texts, using a BERT-based regression model fine-tuned on
    human quality judgments. BLEURT captures nuanced semantic quality beyond
    surface-form overlap or contextual embeddings alone.

Score range:
    Raw BLEURT-20 scores are approximately in [-2, 1] (higher = better).
    This implementation clips scores to [0.0, 1.0] for consistency with
    other metrics in this framework; negative raw scores (clearly poor
    candidates) become 0.0. Scores are NOT linearly calibrated across
    checkpoints, so treat absolute values as checkpoint-specific.

Aggregation over multiple references:
    The candidate is scored against each reference independently; the
    maximum score across references is returned.

Checkpoint:
    model_name defaults to "BLEURT-20" (~1.7 GB, downloaded on first use
    and cached in ~/.cache/huggingface). Smaller options: "bleurt-tiny-128"
    (~39 MB, lower accuracy — useful for quick smoke tests).

Dependency:
    evaluate >= 0.4       (pip install evaluate)
    tensorflow >= 2.x     (BLEURT checkpoint loader requires TF)
"""

from __future__ import annotations
import logging
from src.models import Question, EvaluationContext, MetricResult
from src.metrics.base import BaseReferenceMetric

logger = logging.getLogger(__name__)

try:
    import evaluate as _evaluate
    import bleurt as _bleurt_pkg  # noqa: F401 — required by evaluate's BLEURT loader
    _BLEURT_AVAILABLE = True
except ImportError:
    _evaluate = None      # type: ignore[assignment]
    _bleurt_pkg = None    # type: ignore[assignment]
    _BLEURT_AVAILABLE = False


class BLEURTMetric(BaseReferenceMetric):
    """
    BLEURT score between question.text and context references.

    Args:
        reference_source : Which context field to use as references.
            "learning_objectives" → context.learning_objectives  (list[str])
            "course_content"      → context.course_content       (str)
        model_name       : BLEURT checkpoint identifier.
            Default "BLEURT-20" (~1.7 GB first-run download).
            Use "bleurt-tiny-128" (~39 MB) for fast smoke tests.
        flag_below       : Score threshold below which flagged=True.
            Default 0.5 — tune once you know your typical score range.
    """

    name = "bleurt"
    description = (
        "Learned semantic similarity via BERT fine-tuned on human judgments "
        "(BLEURT, max over references, clipped to [0, 1])."
    )

    def __init__(
        self,
        reference_source: str = "learning_objectives",
        model_name: str = "BLEURT-20",
        flag_below: float = 0.5,
    ):
        if not _BLEURT_AVAILABLE:
            raise ImportError(
                "BLEURTMetric requires evaluate, tensorflow, and the bleurt package.\n"
                "pip install evaluate tensorflow "
                "git+https://github.com/google-research/bleurt.git"
            )
        self.reference_source = reference_source
        self.model_name = model_name
        self.flag_below = flag_below
        self._scorer = None  # lazy-loaded on first compute()

    # ------------------------------------------------------------------

    def _get_scorer(self):
        if self._scorer is None:
            logger.info("Loading BLEURT checkpoint %r …", self.model_name)
            self._scorer = _evaluate.load("bleurt", self.model_name)
        return self._scorer

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

        scorer = self._get_scorer()

        # Score candidate against every reference; take the best.
        raw_scores: list[float] = scorer.compute(
            predictions=[question.text] * len(references),
            references=references,
        )["scores"]

        best_raw = max(raw_scores)
        # Clip: raw BLEURT can be negative (worse than chance) or slightly > 1.
        best_score = min(max(best_raw, 0.0), 1.0)

        quality = "strong" if best_score >= 0.9 else "moderate" if best_score >= 0.75 else "weak"

        return MetricResult(
            metric_name=self.name,
            score=best_score,
            rationale=(
                f"BLEURT {best_score:.4f} (raw: {best_raw:.4f}, {quality} semantic quality) "
                f"vs {len(references)} reference(s) from context.{self.reference_source} "
                f"[{self.model_name}]."
            ),
            flagged=best_score < self.flag_below,
            metadata={
                "reference_source": self.reference_source,
                "model_name": self.model_name,
                "n_references": len(references),
                "raw_score": best_raw,
            },
        )
