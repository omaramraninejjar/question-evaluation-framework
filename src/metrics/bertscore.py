"""
BERTScore metric.

Original paper:
    Zhang et al. (2020). BERTScore: Evaluating Text Generation with BERT.
    ICLR 2020.
    https://arxiv.org/abs/1904.09675

What it measures:
    Token-level semantic similarity between a candidate text and one or more
    reference texts using contextual embeddings from a pre-trained transformer.
    For each candidate token, BERTScore finds the most similar reference token
    via cosine similarity. Precision averages over candidate tokens; recall
    averages over reference tokens; F1 is their harmonic mean.

    Unlike n-gram metrics (BLEU, ROUGE, METEOR), BERTScore captures
    paraphrase and semantic equivalence without requiring surface-form overlap.
    Scores are in [0.0, 1.0]; baseline English F1 for unrelated texts is
    typically ~0.75–0.85, so meaningful thresholds sit higher than for
    n-gram metrics.

Score type:
    "f1"        (default) — harmonic mean of precision and recall
    "precision"           — fraction of candidate token embeddings matched
    "recall"              — fraction of reference token embeddings matched

Aggregation over multiple references:
    All references are scored in a single batched forward pass; the max
    score across references is returned.

Model choice:
    model_type defaults to "distilbert-base-uncased" for speed.
    Swap for "bert-base-uncased" or "roberta-large" for higher accuracy.

Dependency:
    bert-score >= 0.3.13   (pip install bert-score)
    torch >= 1.9           (installed as a bert-score dependency)
"""

from __future__ import annotations
import logging
from src.models import Question, EvaluationContext, MetricResult
from src.metrics.base import BaseReferenceMetric

logger = logging.getLogger(__name__)

try:
    import torch as _torch
    _torch.zeros(1)          # functional test — catches broken/partial installs
    from bert_score import score as _bert_score_fn
    _BERT_SCORE_AVAILABLE = True
except Exception:
    _torch = None            # type: ignore[assignment]
    _bert_score_fn = None    # type: ignore[assignment]
    _BERT_SCORE_AVAILABLE = False

_VALID_SCORE_TYPES = {"f1", "precision", "recall"}


class BERTScoreMetric(BaseReferenceMetric):
    """
    BERTScore between question.text and context references.

    Args:
        reference_source : Which context field to use as references.
            "learning_objectives" → context.learning_objectives  (list[str])
            "course_content"      → context.course_content       (str)
        score_type       : Which BERTScore component to return.
            "f1" (default), "precision", or "recall"
        model_type       : HuggingFace model identifier for embedding.
            Default "distilbert-base-uncased" (fast); use
            "bert-base-uncased" or "roberta-large" for higher accuracy.
        flag_below       : Score threshold below which flagged=True.
            Default 0.5 — tune upward (e.g. 0.85) once you know your
            typical score range on target data.
    """

    name = "bertscore"
    description = (
        "Token-level semantic similarity via contextual BERT embeddings "
        "(bert-score, max over references)."
    )

    def __init__(
        self,
        reference_source: str = "learning_objectives",
        score_type: str = "f1",
        model_type: str = "distilbert-base-uncased",
        flag_below: float = 0.5,
    ):
        if not _BERT_SCORE_AVAILABLE:
            raise ImportError(
                "bert-score is required for BERTScoreMetric.\n"
                "pip install bert-score"
            )
        if score_type not in _VALID_SCORE_TYPES:
            raise ValueError(
                f"score_type must be one of {_VALID_SCORE_TYPES}, got {score_type!r}"
            )

        self.reference_source = reference_source
        self.score_type = score_type
        self.model_type = model_type
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

        # Score the candidate against all references in one batched call,
        # then take the max across references.
        P, R, F1 = _bert_score_fn(
            cands=[question.text] * len(references),
            refs=references,
            model_type=self.model_type,
            verbose=False,
        )
        scores_tensor = {"precision": P, "recall": R, "f1": F1}[self.score_type]
        best_score = min(float(scores_tensor.max().item()), 1.0)

        quality = "strong" if best_score >= 0.9 else "moderate" if best_score >= 0.75 else "weak"

        return MetricResult(
            metric_name=self.name,
            score=best_score,
            rationale=(
                f"BERTScore {self.score_type} {best_score:.4f} ({quality} semantic similarity) "
                f"vs {len(references)} reference(s) from context.{self.reference_source} "
                f"[{self.model_type}]."
            ),
            flagged=best_score < self.flag_below,
            metadata={
                "reference_source": self.reference_source,
                "score_type": self.score_type,
                "n_references": len(references),
                "model_type": self.model_type,
            },
        )
