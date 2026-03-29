"""
NLI Entailment metric.

Original reference (cross-encoder NLI models):
    He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention.
    ICLR 2021.  https://arxiv.org/abs/2006.03654
    (Backbone for the default cross-encoder/nli-deberta-v3-small checkpoint.)

    NLI task originally introduced in:
    Bowman et al. (2015). A large annotated corpus for learning natural language
    inference. EMNLP 2015.  https://aclanthology.org/D15-1075/

What it measures:
    A cross-encoder NLI (Natural Language Inference) model takes a
    (premise, hypothesis) pair and predicts one of three labels:
        contradiction — the hypothesis contradicts the premise
        neutral       — the hypothesis is independent of the premise
        entailment    — the hypothesis follows from / is supported by the premise

    In this framework:
        premise   = reference text (learning objective or course content)
        hypothesis = candidate question

    The entailment probability (after softmax over the three logits) is used
    as the score.  A high entailment score means the question is directly
    supported by / derived from the reference — a strong signal that the
    question genuinely tests the stated objective or relies on content present
    in the course.

Score range:
    [0.0, 1.0] — entailment probability after softmax.
    In practice, scores for well-aligned question–objective pairs typically
    fall in [0.4, 0.9]; unrelated pairs fall below 0.3.

Aggregation over multiple references:
    The candidate is scored against each reference independently;
    the maximum entailment probability across references is returned.

Model choice:
    model_name defaults to "cross-encoder/nli-deberta-v3-small" (~180 MB).
    "cross-encoder/nli-distilroberta-base" (~250 MB) is a faster alternative.
    Both are downloaded on first use and cached in ~/.cache/huggingface.

Label order:
    The default entailment_idx=1 corresponds to the label ordering
    [contradiction=0, entailment=1, neutral=2] used by the default model.
    Adjust entailment_idx if using a different checkpoint with a different
    label ordering.

Dependency:
    sentence-transformers >= 2.2   (pip install sentence-transformers)
    torch >= 2.1                   (already required by bert-score)
"""

from __future__ import annotations
import logging
import numpy as np
from src.models import Question, EvaluationContext, MetricResult
from src.metrics.base import BaseReferenceMetric

logger = logging.getLogger(__name__)

try:
    from sentence_transformers.cross_encoder import CrossEncoder as _CrossEncoder
    _NLI_AVAILABLE = True
except ImportError:
    _CrossEncoder = None  # type: ignore[assignment,misc]
    _NLI_AVAILABLE = False


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class NLIEntailmentMetric(BaseReferenceMetric):
    """
    NLI entailment score between context references (premise) and question (hypothesis).

    Args:
        reference_source : Which context field to use as references.
            "learning_objectives" → context.learning_objectives  (list[str])
            "course_content"      → context.course_content       (str)
        model_name       : HuggingFace cross-encoder NLI checkpoint.
            Default "cross-encoder/nli-deberta-v3-small" (~180 MB).
            "cross-encoder/nli-distilroberta-base" (~250 MB) is faster.
        entailment_idx   : Index of the entailment label in the model's output.
            Default 1 (contradiction=0, entailment=1, neutral=2) for the
            default checkpoint. Adjust if using a different model.
        flag_below       : Score threshold below which flagged=True.
    """

    name = "nli_entailment"
    description = (
        "NLI cross-encoder entailment probability: does the question follow from "
        "the reference? (max over references)."
    )

    def __init__(
        self,
        reference_source: str = "learning_objectives",
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        entailment_idx: int = 1,
        flag_below: float = 0.5,
    ):
        if not _NLI_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for NLIEntailmentMetric.\n"
                "pip install sentence-transformers"
            )
        self.reference_source = reference_source
        self.model_name = model_name
        self.entailment_idx = entailment_idx
        self.flag_below = flag_below
        self._model = None  # lazy-loaded on first compute()

    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            logger.info("Loading NLI cross-encoder %r …", self.model_name)
            self._model = _CrossEncoder(self.model_name)
        return self._model

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

        model = self._get_model()

        # NLI convention: (premise, hypothesis).
        # premise  = reference (objective / course content)
        # hypothesis = question  ("does the question follow from the objective?")
        pairs = [[ref, question.text] for ref in references]
        raw_logits = model.predict(pairs)  # shape: (n_refs, 3)

        # Ensure 2-D even for a single reference.
        raw_logits = np.atleast_2d(raw_logits)
        probs = _softmax(raw_logits)                          # (n_refs, 3)
        entailment_probs = probs[:, self.entailment_idx]      # (n_refs,)
        best_score = float(entailment_probs.max())

        quality = "strong" if best_score >= 0.7 else "moderate" if best_score >= 0.4 else "weak"

        return MetricResult(
            metric_name=self.name,
            score=best_score,
            rationale=(
                f"NLI entailment {best_score:.4f} ({quality} alignment) "
                f"vs {len(references)} reference(s) from context.{self.reference_source} "
                f"[{self.model_name}]."
            ),
            flagged=best_score < self.flag_below,
            metadata={
                "reference_source": self.reference_source,
                "model_name": self.model_name,
                "n_references": len(references),
                "entailment_idx": self.entailment_idx,
            },
        )
