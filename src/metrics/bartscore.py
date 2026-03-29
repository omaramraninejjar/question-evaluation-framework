"""
BARTScore metric.

Original paper:
    Yuan et al. (2021). BARTScore: Evaluating Generated Text as Text Generation.
    NeurIPS 2021.
    https://arxiv.org/abs/2106.11520

What it measures:
    Likelihood-based semantic similarity framed as a text-generation problem.
    A pre-trained BART seq2seq model assigns an average token-level
    log-likelihood to the candidate given the reference (src2tgt direction),
    or vice versa (tgt2src), or both (f, harmonic mean of the two normalized
    directional scores).  Higher log-likelihood means the model considers the
    candidate more probable given the reference — a proxy for semantic quality.

Score direction:
    "src2tgt"  (default) — P(candidate | reference)
                           "given the learning objective, how likely is this question?"
    "tgt2src"            — P(reference | candidate)
                           "given the question, how likely is the objective?"
    "f"                  — harmonic mean of both normalized directional scores

Score normalization:
    Raw BARTScore values are average token log-likelihoods (negative; closer to
    0 is better). This implementation linearly maps [score_min, 0] → [0, 1] and
    clips the result. score_min defaults to -5.0, calibrated for
    facebook/bart-large-cnn; increase its magnitude for weaker checkpoints
    (e.g., score_min=-8.0 for facebook/bart-base on short texts).

Aggregation over multiple references:
    The candidate is scored against each reference independently; the maximum
    (best) score across references is returned.

Model choice:
    model_name defaults to "facebook/bart-large-cnn" (~1.6 GB, recommended).
    Use "facebook/bart-base" (~560 MB) for faster/lighter runs.

Dependency:
    transformers >= 4.x   (pip install transformers)
    torch >= 2.1          (already required by bert-score)
"""

from __future__ import annotations
import logging
from src.models import Question, EvaluationContext, MetricResult
from src.metrics.base import BaseReferenceMetric

logger = logging.getLogger(__name__)

try:
    import torch as _torch
    _torch.zeros(1)   # functional test — catches broken installs
    from transformers import (
        BartTokenizer as _BartTokenizer,
        BartForConditionalGeneration as _BartModel,
    )
    _BART_SCORE_AVAILABLE = True
except Exception:
    _torch = None          # type: ignore[assignment]
    _BartTokenizer = None  # type: ignore[assignment]
    _BartModel = None      # type: ignore[assignment]
    _BART_SCORE_AVAILABLE = False

_VALID_DIRECTIONS = {"src2tgt", "tgt2src", "f"}


class BARTScoreMetric(BaseReferenceMetric):
    """
    BARTScore between question.text and context references.

    Args:
        reference_source : Which context field to use as references.
            "learning_objectives" → context.learning_objectives  (list[str])
            "course_content"      → context.course_content       (str)
        model_name       : HuggingFace BART checkpoint.
            Default "facebook/bart-large-cnn" (~1.6 GB, recommended).
            "facebook/bart-base" (~560 MB) is faster but less accurate.
        score_direction  : Which conditional direction to compute.
            "src2tgt" (default) — P(candidate | reference)
            "tgt2src"           — P(reference | candidate)
            "f"                 — harmonic mean of both normalized directions
        score_min        : Lower bound for raw log-likelihood normalization.
            Scores at or below this value map to 0.0.
            Default -5.0 (calibrated for bart-large-cnn; use -8.0 for bart-base).
        max_length       : Tokenizer truncation limit (BART max: 1024).
        flag_below       : Score threshold below which flagged=True.
    """

    name = "bartscore"
    description = (
        "Likelihood-based semantic similarity via BART seq2seq log-probability "
        "(BARTScore, max over references)."
    )

    def __init__(
        self,
        reference_source: str = "learning_objectives",
        model_name: str = "facebook/bart-large-cnn",
        score_direction: str = "src2tgt",
        score_min: float = -5.0,
        max_length: int = 1024,
        flag_below: float = 0.5,
    ):
        if not _BART_SCORE_AVAILABLE:
            raise ImportError(
                "BARTScoreMetric requires transformers and torch.\n"
                "pip install transformers torch"
            )
        if score_direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"score_direction must be one of {_VALID_DIRECTIONS}, "
                f"got {score_direction!r}"
            )
        if score_min >= 0:
            raise ValueError("score_min must be negative (it is a log-likelihood lower bound)")

        self.reference_source = reference_source
        self.model_name = model_name
        self.score_direction = score_direction
        self.score_min = score_min
        self.max_length = max_length
        self.flag_below = flag_below
        self._model = None       # lazy-loaded on first compute()
        self._tokenizer = None

    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            logger.info("Loading BART checkpoint %r …", self.model_name)
            self._tokenizer = _BartTokenizer.from_pretrained(self.model_name)
            self._model = _BartModel.from_pretrained(self.model_name)
            self._model.eval()
        return self._model, self._tokenizer

    def _raw_score(self, src: str, tgt: str) -> float:
        """Average token log-likelihood of generating *tgt* given *src*."""
        model, tokenizer = self._get_model()
        src_enc = tokenizer(
            src, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        tgt_enc = tokenizer(
            tgt, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        with _torch.no_grad():
            loss = model(
                input_ids=src_enc["input_ids"],
                attention_mask=src_enc["attention_mask"],
                labels=tgt_enc["input_ids"],
            ).loss.item()
        return -loss  # negative NLL = avg token log-likelihood

    def _normalize(self, raw: float) -> float:
        """Map [score_min, 0] → [0, 1], clip outside range."""
        return max(0.0, min(1.0, (raw - self.score_min) / -self.score_min))

    def _score_pair(self, candidate: str, reference: str) -> float:
        if self.score_direction == "src2tgt":
            return self._normalize(self._raw_score(reference, candidate))
        if self.score_direction == "tgt2src":
            return self._normalize(self._raw_score(candidate, reference))
        # "f": harmonic mean of both normalized directional scores
        s2t = self._normalize(self._raw_score(reference, candidate))
        t2s = self._normalize(self._raw_score(candidate, reference))
        if s2t + t2s == 0.0:
            return 0.0
        return 2 * s2t * t2s / (s2t + t2s)

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

        scores = [self._score_pair(question.text, ref) for ref in references]
        best_score = max(scores)

        quality = "strong" if best_score >= 0.9 else "moderate" if best_score >= 0.75 else "weak"

        return MetricResult(
            metric_name=self.name,
            score=best_score,
            rationale=(
                f"BARTScore ({self.score_direction}) {best_score:.4f} "
                f"({quality} semantic quality) "
                f"vs {len(references)} reference(s) from context.{self.reference_source} "
                f"[{self.model_name}]."
            ),
            flagged=best_score < self.flag_below,
            metadata={
                "reference_source": self.reference_source,
                "model_name": self.model_name,
                "score_direction": self.score_direction,
                "n_references": len(references),
            },
        )
