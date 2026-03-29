"""
Base class for readability metrics.

Readability metrics differ from reference-based metrics in two ways:
  - They score question.text directly — no reference is needed.
  - The EvaluationContext is accepted by compute() but not used.

Subclasses implement:
  _compute_raw(text)  → raw metric value (float)
  _normalize(raw)     → normalised score in [0.0, 1.0]
  _quality(score)     → human-readable quality label (str)
"""

from __future__ import annotations
from abc import abstractmethod
from src.models import Question, EvaluationContext, MetricResult
from src.metrics.base import BaseMetric


class BaseReadabilityMetric(BaseMetric):
    """
    Mixin for metrics that characterise question.text without any reference.

    Handles the empty-text early exit and delegates raw computation and
    normalisation to subclasses.
    """

    flag_below: float  # subclasses set in __init__

    @abstractmethod
    def _compute_raw(self, text: str) -> float:
        """Return the raw (unnormalised) metric value for *text*."""
        ...

    @abstractmethod
    def _normalize(self, raw: float) -> float:
        """Map the raw value to [0.0, 1.0]."""
        ...

    def _quality(self, score: float) -> str:
        return "good" if score >= 0.6 else "moderate" if score >= 0.3 else "poor"

    def compute(self, question: Question, context: EvaluationContext) -> MetricResult:
        if not question.text.strip():
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                rationale="Question text is empty.",
                flagged=True,
                metadata={"reason": "empty_text"},
            )
        raw = self._compute_raw(question.text)
        score = self._normalize(raw)
        return self._build_result(score, raw)

    @abstractmethod
    def _build_result(self, score: float, raw: float) -> MetricResult:
        """Build and return the MetricResult from score and raw value."""
        ...
