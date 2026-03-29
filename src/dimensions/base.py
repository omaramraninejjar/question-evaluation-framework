"""
Base class for all dimensions.
"""

from __future__ import annotations
from abc import ABC
from src.models import DimensionName, DimensionResult, Question, EvaluationContext
from src.metrics.base import BaseMetric


class BaseDimension(ABC):
    name: DimensionName
    description: str = ""
    metrics: list[BaseMetric] = []

    def score(self, question: Question, context: EvaluationContext) -> DimensionResult:
        return DimensionResult(
            dimension=self.name,
            scores={m.name: m.compute(question, context) for m in self.metrics},
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, metrics={[m.name for m in self.metrics]})"
        )