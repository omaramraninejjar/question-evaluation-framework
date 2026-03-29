"""
Base class for all aspects.
"""

from __future__ import annotations
from abc import ABC
from src.models import AspectName, AspectResult, Question, EvaluationContext
from src.dimensions.base import BaseDimension


class BaseAspect(ABC):
    name: AspectName
    description: str = ""
    dimensions: list[BaseDimension] = []

    def evaluate(self, question: Question, context: EvaluationContext) -> AspectResult:
        return AspectResult(
            aspect=self.name,
            scores={
                dim.name.value: dim.score(question, context)
                for dim in self.dimensions
            },
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"dimensions={[d.name.value for d in self.dimensions]})"
        )