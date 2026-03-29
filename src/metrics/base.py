"""
Base classes for all metrics.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from src.models import Question, EvaluationContext, MetricResult


class BaseMetric(ABC):
    name: str
    description: str = ""

    @abstractmethod
    def compute(self, question: Question, context: EvaluationContext) -> MetricResult:
        """
        Evaluate the question against the provided context.

        Args:
            question : The assessment item.
            context  : Curriculum/course references the metric compares against.

        Returns:
            MetricResult with a normalised score (0.0–1.0).
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class BaseReferenceMetric(BaseMetric):
    """
    Mixin for metrics that score a question against reference texts drawn
    from EvaluationContext.

    Subclasses must set `reference_source` (typically in __init__) and
    inherit `_collect_references` for free.

    Supported reference_source values:
        "learning_objectives" → context.learning_objectives  (list[str])
        "course_content"      → context.course_content       (str)
        <any other str>       → context.metadata[reference_source]
                                (str or list[str])
    """

    reference_source: str

    def _collect_references(self, context: EvaluationContext) -> list[str]:
        if self.reference_source == "learning_objectives":
            return [o for o in context.learning_objectives if o.strip()]
        if self.reference_source == "course_content":
            return [context.course_content] if context.course_content else []
        # Fallback: arbitrary context.metadata key
        value = context.metadata.get(self.reference_source, [])
        if isinstance(value, str):
            return [value] if value.strip() else []
        return [v for v in value if isinstance(v, str) and v.strip()]