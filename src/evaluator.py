"""
Evaluator — top-level orchestrator.
"""

from __future__ import annotations
from src.models import EvaluationResult, Question, EvaluationContext
from src.aspects.base import BaseAspect
from src.aspects.pedagogical import PedagogicalAspect
from src.aspects.psychometric import PsychometricAspect
from src.aspects.linguistic_structural import LinguisticStructuralAspect
from src.aspects.fairness_ethics import FairnessEthicsAspect


class Evaluator:
    """
    Runs the full Aspect -> Dimension -> Metric pipeline for a single
    (Question, EvaluationContext) pair.

    Usage:
        context   = EvaluationContext(
                        learning_objectives=["explain photosynthesis"],
                        course_content="Chapter 6: Photosynthesis ...",
                    )
        result    = Evaluator().evaluate(question, context)
        flat      = result.flat_scores()
    """

    def __init__(self, aspects: list[BaseAspect] | None = None):
        self.aspects: list[BaseAspect] = aspects or [
            PedagogicalAspect(),
            PsychometricAspect(),
            LinguisticStructuralAspect(),
            FairnessEthicsAspect(),
        ]

    def evaluate(self, question: Question, context: EvaluationContext) -> EvaluationResult:
        return EvaluationResult(
            question_id=question.id,
            scores={
                aspect.name.value: aspect.evaluate(question, context)
                for aspect in self.aspects
            },
        )