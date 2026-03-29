"""Tests for shared data models."""

import pytest
from src.models import (
    AspectName, DimensionName,
    Question, EvaluationContext,
    MetricResult, DimensionResult, AspectResult, EvaluationResult,
)


def _metric(name="m", score=0.8) -> MetricResult:
    return MetricResult(metric_name=name, score=score)


class TestQuestion:

    def test_minimal(self):
        q = Question(id="q1", text="What is 2+2?")
        assert q.options is None and q.metadata == {}

    def test_no_learning_objective_field(self):
        # learning_objective was moved to EvaluationContext
        q = Question(id="q1", text="test")
        assert not hasattr(q, "learning_objective")


class TestEvaluationContext:

    def test_defaults(self):
        ctx = EvaluationContext()
        assert ctx.learning_objectives == []
        assert ctx.course_content is None
        assert ctx.rubric is None
        assert ctx.metadata == {}

    def test_fully_populated(self):
        ctx = EvaluationContext(
            learning_objectives=["explain photosynthesis", "identify reactants"],
            course_content="Chapter 6: Photosynthesis ...",
            rubric="Award 1 point for each correct stage.",
        )
        assert len(ctx.learning_objectives) == 2
        assert ctx.course_content is not None
        assert ctx.rubric is not None


class TestDimensionResult:

    def test_empty(self):
        r = DimensionResult(dimension=DimensionName.READABILITY)
        assert r.scores == {} and r.flagged_metrics() == []

    def test_flagged_metrics(self):
        r = DimensionResult(
            dimension=DimensionName.READABILITY,
            scores={
                "ok":  _metric("ok", 0.9),
                "bad": MetricResult(metric_name="bad", score=0.1, flagged=True),
            },
        )
        assert len(r.flagged_metrics()) == 1


class TestAspectResult:

    def test_flagged_dimensions(self):
        clean = DimensionResult(dimension=DimensionName.READABILITY, scores={"m": _metric()})
        flagged_dim = DimensionResult(
            dimension=DimensionName.HARMFUL_CONTENT_RISK,
            scores={"hcr": MetricResult(metric_name="hcr", score=0.05, flagged=True)},
        )
        r = AspectResult(
            aspect=AspectName.FAIRNESS_ETHICS,
            scores={"readability": clean, "harmful_content_risk": flagged_dim},
        )
        assert len(r.flagged_dimensions()) == 1


class TestEvaluationResult:

    def test_flat_scores(self):
        dim = DimensionResult(DimensionName.READABILITY, scores={"flesch": _metric("flesch", 0.75)})
        asp = AspectResult(AspectName.LINGUISTIC_STRUCTURAL, scores={"readability": dim})
        ev  = EvaluationResult(question_id="q1", scores={"linguistic_structural": asp})
        assert ev.flat_scores()["linguistic_structural"]["readability"]["flesch"] == 0.75

    def test_dimension_count(self):
        assert len(DimensionName) == 21