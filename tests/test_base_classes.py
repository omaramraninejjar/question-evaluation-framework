"""Tests for BaseMetric, BaseDimension, BaseAspect, and Evaluator."""

import pytest
from src.models import (
    AspectName, DimensionName, Question, EvaluationContext, MetricResult,
)
from src.metrics.base import BaseMetric
from src.dimensions.base import BaseDimension
from src.aspects.base import BaseAspect
from src.evaluator import Evaluator


def _q()   -> Question:          return Question(id="q1", text="Sample?")
def _ctx() -> EvaluationContext: return EvaluationContext()


class FixedMetric(BaseMetric):
    def __init__(self, name: str, score: float):
        self.name = name
        self._score = score
    def compute(self, question, context) -> MetricResult:
        return MetricResult(metric_name=self.name, score=self._score)


class SimpleDimension(BaseDimension):
    name = DimensionName.READABILITY
    def __init__(self, metrics=None):
        self.metrics = metrics or []


class SimpleAspect(BaseAspect):
    name = AspectName.LINGUISTIC_STRUCTURAL
    def __init__(self, dimensions=None):
        self.dimensions = dimensions or []


# BaseMetric
def test_metric_cannot_instantiate_abstract():
    with pytest.raises(TypeError): BaseMetric()

def test_metric_compute_returns_result():
    r = FixedMetric("foo", 0.75).compute(_q(), _ctx())
    assert r.score == 0.75 and r.metric_name == "foo"


# BaseDimension
def test_dimension_empty_metrics_returns_empty_dict():
    assert SimpleDimension().score(_q(), _ctx()).scores == {}

def test_dimension_scores_keyed_by_metric_name():
    dim = SimpleDimension(metrics=[FixedMetric("m1", 0.6), FixedMetric("m2", 0.8)])
    scores = dim.score(_q(), _ctx()).scores
    assert set(scores.keys()) == {"m1", "m2"}

def test_dimension_no_scalar_score():
    result = SimpleDimension(metrics=[FixedMetric("m", 0.5)]).score(_q(), _ctx())
    assert not hasattr(result, "score")


# BaseAspect
def test_aspect_empty_dimensions_returns_empty_dict():
    assert SimpleAspect().evaluate(_q(), _ctx()).scores == {}

def test_aspect_scores_keyed_by_dimension_name():
    d = SimpleDimension(metrics=[FixedMetric("m", 0.7)])
    asp = SimpleAspect(dimensions=[d])
    assert "readability" in asp.evaluate(_q(), _ctx()).scores

def test_aspect_no_scalar_score():
    assert not hasattr(SimpleAspect().evaluate(_q(), _ctx()), "score")


# Evaluator
def test_evaluator_default_has_four_aspects():
    assert len(Evaluator().aspects) == 4

def test_evaluator_question_id_propagated():
    ev = Evaluator(aspects=[])
    assert ev.evaluate(Question(id="qXYZ", text="test"), _ctx()).question_id == "qXYZ"

def test_evaluator_flat_scores():
    d = SimpleDimension(metrics=[FixedMetric("m", 0.9)])
    d.name = DimensionName.CURRICULUM_ALIGNMENT
    a = SimpleAspect(dimensions=[d])
    a.name = AspectName.PEDAGOGICAL
    result = Evaluator(aspects=[a]).evaluate(_q(), _ctx())
    assert result.flat_scores()["pedagogical"]["curriculum_alignment"]["m"] == pytest.approx(0.9)

def test_evaluator_no_overall_score():
    assert not hasattr(Evaluator(aspects=[]).evaluate(_q(), _ctx()), "overall_score")