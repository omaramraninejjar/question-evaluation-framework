"""Tests for WellFormednessDimension."""

from __future__ import annotations
import pytest
from src.models import Question, EvaluationContext, DimensionName
from src.dimensions.linguistic_structural.well_formedness import WellFormednessDimension
from src.metrics.lm_perplexity import _LM_AVAILABLE

_CORE_METRICS = {"question_mark", "negation_rate", "has_verb"}


def _q(text: str) -> Question:
    return Question(id="q1", text=text)

_CTX = EvaluationContext(learning_objectives=[], course_content=None)


def test_dimension_name():
    assert WellFormednessDimension().name == DimensionName.WELL_FORMEDNESS


def test_core_metrics_always_present():
    names = {m.name for m in WellFormednessDimension().metrics}
    assert _CORE_METRICS.issubset(names)


def test_lm_perplexity_present_when_available():
    if _LM_AVAILABLE:
        names = {m.name for m in WellFormednessDimension().metrics}
        assert "lm_perplexity" in names


def test_each_instantiation_gets_own_list():
    d1 = WellFormednessDimension()
    d2 = WellFormednessDimension()
    assert d1.metrics is not d2.metrics


def test_empty_question_all_zero_flagged():
    d = WellFormednessDimension()
    result = d.score(_q(""), _CTX)
    for name, mr in result.scores.items():
        assert mr.score == 0.0, f"{name}: expected 0.0 on empty text"
        assert mr.flagged is True


def test_result_keys_match_metrics():
    """Use empty text so no model inference is triggered."""
    d = WellFormednessDimension()
    result = d.score(_q(""), _CTX)
    assert set(result.scores.keys()) == {m.name for m in d.metrics}


def test_well_formed_question_scores_well():
    """A proper question should score 1.0 on question_mark, has_verb, and negation_rate."""
    d = WellFormednessDimension()
    # Score only on core (no-LM) metrics to avoid GPT-2 download in tests
    q = _q("What does the mitochondrion produce?")
    from src.metrics.question_mark import QuestionMarkMetric
    from src.metrics.has_verb import HasVerbMetric
    from src.metrics.negation_rate import NegationRateMetric
    assert QuestionMarkMetric().compute(q, _CTX).score == pytest.approx(1.0)
    assert HasVerbMetric().compute(q, _CTX).score == pytest.approx(1.0)
    assert NegationRateMetric().compute(q, _CTX).score == pytest.approx(1.0)


def test_negated_question_flagged():
    from src.metrics.negation_rate import NegationRateMetric
    r = NegationRateMetric().compute(_q("Which of the following is NOT correct?"), _CTX)
    assert r.score < 1.0
