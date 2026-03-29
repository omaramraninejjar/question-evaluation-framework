"""Tests for AmbiguityDimension."""

from __future__ import annotations
import pytest
from src.models import Question, EvaluationContext, DimensionName
from src.dimensions.linguistic_structural.ambiguity import AmbiguityDimension
from src.metrics.polysemy import _NLTK_WN_AVAILABLE

_CORE_METRICS = {"wh_word_type", "pronoun_ratio"}


def _q(text: str) -> Question:
    return Question(id="q1", text=text)

_CTX = EvaluationContext(learning_objectives=[], course_content=None)


def test_dimension_name():
    assert AmbiguityDimension().name == DimensionName.AMBIGUITY


def test_core_metrics_always_present():
    names = {m.name for m in AmbiguityDimension().metrics}
    assert _CORE_METRICS.issubset(names)


def test_polysemy_present_when_wordnet_available():
    if _NLTK_WN_AVAILABLE:
        names = {m.name for m in AmbiguityDimension().metrics}
        assert "polysemy_score" in names


def test_each_instantiation_gets_own_list():
    d1 = AmbiguityDimension()
    d2 = AmbiguityDimension()
    assert d1.metrics is not d2.metrics


def test_empty_question_all_zero_flagged():
    d = AmbiguityDimension()
    result = d.score(_q(""), _CTX)
    for name, mr in result.scores.items():
        assert mr.score == 0.0, f"{name}: expected 0.0 on empty text"
        assert mr.flagged is True


def test_result_keys_match_metrics():
    d = AmbiguityDimension()
    result = d.score(_q("What is photosynthesis?"), _CTX)
    assert set(result.scores.keys()) == {m.name for m in d.metrics}


def test_specific_question_scores_on_core_metrics():
    d = AmbiguityDimension()
    result = d.score(_q("Who discovered penicillin?"), _CTX)
    for name in _CORE_METRICS:
        assert result.scores[name].score >= 0.0
