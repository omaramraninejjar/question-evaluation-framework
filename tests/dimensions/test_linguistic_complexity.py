"""Tests for LinguisticComplexityDimension."""

from __future__ import annotations
import pytest
from src.models import Question, EvaluationContext, DimensionName
from src.dimensions.linguistic_structural.linguistic_complexity import LinguisticComplexityDimension
from src.metrics.zipf_frequency import _WORDFREQ_AVAILABLE
from src.metrics.parse_tree_depth import _SPACY_AVAILABLE


def _q(text: str) -> Question:
    return Question(id="q1", text=text)

_CTX = EvaluationContext(learning_objectives=[], course_content=None)


def test_dimension_name():
    assert LinguisticComplexityDimension().name == DimensionName.LINGUISTIC_COMPLEXITY


def test_each_instantiation_gets_own_list():
    d1 = LinguisticComplexityDimension()
    d2 = LinguisticComplexityDimension()
    assert d1.metrics is not d2.metrics


def test_empty_question_all_zero_flagged():
    d = LinguisticComplexityDimension()
    if not d.metrics:
        pytest.skip("No metrics loaded — install wordfreq or spacy")
    result = d.score(_q(""), _CTX)
    for name, mr in result.scores.items():
        assert mr.score == 0.0, f"{name}: expected 0.0 on empty text"
        assert mr.flagged is True


def test_result_keys_match_metrics():
    d = LinguisticComplexityDimension()
    if not d.metrics:
        pytest.skip("No metrics loaded")
    result = d.score(_q("What is the role of mitochondria?"), _CTX)
    assert set(result.scores.keys()) == {m.name for m in d.metrics}


@pytest.mark.skipif(not _WORDFREQ_AVAILABLE, reason="wordfreq not installed")
def test_zipf_metric_present():
    names = {m.name for m in LinguisticComplexityDimension().metrics}
    assert "zipf_word_frequency" in names


@pytest.mark.skipif(not _SPACY_AVAILABLE, reason="spacy not installed")
def test_spacy_metrics_present():
    names = {m.name for m in LinguisticComplexityDimension().metrics}
    assert "parse_tree_depth" in names
    assert "dependency_distance" in names
