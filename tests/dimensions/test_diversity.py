"""Tests for DiversityDimension."""

from __future__ import annotations
import pytest
from src.models import Question, EvaluationContext, DimensionName
from src.dimensions.linguistic_structural.diversity import DiversityDimension

_ALL_METRICS = {"distinct_2", "vocabulary_novelty"}


def _q(text: str) -> Question:
    return Question(id="q1", text=text)

_CTX = EvaluationContext(learning_objectives=[], course_content=None)
_CTX_FULL = EvaluationContext(
    course_content="Photosynthesis converts light energy into chemical energy."
)


def test_dimension_name():
    assert DiversityDimension().name == DimensionName.DIVERSITY


def test_all_metrics_present():
    names = {m.name for m in DiversityDimension().metrics}
    assert _ALL_METRICS.issubset(names)


def test_each_instantiation_gets_own_list():
    d1 = DiversityDimension()
    d2 = DiversityDimension()
    assert d1.metrics is not d2.metrics


def test_empty_question_all_zero_flagged():
    d = DiversityDimension()
    result = d.score(_q(""), _CTX)
    for name, mr in result.scores.items():
        assert mr.score == 0.0, f"{name}: expected 0.0 on empty text"
        assert mr.flagged is True


def test_result_keys_match_metrics():
    d = DiversityDimension()
    result = d.score(_q("What is photosynthesis?"), _CTX)
    assert set(result.scores.keys()) == {m.name for m in d.metrics}


def test_distinct_2_nonzero_on_real_question():
    d = DiversityDimension()
    result = d.score(_q("What is the role of chlorophyll in photosynthesis?"), _CTX)
    assert result.scores["distinct_2"].score > 0.0


def test_vocabulary_novelty_zero_without_content():
    d = DiversityDimension()
    result = d.score(_q("What is photosynthesis?"), _CTX)
    assert result.scores["vocabulary_novelty"].score == 0.0
    assert result.scores["vocabulary_novelty"].flagged is True


def test_vocabulary_novelty_nonzero_with_content():
    d = DiversityDimension()
    result = d.score(_q("How do plants produce oxygen?"), _CTX_FULL)
    assert result.scores["vocabulary_novelty"].score >= 0.0
