"""
Tests for ROUGELMetric (src/metrics/rouge.py).
Requires: pip install rouge-score
"""

from __future__ import annotations

import pytest
from src.models import Question, EvaluationContext
from src.metrics.rouge import ROUGELMetric


def _q(text: str) -> Question:
    return Question(id="q1", text=text)

def _ctx(objectives: list[str] | None = None, course_content: str | None = None) -> EvaluationContext:
    return EvaluationContext(
        learning_objectives=objectives or [],
        course_content=course_content,
    )


# Construction
def test_invalid_score_type_raises():
    with pytest.raises(ValueError):
        ROUGELMetric(score_type="invalid")

# No references / empty input
def test_no_references_returns_zero_flagged():
    result = ROUGELMetric().compute(_q("What is osmosis?"), _ctx())
    assert result.score == 0.0 and result.flagged is True

def test_empty_candidate_returns_zero_flagged():
    result = ROUGELMetric().compute(_q(""), _ctx(objectives=["understand osmosis"]))
    assert result.score == 0.0 and result.flagged is True

# Core behaviour
def test_score_in_unit_interval():
    result = ROUGELMetric().compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert 0.0 <= result.score <= 1.0

def test_identical_texts_score_one():
    text = "understand the process of photosynthesis"
    result = ROUGELMetric().compute(_q(text), _ctx(objectives=[text]))
    assert result.score == pytest.approx(1.0, abs=1e-4)

def test_unrelated_texts_score_low():
    result = ROUGELMetric().compute(
        _q("calculate the area of a triangle"),
        _ctx(objectives=["describe the water cycle"]),
    )
    assert result.score < 0.3

def test_higher_overlap_scores_higher():
    m = ROUGELMetric()
    ctx = _ctx(objectives=["identify prime numbers"])
    high = m.compute(_q("identify prime numbers in a list"), ctx)
    low  = m.compute(_q("calculate polynomial derivatives"), ctx)
    assert high.score > low.score

# Score type variants
def test_recall_score_type():
    result = ROUGELMetric(score_type="recall").compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert 0.0 <= result.score <= 1.0
    assert result.metadata["score_type"] == "recall"

def test_precision_score_type():
    result = ROUGELMetric(score_type="precision").compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert 0.0 <= result.score <= 1.0
    assert result.metadata["score_type"] == "precision"

# Concept Coverage use case
def test_course_content_reference_source():
    result = ROUGELMetric(reference_source="course_content").compute(
        _q("what is photosynthesis"),
        _ctx(course_content="photosynthesis converts light to energy"),
    )
    assert result.score > 0.0
    assert result.metadata["reference_source"] == "course_content"

def test_missing_course_content_returns_zero_flagged():
    result = ROUGELMetric(reference_source="course_content").compute(
        _q("what is osmosis?"), _ctx()
    )
    assert result.score == 0.0 and result.flagged is True

# Multiple references — best score is kept
def test_multiple_objectives_picks_best():
    m = ROUGELMetric()
    ctx_multi  = _ctx(objectives=["identify prime numbers", "classify numbers as prime or composite"])
    ctx_single = _ctx(objectives=["identify prime numbers"])
    multi  = m.compute(_q("identify prime numbers"), ctx_multi)
    single = m.compute(_q("identify prime numbers"), ctx_single)
    # More references can only keep or improve the best score
    assert multi.score >= single.score
    assert multi.metadata["n_references"] == 2

# Flagging threshold
def test_flagged_when_score_below_threshold():
    result = ROUGELMetric(flag_below=0.9).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["describe the water cycle"]),
    )
    assert result.flagged is True

def test_not_flagged_when_score_above_threshold():
    text = "understand the process of photosynthesis"
    result = ROUGELMetric(flag_below=0.5).compute(_q(text), _ctx(objectives=[text]))
    assert result.flagged is False
