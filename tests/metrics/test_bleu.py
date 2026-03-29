"""
Tests for BLEUMetric (src/metrics/bleu.py).
Requires: pip install nltk && python -c "import nltk; nltk.download('punkt_tab')"
"""

from __future__ import annotations

import pytest
from src.models import Question, EvaluationContext
from src.metrics.bleu import BLEUMetric


def _q(text: str) -> Question:
    return Question(id="q1", text=text)

def _ctx(objectives: list[str] | None = None, course_content: str | None = None) -> EvaluationContext:
    return EvaluationContext(
        learning_objectives=objectives or [],
        course_content=course_content,
    )


# Construction
def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        BLEUMetric(weights=(0.5, 0.5, 0.5, 0.5))

# No references / empty input
def test_no_references_returns_zero_flagged():
    result = BLEUMetric().compute(_q("What is osmosis?"), _ctx())
    assert result.score == 0.0 and result.flagged is True

def test_empty_candidate_returns_zero_flagged():
    result = BLEUMetric().compute(_q(""), _ctx(objectives=["understand osmosis"]))
    assert result.score == 0.0 and result.flagged is True

# Core behaviour
def test_score_in_unit_interval():
    result = BLEUMetric().compute(_q("identify prime numbers"), _ctx(objectives=["identify primes"]))
    assert 0.0 <= result.score <= 1.0

def test_identical_texts_score_one():
    text = "understand the process of photosynthesis"
    result = BLEUMetric().compute(_q(text), _ctx(objectives=[text]))
    assert result.score == pytest.approx(1.0, abs=1e-4)

def test_unrelated_texts_score_low():
    result = BLEUMetric().compute(
        _q("calculate the area of a triangle"),
        _ctx(objectives=["describe the water cycle"]),
    )
    assert result.score < 0.2

def test_higher_overlap_scores_higher():
    m = BLEUMetric()
    ctx = _ctx(objectives=["identify prime numbers"])
    high = m.compute(_q("identify prime numbers in a list"), ctx)
    low  = m.compute(_q("calculate polynomial derivatives"), ctx)
    assert high.score > low.score

# Concept Coverage use case
def test_course_content_reference_source():
    result = BLEUMetric(reference_source="course_content").compute(
        _q("what is photosynthesis"),
        _ctx(course_content="photosynthesis converts light to energy"),
    )
    assert result.score > 0.0
    assert result.metadata["reference_source"] == "course_content"

def test_missing_course_content_returns_zero_flagged():
    result = BLEUMetric(reference_source="course_content").compute(_q("what is osmosis?"), _ctx())
    assert result.score == 0.0 and result.flagged is True

# Multiple learning objectives
def test_multiple_objectives():
    result = BLEUMetric().compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes", "classify numbers as prime or composite"]),
    )
    assert result.metadata["n_references"] == 2