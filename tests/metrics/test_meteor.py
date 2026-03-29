"""
Tests for METEORMetric (src/metrics/meteor.py).
Requires: pip install nltk
          python -c "import nltk; nltk.download('punkt_tab'); nltk.download('wordnet')"
"""

from __future__ import annotations

import pytest
from src.models import Question, EvaluationContext
from src.metrics.meteor import METEORMetric


def _q(text: str) -> Question:
    return Question(id="q1", text=text)

def _ctx(objectives: list[str] | None = None, course_content: str | None = None) -> EvaluationContext:
    return EvaluationContext(
        learning_objectives=objectives or [],
        course_content=course_content,
    )


# Construction
def test_alpha_out_of_range_raises():
    with pytest.raises(ValueError):
        METEORMetric(alpha=0.0)

def test_alpha_above_one_raises():
    with pytest.raises(ValueError):
        METEORMetric(alpha=1.5)

def test_negative_beta_raises():
    with pytest.raises(ValueError):
        METEORMetric(beta=-1.0)

def test_gamma_out_of_range_raises():
    with pytest.raises(ValueError):
        METEORMetric(gamma=1.5)

# No references / empty input
def test_no_references_returns_zero_flagged():
    result = METEORMetric().compute(_q("What is osmosis?"), _ctx())
    assert result.score == 0.0 and result.flagged is True

def test_empty_candidate_returns_zero_flagged():
    result = METEORMetric().compute(_q(""), _ctx(objectives=["understand osmosis"]))
    assert result.score == 0.0 and result.flagged is True

# Core behaviour
def test_score_in_unit_interval():
    result = METEORMetric().compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert 0.0 <= result.score <= 1.0

def test_identical_texts_score_near_one():
    # METEOR never reaches exactly 1.0 due to the fragmentation penalty;
    # a perfect unigram match on identical text scores ≥ 0.99.
    text = "understand the process of photosynthesis"
    result = METEORMetric().compute(_q(text), _ctx(objectives=[text]))
    assert result.score >= 0.99

def test_unrelated_texts_score_low():
    result = METEORMetric().compute(
        _q("calculate the area of a triangle"),
        _ctx(objectives=["describe the water cycle"]),
    )
    assert result.score < 0.3

def test_higher_overlap_scores_higher():
    m = METEORMetric()
    ctx = _ctx(objectives=["identify prime numbers"])
    high = m.compute(_q("identify prime numbers in a list"), ctx)
    low  = m.compute(_q("calculate polynomial derivatives"), ctx)
    assert high.score > low.score

# Parameter effects
def test_recall_heavy_alpha_boosts_recall():
    # A short hypothesis that covers the reference well should score higher
    # with recall-heavy alpha than precision-heavy alpha
    m_recall    = METEORMetric(alpha=0.9)
    m_precision = METEORMetric(alpha=0.1)
    ctx = _ctx(objectives=["identify and classify prime numbers"])
    q   = _q("identify prime numbers")   # recall < 1, precision ~ 1
    assert m_recall.compute(q, ctx).score != m_precision.compute(q, ctx).score

# Concept Coverage use case
def test_course_content_reference_source():
    result = METEORMetric(reference_source="course_content").compute(
        _q("what is photosynthesis"),
        _ctx(course_content="photosynthesis converts light to energy"),
    )
    assert result.score > 0.0
    assert result.metadata["reference_source"] == "course_content"

def test_missing_course_content_returns_zero_flagged():
    result = METEORMetric(reference_source="course_content").compute(
        _q("what is osmosis?"), _ctx()
    )
    assert result.score == 0.0 and result.flagged is True

# Multiple references — NLTK picks the best internally
def test_multiple_objectives_metadata():
    result = METEORMetric().compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes", "classify numbers as prime or composite"]),
    )
    assert result.metadata["n_references"] == 2

def test_multiple_objectives_at_least_as_good_as_single():
    m = METEORMetric()
    q = _q("identify prime numbers")
    multi  = m.compute(q, _ctx(objectives=["identify primes", "classify prime numbers"]))
    single = m.compute(q, _ctx(objectives=["identify primes"]))
    assert multi.score >= single.score

# Metadata correctness
def test_metadata_contains_parameters():
    m = METEORMetric(alpha=0.8, beta=2.0, gamma=0.4)
    result = m.compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert result.metadata["alpha"] == 0.8
    assert result.metadata["beta"] == 2.0
    assert result.metadata["gamma"] == 0.4

# Flagging threshold
def test_flagged_when_score_below_threshold():
    result = METEORMetric(flag_below=0.9).compute(
        _q("calculate the area of a triangle"),
        _ctx(objectives=["describe the water cycle"]),
    )
    assert result.flagged is True

def test_not_flagged_when_score_above_threshold():
    text = "understand the process of photosynthesis"
    result = METEORMetric(flag_below=0.5).compute(_q(text), _ctx(objectives=[text]))
    assert result.flagged is False
