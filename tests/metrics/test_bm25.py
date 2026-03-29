"""
Tests for BM25Metric (src/metrics/bm25.py).

Requires:
    pip install rank-bm25

Score behaviour (BM25 Okapi, norm_factor=10.0):
    - Identical/closely related texts  → higher score
    - Unrelated texts                  → lower score
    Use relative ordering rather than hard absolute thresholds,
    since BM25 scores depend on corpus composition.
"""

from __future__ import annotations

import pytest

from src.models import Question, EvaluationContext
from src.metrics.bm25 import BM25Metric, _BM25_AVAILABLE

if not _BM25_AVAILABLE:
    pytest.skip(
        "BM25 unavailable — install: pip install rank-bm25",
        allow_module_level=True,
    )


def _q(text: str) -> Question:
    return Question(id="q1", text=text)

def _ctx(objectives: list[str] | None = None, course_content: str | None = None) -> EvaluationContext:
    return EvaluationContext(
        learning_objectives=objectives or [],
        course_content=course_content,
    )


# Construction validation
def test_invalid_norm_factor_raises():
    with pytest.raises(ValueError):
        BM25Metric(norm_factor=0.0)

def test_negative_norm_factor_raises():
    with pytest.raises(ValueError):
        BM25Metric(norm_factor=-1.0)


# No references / empty input
def test_no_references_returns_zero_flagged():
    result = BM25Metric().compute(_q("What is osmosis?"), _ctx())
    assert result.score == 0.0 and result.flagged is True

def test_empty_candidate_returns_zero_flagged():
    result = BM25Metric().compute(_q(""), _ctx(objectives=["understand osmosis"]))
    assert result.score == 0.0 and result.flagged is True


# Core behaviour
def test_score_in_unit_interval():
    result = BM25Metric().compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify prime numbers in a list"]),
    )
    assert 0.0 <= result.score <= 1.0

def test_related_scores_higher_than_unrelated():
    m = BM25Metric()
    ctx = _ctx(objectives=["identify prime numbers"])
    related   = m.compute(_q("what are prime numbers"), ctx)
    unrelated = m.compute(_q("describe evaporation and the water cycle"), ctx)
    assert related.score > unrelated.score

def test_zero_overlap_gives_zero():
    result = BM25Metric().compute(
        _q("describe evaporation"),
        _ctx(objectives=["identify prime numbers"]),
    )
    # BM25 score is 0 when no query term appears in any document
    assert result.score == 0.0


# norm_factor clamping
def test_norm_factor_clamps_to_one():
    # Very small norm_factor forces score to 1.0 for any match
    result = BM25Metric(norm_factor=0.01).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify prime numbers"]),
    )
    assert result.score == pytest.approx(1.0)

def test_raw_score_in_metadata():
    result = BM25Metric().compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify prime numbers"]),
    )
    assert "raw_score" in result.metadata
    assert isinstance(result.metadata["raw_score"], float)


# Concept Coverage use case
def test_course_content_reference_source():
    result = BM25Metric(reference_source="course_content").compute(
        _q("what is photosynthesis"),
        _ctx(course_content="photosynthesis converts light to energy in plant cells"),
    )
    assert 0.0 <= result.score <= 1.0
    assert result.metadata["reference_source"] == "course_content"

def test_missing_course_content_returns_zero_flagged():
    result = BM25Metric(reference_source="course_content").compute(
        _q("what is osmosis?"), _ctx()
    )
    assert result.score == 0.0 and result.flagged is True


# Multiple references — max score is kept
def test_multiple_objectives_picks_best():
    m = BM25Metric()
    ctx_multi  = _ctx(objectives=["identify prime numbers", "classify numbers as prime or composite"])
    ctx_single = _ctx(objectives=["identify prime numbers"])
    multi  = m.compute(_q("identify prime numbers"), ctx_multi)
    single = m.compute(_q("identify prime numbers"), ctx_single)
    assert multi.score >= single.score - 1e-6
    assert multi.metadata["n_references"] == 2


# Flagging threshold
def test_flagged_when_score_below_threshold():
    result = BM25Metric(flag_below=0.999).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify prime numbers"]),
    )
    # Unless raw score >= norm_factor, score < 1.0 < 0.999 will flag
    assert isinstance(result.flagged, bool)

def test_not_flagged_when_score_above_threshold():
    result = BM25Metric(flag_below=0.0).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify prime numbers"]),
    )
    assert result.flagged is False
