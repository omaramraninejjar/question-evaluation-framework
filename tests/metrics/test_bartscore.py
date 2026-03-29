"""
Tests for BARTScoreMetric (src/metrics/bartscore.py).

Requires:
    pip install transformers torch   # torch already installed for BERTScore
    # facebook/bart-base (~560 MB) is downloaded on first run and cached.

Score behaviour (after normalization with score_min=-8.0 for bart-base):
    - Identical/closely related texts → higher score
    - Unrelated texts                 → lower score
    Use relative ordering rather than hard absolute thresholds.
"""

from __future__ import annotations

import pytest

from src.models import Question, EvaluationContext
from src.metrics.bartscore import BARTScoreMetric, _BART_SCORE_AVAILABLE

if not _BART_SCORE_AVAILABLE:
    pytest.skip(
        "BARTScore unavailable — install: pip install transformers torch",
        allow_module_level=True,
    )

# Use bart-base for tests: smaller download (~560 MB), same API.
# score_min is looser (-8.0) to account for bart-base's higher NLL.
_MODEL = "facebook/bart-base"
_SCORE_MIN = -8.0


def _q(text: str) -> Question:
    return Question(id="q1", text=text)

def _ctx(objectives: list[str] | None = None, course_content: str | None = None) -> EvaluationContext:
    return EvaluationContext(
        learning_objectives=objectives or [],
        course_content=course_content,
    )

def _m(**kwargs) -> BARTScoreMetric:
    return BARTScoreMetric(model_name=_MODEL, score_min=_SCORE_MIN, **kwargs)


# Construction
def test_invalid_direction_raises():
    with pytest.raises(ValueError):
        BARTScoreMetric(score_direction="invalid")

def test_invalid_score_min_raises():
    with pytest.raises(ValueError):
        BARTScoreMetric(score_min=0.5)

# No references / empty input
def test_no_references_returns_zero_flagged():
    result = _m().compute(_q("What is osmosis?"), _ctx())
    assert result.score == 0.0 and result.flagged is True

def test_empty_candidate_returns_zero_flagged():
    result = _m().compute(_q(""), _ctx(objectives=["understand osmosis"]))
    assert result.score == 0.0 and result.flagged is True

# Core behaviour
def test_score_in_unit_interval():
    result = _m().compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert 0.0 <= result.score <= 1.0

def test_related_scores_higher_than_unrelated():
    ctx = _ctx(objectives=["identify prime numbers"])
    related   = _m().compute(_q("identify prime numbers in a list"), ctx)
    unrelated = _m().compute(_q("describe the water cycle and evaporation"), ctx)
    assert related.score > unrelated.score

# Score direction variants
def test_tgt2src_direction():
    result = _m(score_direction="tgt2src").compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert 0.0 <= result.score <= 1.0
    assert result.metadata["score_direction"] == "tgt2src"

def test_f_direction():
    result = _m(score_direction="f").compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert 0.0 <= result.score <= 1.0
    assert result.metadata["score_direction"] == "f"

# Concept Coverage use case
def test_course_content_reference_source():
    result = _m(reference_source="course_content").compute(
        _q("what is photosynthesis"),
        _ctx(course_content="photosynthesis converts light to energy"),
    )
    assert 0.0 <= result.score <= 1.0
    assert result.metadata["reference_source"] == "course_content"

def test_missing_course_content_returns_zero_flagged():
    result = _m(reference_source="course_content").compute(
        _q("what is osmosis?"), _ctx()
    )
    assert result.score == 0.0 and result.flagged is True

# Multiple references — max score is kept
def test_multiple_objectives_picks_best():
    ctx_multi  = _ctx(objectives=["identify prime numbers", "classify numbers as prime or composite"])
    ctx_single = _ctx(objectives=["identify prime numbers"])
    multi  = _m().compute(_q("identify prime numbers"), ctx_multi)
    single = _m().compute(_q("identify prime numbers"), ctx_single)
    assert multi.score >= single.score
    assert multi.metadata["n_references"] == 2

# Metadata correctness
def test_metadata_contains_model_name():
    result = _m().compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert result.metadata["model_name"] == _MODEL
    assert result.metadata["score_direction"] == "src2tgt"

# Flagging threshold
def test_flagged_when_score_below_threshold():
    result = _m(flag_below=0.999).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["describe the water cycle"]),
    )
    assert result.flagged is True

def test_not_flagged_when_score_above_threshold():
    result = _m(flag_below=0.0).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify prime numbers"]),
    )
    assert result.flagged is False
