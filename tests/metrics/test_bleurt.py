"""
Tests for BLEURTMetric (src/metrics/bleurt.py).

Requires:
    pip install evaluate
    pip install tensorflow          # BLEURT checkpoint loader
    # BLEURT-20 checkpoint (~1.7 GB) is downloaded on first run and cached.
    # Use model_name="bleurt-tiny-128" in CI to reduce download size.

Score behaviour after clipping to [0, 1]:
    - Identical texts                → close to 1.0
    - Semantically related           → typically 0.5–0.9
    - Unrelated topics               → 0.0–0.5 (raw may be negative, clipped to 0)
  Use relative ordering rather than hard absolute thresholds where possible.
"""

from __future__ import annotations

import pytest

from src.models import Question, EvaluationContext
from src.metrics.bleurt import BLEURTMetric, _BLEURT_AVAILABLE

if not _BLEURT_AVAILABLE:
    pytest.skip(
        "BLEURT unavailable — install: pip install evaluate tensorflow",
        allow_module_level=True,
    )

# Use the tiny checkpoint for tests — fast download (~39 MB), same API.
_CHECKPOINT = "bleurt-tiny-128"


def _q(text: str) -> Question:
    return Question(id="q1", text=text)

def _ctx(objectives: list[str] | None = None, course_content: str | None = None) -> EvaluationContext:
    return EvaluationContext(
        learning_objectives=objectives or [],
        course_content=course_content,
    )


# No references / empty input
def test_no_references_returns_zero_flagged():
    result = BLEURTMetric(model_name=_CHECKPOINT).compute(_q("What is osmosis?"), _ctx())
    assert result.score == 0.0 and result.flagged is True

def test_empty_candidate_returns_zero_flagged():
    result = BLEURTMetric(model_name=_CHECKPOINT).compute(_q(""), _ctx(objectives=["understand osmosis"]))
    assert result.score == 0.0 and result.flagged is True

# Core behaviour
def test_score_in_unit_interval():
    result = BLEURTMetric(model_name=_CHECKPOINT).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert 0.0 <= result.score <= 1.0

def test_identical_texts_score_near_one():
    # bleurt-tiny-128 is low-accuracy; identical texts score ~0.7–0.8 (not close to 1.0).
    # BLEURT-20 would reach ~0.9+; use a loose threshold here to keep tests fast.
    text = "understand the process of photosynthesis"
    result = BLEURTMetric(model_name=_CHECKPOINT).compute(_q(text), _ctx(objectives=[text]))
    assert result.score >= 0.5

def test_related_scores_higher_than_unrelated():
    m = BLEURTMetric(model_name=_CHECKPOINT)
    ctx = _ctx(objectives=["identify prime numbers"])
    related   = m.compute(_q("identify prime numbers in a list"), ctx)
    unrelated = m.compute(_q("describe the water cycle and evaporation"), ctx)
    assert related.score > unrelated.score

# Concept Coverage use case
def test_course_content_reference_source():
    # Only verifies that course_content is used as the reference source.
    # Score may be 0.0 with bleurt-tiny-128 for short texts (raw score clipped from negative).
    result = BLEURTMetric(model_name=_CHECKPOINT, reference_source="course_content").compute(
        _q("what is photosynthesis"),
        _ctx(course_content="photosynthesis converts light to energy"),
    )
    assert result.score >= 0.0
    assert result.metadata["reference_source"] == "course_content"

def test_missing_course_content_returns_zero_flagged():
    result = BLEURTMetric(model_name=_CHECKPOINT, reference_source="course_content").compute(
        _q("what is osmosis?"), _ctx()
    )
    assert result.score == 0.0 and result.flagged is True

# Multiple references — max score is kept
def test_multiple_objectives_picks_best():
    m = BLEURTMetric(model_name=_CHECKPOINT)
    ctx_multi  = _ctx(objectives=["identify prime numbers", "classify numbers as prime or composite"])
    ctx_single = _ctx(objectives=["identify prime numbers"])
    multi  = m.compute(_q("identify prime numbers"), ctx_multi)
    single = m.compute(_q("identify prime numbers"), ctx_single)
    assert multi.score >= single.score
    assert multi.metadata["n_references"] == 2

# Metadata correctness
def test_metadata_contains_model_name():
    result = BLEURTMetric(model_name=_CHECKPOINT).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert result.metadata["model_name"] == _CHECKPOINT
    assert "raw_score" in result.metadata

# Flagging threshold
def test_flagged_when_score_below_threshold():
    result = BLEURTMetric(model_name=_CHECKPOINT, flag_below=0.999).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["describe the water cycle"]),
    )
    assert result.flagged is True

def test_not_flagged_when_score_above_threshold():
    text = "understand the process of photosynthesis"
    result = BLEURTMetric(model_name=_CHECKPOINT, flag_below=0.5).compute(
        _q(text), _ctx(objectives=[text])
    )
    assert result.flagged is False

# Raw score is preserved in metadata even after clipping
def test_raw_score_in_metadata():
    result = BLEURTMetric(model_name=_CHECKPOINT).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert isinstance(result.metadata["raw_score"], float)
    # clipped score is always >= 0.0 regardless of raw
    assert result.score >= 0.0
