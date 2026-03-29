"""
Tests for COMETMetric (src/metrics/comet.py).

Requires:
    pip install unbabel-comet
    # Unbabel/wmt22-comet-da (~1.9 GB) is downloaded on first run and cached
    # in ~/.cache/huggingface.

Score behaviour (wmt22-comet-da, approximately in [0, 1]):
    - Semantically aligned texts  → typically 0.7–1.0
    - Loosely related texts        → typically 0.4–0.7
    - Unrelated texts              → typically < 0.5
  Use relative ordering rather than hard absolute thresholds where possible.
"""

from __future__ import annotations

import pytest

from src.models import Question, EvaluationContext
from src.metrics.comet import COMETMetric, _COMET_AVAILABLE

if not _COMET_AVAILABLE:
    pytest.skip(
        "COMET unavailable — install: pip install unbabel-comet",
        allow_module_level=True,
    )

_MODEL = "Unbabel/wmt22-comet-da"


def _q(text: str) -> Question:
    return Question(id="q1", text=text)

def _ctx(objectives: list[str] | None = None, course_content: str | None = None) -> EvaluationContext:
    return EvaluationContext(
        learning_objectives=objectives or [],
        course_content=course_content,
    )


# No references / empty input
def test_no_references_returns_zero_flagged():
    result = COMETMetric(model_name=_MODEL).compute(_q("What is osmosis?"), _ctx())
    assert result.score == 0.0 and result.flagged is True

def test_empty_candidate_returns_zero_flagged():
    result = COMETMetric(model_name=_MODEL).compute(_q(""), _ctx(objectives=["understand osmosis"]))
    assert result.score == 0.0 and result.flagged is True

# Core behaviour
def test_score_in_unit_interval():
    result = COMETMetric(model_name=_MODEL).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert 0.0 <= result.score <= 1.0

def test_related_scores_higher_than_unrelated():
    m = COMETMetric(model_name=_MODEL)
    ctx = _ctx(objectives=["identify prime numbers"])
    related   = m.compute(_q("identify prime numbers in a list"), ctx)
    unrelated = m.compute(_q("describe the water cycle and evaporation"), ctx)
    assert related.score > unrelated.score

# Concept Coverage use case
def test_course_content_reference_source():
    result = COMETMetric(model_name=_MODEL, reference_source="course_content").compute(
        _q("what is photosynthesis"),
        _ctx(course_content="photosynthesis converts light to energy"),
    )
    assert 0.0 <= result.score <= 1.0
    assert result.metadata["reference_source"] == "course_content"

def test_missing_course_content_returns_zero_flagged():
    result = COMETMetric(model_name=_MODEL, reference_source="course_content").compute(
        _q("what is osmosis?"), _ctx()
    )
    assert result.score == 0.0 and result.flagged is True

# Multiple references — max score is kept
def test_multiple_objectives_picks_best():
    m = COMETMetric(model_name=_MODEL)
    ctx_multi  = _ctx(objectives=["identify prime numbers", "classify numbers as prime or composite"])
    ctx_single = _ctx(objectives=["identify prime numbers"])
    multi  = m.compute(_q("identify prime numbers"), ctx_multi)
    single = m.compute(_q("identify prime numbers"), ctx_single)
    assert multi.score >= single.score - 1e-6   # tolerance for float rounding
    assert multi.metadata["n_references"] == 2

# Metadata correctness
def test_metadata_contains_model_name():
    result = COMETMetric(model_name=_MODEL).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert result.metadata["model_name"] == _MODEL
    assert "raw_score" in result.metadata

# Flagging threshold
def test_flagged_when_score_below_threshold():
    result = COMETMetric(model_name=_MODEL, flag_below=0.999).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["describe the water cycle"]),
    )
    assert result.flagged is True

def test_not_flagged_when_score_above_threshold():
    result = COMETMetric(model_name=_MODEL, flag_below=0.0).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify prime numbers"]),
    )
    assert result.flagged is False

# Raw score preserved in metadata after clipping
def test_raw_score_in_metadata():
    result = COMETMetric(model_name=_MODEL).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert isinstance(result.metadata["raw_score"], float)
    assert result.score >= 0.0
