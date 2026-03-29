"""
Tests for chrFMetric (src/metrics/chrf.py).

Requires:
    pip install sacrebleu

Score behaviour (chrF, char_order=6):
    - Identical texts               → 1.0
    - Closely related texts         → typically 0.4–0.8
    - Unrelated texts               → typically < 0.3
    Use relative ordering rather than hard absolute thresholds where possible.
"""

from __future__ import annotations

import pytest

from src.models import Question, EvaluationContext
from src.metrics.chrf import chrFMetric, _CHRF_AVAILABLE

if not _CHRF_AVAILABLE:
    pytest.skip(
        "chrF unavailable — install: pip install sacrebleu",
        allow_module_level=True,
    )


def _q(text: str) -> Question:
    return Question(id="q1", text=text)

def _ctx(objectives: list[str] | None = None, course_content: str | None = None) -> EvaluationContext:
    return EvaluationContext(
        learning_objectives=objectives or [],
        course_content=course_content,
    )


# No references / empty input
def test_no_references_returns_zero_flagged():
    result = chrFMetric().compute(_q("What is osmosis?"), _ctx())
    assert result.score == 0.0 and result.flagged is True

def test_empty_candidate_returns_zero_flagged():
    result = chrFMetric().compute(_q(""), _ctx(objectives=["understand osmosis"]))
    assert result.score == 0.0 and result.flagged is True


# Core behaviour
def test_identical_texts_score_one():
    result = chrFMetric().compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify prime numbers"]),
    )
    assert result.score == pytest.approx(1.0)

def test_score_in_unit_interval():
    result = chrFMetric().compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert 0.0 <= result.score <= 1.0

def test_related_scores_higher_than_unrelated():
    m = chrFMetric()
    ctx = _ctx(objectives=["identify prime numbers"])
    related   = m.compute(_q("identify prime numbers in a list"), ctx)
    unrelated = m.compute(_q("describe the water cycle and evaporation"), ctx)
    assert related.score > unrelated.score


# chrF++ mode (word_order=2)
def test_chrfpp_mode_score_in_unit_interval():
    result = chrFMetric(word_order=2).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert 0.0 <= result.score <= 1.0
    assert result.metadata["mode"] == "chrF++"

def test_chrf_mode_metadata():
    result = chrFMetric(word_order=0).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert result.metadata["mode"] == "chrF"


# Concept Coverage use case
def test_course_content_reference_source():
    result = chrFMetric(reference_source="course_content").compute(
        _q("what is photosynthesis"),
        _ctx(course_content="photosynthesis converts light to energy"),
    )
    assert 0.0 <= result.score <= 1.0
    assert result.metadata["reference_source"] == "course_content"

def test_missing_course_content_returns_zero_flagged():
    result = chrFMetric(reference_source="course_content").compute(
        _q("what is osmosis?"), _ctx()
    )
    assert result.score == 0.0 and result.flagged is True


# Multiple references — max score is kept
def test_multiple_objectives_picks_best():
    m = chrFMetric()
    ctx_multi  = _ctx(objectives=["identify prime numbers", "classify numbers as prime or composite"])
    ctx_single = _ctx(objectives=["identify prime numbers"])
    multi  = m.compute(_q("identify prime numbers"), ctx_multi)
    single = m.compute(_q("identify prime numbers"), ctx_single)
    assert multi.score >= single.score - 1e-6
    assert multi.metadata["n_references"] == 2


# Flagging threshold
def test_flagged_when_score_below_threshold():
    result = chrFMetric(flag_below=0.999).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["describe the water cycle"]),
    )
    assert result.flagged is True

def test_not_flagged_when_score_above_threshold():
    result = chrFMetric(flag_below=0.0).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify prime numbers"]),
    )
    assert result.flagged is False


# Metadata correctness
def test_metadata_fields():
    result = chrFMetric(char_order=6, word_order=0).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert result.metadata["char_order"] == 6
    assert result.metadata["word_order"] == 0
    assert result.metadata["n_references"] == 1
