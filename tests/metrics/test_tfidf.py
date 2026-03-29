"""
Tests for TFIDFMetric (src/metrics/tfidf.py).

Requires:
    pip install scikit-learn

Score behaviour (TF-IDF cosine, word unigrams + bigrams):
    - Identical texts               → 1.0
    - Closely related texts         → typically 0.3–0.8
    - Completely unrelated texts    → typically < 0.2
    Use relative ordering rather than hard absolute thresholds where possible.
"""

from __future__ import annotations

import pytest

from src.models import Question, EvaluationContext
from src.metrics.tfidf import TFIDFMetric, _TFIDF_AVAILABLE

if not _TFIDF_AVAILABLE:
    pytest.skip(
        "TF-IDF unavailable — install: pip install scikit-learn",
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
    result = TFIDFMetric().compute(_q("What is osmosis?"), _ctx())
    assert result.score == 0.0 and result.flagged is True

def test_empty_candidate_returns_zero_flagged():
    result = TFIDFMetric().compute(_q(""), _ctx(objectives=["understand osmosis"]))
    assert result.score == 0.0 and result.flagged is True


# Core behaviour
def test_identical_texts_score_one():
    result = TFIDFMetric().compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify prime numbers"]),
    )
    assert result.score == pytest.approx(1.0)

def test_score_in_unit_interval():
    result = TFIDFMetric().compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes and composite numbers"]),
    )
    assert 0.0 <= result.score <= 1.0

def test_related_scores_higher_than_unrelated():
    m = TFIDFMetric()
    ctx = _ctx(objectives=["identify prime numbers"])
    related   = m.compute(_q("what are prime numbers"), ctx)
    unrelated = m.compute(_q("describe the water cycle and evaporation"), ctx)
    assert related.score > unrelated.score


# analyzer variants
def test_char_wb_analyzer():
    result = TFIDFMetric(analyzer="char_wb").compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert 0.0 <= result.score <= 1.0
    assert result.metadata["analyzer"] == "char_wb"


# Concept Coverage use case
def test_course_content_reference_source():
    result = TFIDFMetric(reference_source="course_content").compute(
        _q("what is photosynthesis"),
        _ctx(course_content="photosynthesis converts light to energy in plant cells"),
    )
    assert 0.0 <= result.score <= 1.0
    assert result.metadata["reference_source"] == "course_content"

def test_missing_course_content_returns_zero_flagged():
    result = TFIDFMetric(reference_source="course_content").compute(
        _q("what is osmosis?"), _ctx()
    )
    assert result.score == 0.0 and result.flagged is True


# Multiple references — max score is kept
def test_multiple_objectives_picks_best():
    m = TFIDFMetric()
    ctx_multi  = _ctx(objectives=["identify prime numbers", "classify numbers as prime or composite"])
    ctx_single = _ctx(objectives=["identify prime numbers"])
    multi  = m.compute(_q("identify prime numbers"), ctx_multi)
    single = m.compute(_q("identify prime numbers"), ctx_single)
    assert multi.score >= single.score - 1e-6
    assert multi.metadata["n_references"] == 2


# Flagging threshold
def test_flagged_when_score_below_threshold():
    result = TFIDFMetric(flag_below=0.999).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["describe the water cycle"]),
    )
    assert result.flagged is True

def test_not_flagged_when_score_above_threshold():
    result = TFIDFMetric(flag_below=0.0).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify prime numbers"]),
    )
    assert result.flagged is False


# Metadata correctness
def test_metadata_fields():
    result = TFIDFMetric(ngram_range=(1, 1)).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert result.metadata["ngram_range"] == [1, 1]
    assert result.metadata["n_references"] == 1
    assert result.metadata["analyzer"] == "word"
