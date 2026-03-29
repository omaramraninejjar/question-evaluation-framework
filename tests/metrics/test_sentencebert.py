"""
Tests for SentenceBERTMetric (src/metrics/sentencebert.py).

Requires:
    pip install sentence-transformers
    # all-MiniLM-L6-v2 (~80 MB) is downloaded on first run and cached
    # in ~/.cache/huggingface.

Score behaviour (all-MiniLM-L6-v2, cosine similarity in [0, 1]):
    - Semantically identical texts   → close to 1.0
    - Related texts                  → typically 0.5–0.9
    - Unrelated texts                → typically < 0.4
    Use relative ordering rather than hard absolute thresholds where possible.
"""

from __future__ import annotations

import pytest

from src.models import Question, EvaluationContext
from src.metrics.sentencebert import SentenceBERTMetric, _SBERT_AVAILABLE

if not _SBERT_AVAILABLE:
    pytest.skip(
        "Sentence-BERT unavailable — install: pip install sentence-transformers",
        allow_module_level=True,
    )

_MODEL = "all-MiniLM-L6-v2"


def _q(text: str) -> Question:
    return Question(id="q1", text=text)

def _ctx(objectives: list[str] | None = None, course_content: str | None = None) -> EvaluationContext:
    return EvaluationContext(
        learning_objectives=objectives or [],
        course_content=course_content,
    )


# No references / empty input
def test_no_references_returns_zero_flagged():
    result = SentenceBERTMetric(model_name=_MODEL).compute(_q("What is osmosis?"), _ctx())
    assert result.score == 0.0 and result.flagged is True

def test_empty_candidate_returns_zero_flagged():
    result = SentenceBERTMetric(model_name=_MODEL).compute(
        _q(""), _ctx(objectives=["understand osmosis"])
    )
    assert result.score == 0.0 and result.flagged is True


# Core behaviour
def test_identical_texts_score_near_one():
    result = SentenceBERTMetric(model_name=_MODEL).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify prime numbers"]),
    )
    assert result.score >= 0.99

def test_score_in_unit_interval():
    result = SentenceBERTMetric(model_name=_MODEL).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert 0.0 <= result.score <= 1.0

def test_related_scores_higher_than_unrelated():
    m = SentenceBERTMetric(model_name=_MODEL)
    ctx = _ctx(objectives=["identify prime numbers"])
    related   = m.compute(_q("what are prime numbers and how to find them"), ctx)
    unrelated = m.compute(_q("describe the water cycle and evaporation"), ctx)
    assert related.score > unrelated.score


# Concept Coverage use case
def test_course_content_reference_source():
    result = SentenceBERTMetric(model_name=_MODEL, reference_source="course_content").compute(
        _q("what is photosynthesis"),
        _ctx(course_content="photosynthesis converts light energy into chemical energy in plants"),
    )
    assert 0.0 <= result.score <= 1.0
    assert result.metadata["reference_source"] == "course_content"

def test_missing_course_content_returns_zero_flagged():
    result = SentenceBERTMetric(model_name=_MODEL, reference_source="course_content").compute(
        _q("what is osmosis?"), _ctx()
    )
    assert result.score == 0.0 and result.flagged is True


# Multiple references — max score is kept
def test_multiple_objectives_picks_best():
    m = SentenceBERTMetric(model_name=_MODEL)
    ctx_multi  = _ctx(objectives=["identify prime numbers", "classify numbers as prime or composite"])
    ctx_single = _ctx(objectives=["identify prime numbers"])
    multi  = m.compute(_q("identify prime numbers"), ctx_multi)
    single = m.compute(_q("identify prime numbers"), ctx_single)
    assert multi.score >= single.score - 1e-6
    assert multi.metadata["n_references"] == 2


# Metadata correctness
def test_metadata_contains_model_name():
    result = SentenceBERTMetric(model_name=_MODEL).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert result.metadata["model_name"] == _MODEL
    assert "raw_score" in result.metadata

def test_raw_score_in_metadata():
    result = SentenceBERTMetric(model_name=_MODEL).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert isinstance(result.metadata["raw_score"], float)
    assert result.score >= 0.0


# Flagging threshold
def test_flagged_when_score_below_threshold():
    result = SentenceBERTMetric(model_name=_MODEL, flag_below=0.999).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["describe the water cycle"]),
    )
    assert result.flagged is True

def test_not_flagged_when_score_above_threshold():
    result = SentenceBERTMetric(model_name=_MODEL, flag_below=0.0).compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify prime numbers"]),
    )
    assert result.flagged is False
