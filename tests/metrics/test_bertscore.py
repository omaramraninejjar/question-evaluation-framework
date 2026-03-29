"""
Tests for BERTScoreMetric (src/metrics/bertscore.py).
Requires:
    pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only
    pip install bert-score

NOTE: These tests load a transformer model on first run (~250 MB for
distilbert-base-uncased). Subsequent runs use the cached model.

BERTScore baselines for English:
    - Identical texts                  → ~1.00
    - Closely related / paraphrased    → ~0.88–0.96
    - Loosely related (same domain)    → ~0.80–0.88
    - Unrelated topics                 → ~0.75–0.83
  Use relative ordering rather than hard absolute thresholds where possible.
"""

from __future__ import annotations

import pytest

from src.models import Question, EvaluationContext
from src.metrics.bertscore import BERTScoreMetric, _BERT_SCORE_AVAILABLE

if not _BERT_SCORE_AVAILABLE:
    pytest.skip(
        "BERTScore unavailable — install: "
        "pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install bert-score",
        allow_module_level=True,
    )


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
        BERTScoreMetric(score_type="fmeasure")

# No references / empty input
def test_no_references_returns_zero_flagged():
    result = BERTScoreMetric().compute(_q("What is osmosis?"), _ctx())
    assert result.score == 0.0 and result.flagged is True

def test_empty_candidate_returns_zero_flagged():
    result = BERTScoreMetric().compute(_q(""), _ctx(objectives=["understand osmosis"]))
    assert result.score == 0.0 and result.flagged is True

# Core behaviour
def test_score_in_unit_interval():
    result = BERTScoreMetric().compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert 0.0 <= result.score <= 1.0

def test_identical_texts_score_near_one():
    text = "understand the process of photosynthesis"
    result = BERTScoreMetric().compute(_q(text), _ctx(objectives=[text]))
    assert result.score >= 0.99

def test_related_scores_higher_than_unrelated():
    # BERTScore absolute values are high even for unrelated text,
    # so test relative ordering rather than absolute thresholds.
    m = BERTScoreMetric()
    ctx = _ctx(objectives=["identify prime numbers"])
    related   = m.compute(_q("identify prime numbers in a list"), ctx)
    unrelated = m.compute(_q("describe the water cycle and evaporation"), ctx)
    assert related.score > unrelated.score

# Score type variants
def test_recall_score_type():
    result = BERTScoreMetric(score_type="recall").compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert 0.0 <= result.score <= 1.0
    assert result.metadata["score_type"] == "recall"

def test_precision_score_type():
    result = BERTScoreMetric(score_type="precision").compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert 0.0 <= result.score <= 1.0
    assert result.metadata["score_type"] == "precision"

# Concept Coverage use case
def test_course_content_reference_source():
    result = BERTScoreMetric(reference_source="course_content").compute(
        _q("what is photosynthesis"),
        _ctx(course_content="photosynthesis converts light to energy"),
    )
    assert result.score > 0.0
    assert result.metadata["reference_source"] == "course_content"

def test_missing_course_content_returns_zero_flagged():
    result = BERTScoreMetric(reference_source="course_content").compute(
        _q("what is osmosis?"), _ctx()
    )
    assert result.score == 0.0 and result.flagged is True

# Multiple references — max score is kept
def test_multiple_objectives_picks_best():
    m = BERTScoreMetric()
    ctx_multi  = _ctx(objectives=["identify prime numbers", "classify numbers as prime or composite"])
    ctx_single = _ctx(objectives=["identify prime numbers"])
    multi  = m.compute(_q("identify prime numbers"), ctx_multi)
    single = m.compute(_q("identify prime numbers"), ctx_single)
    assert multi.score >= single.score
    assert multi.metadata["n_references"] == 2

# Metadata correctness
def test_metadata_contains_model_type():
    m = BERTScoreMetric(model_type="distilbert-base-uncased")
    result = m.compute(
        _q("identify prime numbers"),
        _ctx(objectives=["identify primes"]),
    )
    assert result.metadata["model_type"] == "distilbert-base-uncased"

# Flagging threshold
def test_flagged_when_score_below_threshold():
    # Use a very high threshold to force flagging on a valid score
    text = "identify prime numbers"
    result = BERTScoreMetric(flag_below=0.999).compute(
        _q(text),
        _ctx(objectives=["describe the water cycle"]),
    )
    assert result.flagged is True

def test_not_flagged_when_score_above_threshold():
    text = "understand the process of photosynthesis"
    result = BERTScoreMetric(flag_below=0.5).compute(_q(text), _ctx(objectives=[text]))
    assert result.flagged is False
