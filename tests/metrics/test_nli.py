"""
Tests for NLIEntailmentMetric (src/metrics/nli.py).

Requires:
    pip install sentence-transformers
    # cross-encoder/nli-deberta-v3-small (~180 MB) is downloaded on first run
    # and cached in ~/.cache/huggingface.

Score behaviour (entailment probability in [0, 1]):
    - Strongly aligned question–objective pairs → typically 0.5–0.9
    - Weakly related pairs                      → typically 0.2–0.5
    - Contradictory / unrelated pairs           → typically < 0.3
    Use relative ordering rather than hard absolute thresholds where possible.
"""

from __future__ import annotations

import pytest

from src.models import Question, EvaluationContext
from src.metrics.nli import NLIEntailmentMetric, _NLI_AVAILABLE

if not _NLI_AVAILABLE:
    pytest.skip(
        "NLI unavailable — install: pip install sentence-transformers",
        allow_module_level=True,
    )

_MODEL = "cross-encoder/nli-deberta-v3-small"


def _q(text: str) -> Question:
    return Question(id="q1", text=text)

def _ctx(objectives: list[str] | None = None, course_content: str | None = None) -> EvaluationContext:
    return EvaluationContext(
        learning_objectives=objectives or [],
        course_content=course_content,
    )


# No references / empty input
def test_no_references_returns_zero_flagged():
    result = NLIEntailmentMetric(model_name=_MODEL).compute(_q("What is osmosis?"), _ctx())
    assert result.score == 0.0 and result.flagged is True

def test_empty_candidate_returns_zero_flagged():
    result = NLIEntailmentMetric(model_name=_MODEL).compute(
        _q(""), _ctx(objectives=["understand osmosis"])
    )
    assert result.score == 0.0 and result.flagged is True


# Core behaviour
def test_score_in_unit_interval():
    result = NLIEntailmentMetric(model_name=_MODEL).compute(
        _q("What is a prime number?"),
        _ctx(objectives=["Students can identify and define prime numbers."]),
    )
    assert 0.0 <= result.score <= 1.0

def test_aligned_scores_higher_than_unrelated():
    m = NLIEntailmentMetric(model_name=_MODEL)
    objective = ["Students can identify prime numbers."]
    aligned   = m.compute(_q("What is a prime number?"), _ctx(objectives=objective))
    unrelated = m.compute(_q("Describe the water cycle and evaporation."), _ctx(objectives=objective))
    assert aligned.score > unrelated.score


# Concept Coverage use case
def test_course_content_reference_source():
    result = NLIEntailmentMetric(model_name=_MODEL, reference_source="course_content").compute(
        _q("What does photosynthesis produce?"),
        _ctx(course_content="Photosynthesis converts light energy into chemical energy stored in glucose."),
    )
    assert 0.0 <= result.score <= 1.0
    assert result.metadata["reference_source"] == "course_content"

def test_missing_course_content_returns_zero_flagged():
    result = NLIEntailmentMetric(model_name=_MODEL, reference_source="course_content").compute(
        _q("what is osmosis?"), _ctx()
    )
    assert result.score == 0.0 and result.flagged is True


# Multiple references — max score is kept
def test_multiple_objectives_picks_best():
    m = NLIEntailmentMetric(model_name=_MODEL)
    ctx_multi  = _ctx(objectives=[
        "Students can identify prime numbers.",
        "Students can classify numbers as prime or composite.",
    ])
    ctx_single = _ctx(objectives=["Students can identify prime numbers."])
    multi  = m.compute(_q("What is a prime number?"), ctx_multi)
    single = m.compute(_q("What is a prime number?"), ctx_single)
    assert multi.score >= single.score - 1e-6
    assert multi.metadata["n_references"] == 2


# Metadata correctness
def test_metadata_contains_model_name():
    result = NLIEntailmentMetric(model_name=_MODEL).compute(
        _q("What is a prime number?"),
        _ctx(objectives=["Students can identify prime numbers."]),
    )
    assert result.metadata["model_name"] == _MODEL
    assert result.metadata["entailment_idx"] == 1

def test_metadata_n_references():
    result = NLIEntailmentMetric(model_name=_MODEL).compute(
        _q("What is a prime number?"),
        _ctx(objectives=["identify primes", "define prime numbers"]),
    )
    assert result.metadata["n_references"] == 2


# Flagging threshold
def test_flagged_when_score_below_threshold():
    result = NLIEntailmentMetric(model_name=_MODEL, flag_below=0.999).compute(
        _q("What is a prime number?"),
        _ctx(objectives=["Describe the water cycle."]),
    )
    assert result.flagged is True

def test_not_flagged_when_score_above_threshold():
    result = NLIEntailmentMetric(model_name=_MODEL, flag_below=0.0).compute(
        _q("What is a prime number?"),
        _ctx(objectives=["Students can identify prime numbers."]),
    )
    assert result.flagged is False
