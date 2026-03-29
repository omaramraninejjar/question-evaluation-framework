"""
Tests for ReadabilityDimension (src/dimensions/linguistic_structural/readability.py).

Structure tests (fast — no external model loading):
    Verify dimension identity, metric presence, and that metrics score
    question.text without needing context.

Core scoring tests:
    Count metrics are always loaded; formula metrics load when textstat is
    installed.  Tests are guarded by _TEXTSTAT_AVAILABLE.
"""

from __future__ import annotations

import pytest

from src.models import Question, EvaluationContext, DimensionName
from src.dimensions.linguistic_structural.readability import ReadabilityDimension
from src.metrics.flesch_ease import _TEXTSTAT_AVAILABLE

_COUNT_METRICS = {
    "word_count",
    "sentence_count",
    "avg_word_length",
    "avg_syllables",
    "long_word_ratio",
    "type_token_ratio",
}

_FORMULA_METRICS = {
    "flesch_ease",
    "flesch_kincaid",
    "dale_chall",
    "gunning_fog",
    "coleman_liau",
    "ari",
    "smog",
    "linsear_write",
}


def _q(text: str) -> Question:
    return Question(id="q1", text=text)


def _ctx() -> EvaluationContext:
    return EvaluationContext(learning_objectives=[], course_content=None)


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

def test_dimension_name():
    assert ReadabilityDimension().name == DimensionName.READABILITY

def test_count_metrics_always_present():
    names = {m.name for m in ReadabilityDimension().metrics}
    assert _COUNT_METRICS.issubset(names), (
        f"Missing count metrics: {_COUNT_METRICS - names}"
    )

def test_formula_metrics_present_when_textstat_available():
    names = {m.name for m in ReadabilityDimension().metrics}
    if _TEXTSTAT_AVAILABLE:
        assert _FORMULA_METRICS.issubset(names), (
            f"Missing formula metrics: {_FORMULA_METRICS - names}"
        )

def test_formula_metrics_absent_without_textstat():
    """This test is informational — it only runs when textstat is absent."""
    if _TEXTSTAT_AVAILABLE:
        pytest.skip("textstat is installed — formula metrics are present")
    names = {m.name for m in ReadabilityDimension().metrics}
    assert _FORMULA_METRICS.isdisjoint(names), (
        f"Formula metrics should be absent without textstat: {_FORMULA_METRICS & names}"
    )

def test_each_instantiation_gets_its_own_metrics_list():
    d1 = ReadabilityDimension()
    d2 = ReadabilityDimension()
    assert d1.metrics is not d2.metrics

def test_repr_includes_class_and_count_metric_names():
    d = ReadabilityDimension()
    r = repr(d)
    assert "ReadabilityDimension" in r
    for name in _COUNT_METRICS:
        assert name in r


# ---------------------------------------------------------------------------
# Edge-case scoring
# ---------------------------------------------------------------------------

def test_empty_question_all_metrics_zero_flagged():
    d = ReadabilityDimension()
    result = d.score(_q(""), _ctx())
    for name, mr in result.scores.items():
        assert mr.score == 0.0, f"{name}: expected score=0.0 on empty text"
        assert mr.flagged is True, f"{name}: expected flagged=True on empty text"

def test_dimension_result_keys_match_loaded_metrics():
    d = ReadabilityDimension()
    result = d.score(_q("What is photosynthesis?"), _ctx())
    assert set(result.scores.keys()) == {m.name for m in d.metrics}

def test_context_does_not_change_scores():
    """Readability scores must be independent of context."""
    d = ReadabilityDimension()
    ctx_empty = _ctx()
    ctx_full  = EvaluationContext(
        learning_objectives=["understand photosynthesis"],
        course_content="Photosynthesis converts light energy.",
    )
    q = _q("What is the role of chlorophyll in photosynthesis?")
    r_empty = d.score(q, ctx_empty)
    r_full  = d.score(q, ctx_full)
    for name in r_empty.scores:
        assert r_empty.scores[name].score == r_full.scores[name].score, (
            f"{name}: score changed with context"
        )


# ---------------------------------------------------------------------------
# Core scoring (count metrics)
# ---------------------------------------------------------------------------

def test_count_metrics_return_nonzero_on_real_question():
    d = ReadabilityDimension()
    result = d.score(_q("What is the main function of mitochondria?"), _ctx())
    for name in _COUNT_METRICS:
        assert result.scores[name].score > 0.0, f"{name}: expected score > 0.0"

def test_longer_question_higher_word_count_score():
    d = ReadabilityDimension()
    short_r = d.score(_q("What?"), _ctx())
    long_r  = d.score(
        _q("What is the primary role of mitochondria in eukaryotic cells?"), _ctx()
    )
    assert long_r.scores["word_count"].score > short_r.scores["word_count"].score

def test_repeated_words_lower_ttr():
    d = ReadabilityDimension()
    varied   = d.score(_q("cats dogs birds fish frogs"), _ctx())
    repeated = d.score(_q("cat cat cat cat cat"), _ctx())
    assert varied.scores["type_token_ratio"].score > repeated.scores["type_token_ratio"].score


# ---------------------------------------------------------------------------
# Formula metric scoring (guarded)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _TEXTSTAT_AVAILABLE, reason="textstat not installed")
def test_formula_metrics_return_nonzero_on_real_question():
    d = ReadabilityDimension()
    result = d.score(_q("What is the main function of mitochondria?"), _ctx())
    for name in _FORMULA_METRICS:
        assert result.scores[name].score >= 0.0, f"{name}: score should be >= 0"

@pytest.mark.skipif(not _TEXTSTAT_AVAILABLE, reason="textstat not installed")
def test_simple_text_accessible_on_flesch_ease():
    d = ReadabilityDimension()
    result = d.score(_q("What is a cat?"), _ctx())
    assert result.scores["flesch_ease"].score > 0.0
