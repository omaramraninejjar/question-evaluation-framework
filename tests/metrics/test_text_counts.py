"""
Tests for count-based readability metrics (src/metrics/text_counts.py).

Covers: WordCountMetric, SentenceCountMetric, AvgWordLengthMetric,
        AvgSyllablesMetric, LongWordRatioMetric, TypeTokenRatioMetric.

No external dependencies — all metrics are always available.

Score convention:
    Higher score → better (more accessible / more diverse vocabulary).
    Scores are always in [0.0, 1.0].
"""

from __future__ import annotations

import pytest

from src.models import Question, EvaluationContext
from src.metrics.text_counts import (
    WordCountMetric,
    SentenceCountMetric,
    AvgWordLengthMetric,
    AvgSyllablesMetric,
    LongWordRatioMetric,
    TypeTokenRatioMetric,
)

_CTX = EvaluationContext(learning_objectives=[], course_content=None)


def _q(text: str) -> Question:
    return Question(id="q1", text=text)


_ALL_METRICS = [
    WordCountMetric,
    SentenceCountMetric,
    AvgWordLengthMetric,
    AvgSyllablesMetric,
    LongWordRatioMetric,
    TypeTokenRatioMetric,
]


# ---------------------------------------------------------------------------
# Shared contract tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("MetricClass", _ALL_METRICS)
def test_score_in_unit_interval(MetricClass):
    result = MetricClass().compute(_q("What is the function of mitochondria?"), _CTX)
    assert 0.0 <= result.score <= 1.0

@pytest.mark.parametrize("MetricClass", _ALL_METRICS)
def test_empty_text_returns_zero_flagged(MetricClass):
    result = MetricClass().compute(_q(""), _CTX)
    assert result.score == 0.0
    assert result.flagged is True

@pytest.mark.parametrize("MetricClass", _ALL_METRICS)
def test_result_has_required_fields(MetricClass):
    result = MetricClass().compute(_q("What is photosynthesis?"), _CTX)
    assert result.metric_name == MetricClass().name
    assert isinstance(result.rationale, str) and result.rationale
    assert isinstance(result.metadata, dict)

@pytest.mark.parametrize("MetricClass", _ALL_METRICS)
def test_context_is_ignored(MetricClass):
    ctx_full = EvaluationContext(
        learning_objectives=["understand photosynthesis"],
        course_content="Photosynthesis converts light energy.",
    )
    m = MetricClass()
    r1 = m.compute(_q("What is photosynthesis?"), _CTX)
    r2 = m.compute(_q("What is photosynthesis?"), ctx_full)
    assert r1.score == r2.score

@pytest.mark.parametrize("MetricClass", _ALL_METRICS)
def test_metric_names_are_non_empty_strings(MetricClass):
    assert isinstance(MetricClass().name, str) and MetricClass().name


# ---------------------------------------------------------------------------
# WordCountMetric
# ---------------------------------------------------------------------------

def test_word_count_single_word():
    result = WordCountMetric(max_words=50).compute(_q("photosynthesis"), _CTX)
    assert result.metadata["word_count"] == 1
    assert result.score == pytest.approx(1 / 50)

def test_word_count_saturates_at_one():
    long_text = " ".join(["word"] * 60)
    result = WordCountMetric(max_words=50).compute(_q(long_text), _CTX)
    assert result.score == 1.0

def test_word_count_more_words_higher_score():
    short = WordCountMetric().compute(_q("What?"), _CTX)
    long  = WordCountMetric().compute(_q("What is the main function of the mitochondria?"), _CTX)
    assert long.score > short.score


# ---------------------------------------------------------------------------
# SentenceCountMetric
# ---------------------------------------------------------------------------

def test_sentence_count_single_sentence():
    result = SentenceCountMetric(max_sentences=5).compute(
        _q("What is photosynthesis?"), _CTX
    )
    assert result.metadata["sentence_count"] == 1
    assert result.score == pytest.approx(1 / 5)

def test_sentence_count_multiple_sentences():
    result = SentenceCountMetric(max_sentences=5).compute(
        _q("Describe osmosis. What drives it? Give an example."), _CTX
    )
    assert result.metadata["sentence_count"] == 3
    assert result.score == pytest.approx(3 / 5)

def test_sentence_count_saturates_at_one():
    result = SentenceCountMetric(max_sentences=2).compute(
        _q("First. Second. Third."), _CTX
    )
    assert result.score == 1.0


# ---------------------------------------------------------------------------
# AvgWordLengthMetric
# ---------------------------------------------------------------------------

def test_avg_word_length_short_words_higher_score():
    short_words = AvgWordLengthMetric().compute(_q("I am a dog"), _CTX)
    long_words  = AvgWordLengthMetric().compute(
        _q("thermodynamic equilibrium crystallography"), _CTX
    )
    assert short_words.score > long_words.score

def test_avg_word_length_metadata():
    result = AvgWordLengthMetric().compute(_q("cat"), _CTX)
    assert result.metadata["avg_word_length_chars"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# AvgSyllablesMetric
# ---------------------------------------------------------------------------

def test_avg_syllables_monosyllabic_higher_score():
    mono = AvgSyllablesMetric().compute(_q("what is a cat"), _CTX)
    poly = AvgSyllablesMetric().compute(_q("thermodynamic crystallography"), _CTX)
    assert mono.score > poly.score

def test_avg_syllables_score_in_unit_interval():
    result = AvgSyllablesMetric().compute(_q("What is photosynthesis?"), _CTX)
    assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# LongWordRatioMetric
# ---------------------------------------------------------------------------

def test_long_word_ratio_all_short_words():
    """All words ≤ 6 chars → ratio = 0.0 → score = 1.0."""
    result = LongWordRatioMetric(long_threshold=6).compute(_q("what is a cat"), _CTX)
    assert result.metadata["long_word_ratio"] == 0.0
    assert result.score == 1.0

def test_long_word_ratio_all_long_words():
    """All words > 6 chars → ratio = 1.0 → score = 0.0."""
    result = LongWordRatioMetric(long_threshold=6).compute(
        _q("thermodynamic crystallography equilibrium"), _CTX
    )
    assert result.metadata["long_word_ratio"] == 1.0
    assert result.score == 0.0

def test_long_word_ratio_fewer_long_words_higher_score():
    few_long = LongWordRatioMetric().compute(_q("what is a dog"), _CTX)
    many_long = LongWordRatioMetric().compute(
        _q("thermodynamic crystallography"), _CTX
    )
    assert few_long.score >= many_long.score


# ---------------------------------------------------------------------------
# TypeTokenRatioMetric
# ---------------------------------------------------------------------------

def test_ttr_all_unique_words():
    """All unique words → TTR = 1.0."""
    result = TypeTokenRatioMetric().compute(_q("cat sat on mat"), _CTX)
    assert result.score == pytest.approx(1.0)

def test_ttr_repeated_words():
    """All same word → TTR = 1/N < 1."""
    result = TypeTokenRatioMetric().compute(_q("cat cat cat cat"), _CTX)
    assert result.score == pytest.approx(0.25)

def test_ttr_score_in_unit_interval():
    result = TypeTokenRatioMetric().compute(
        _q("What is the role of the mitochondria in the cell?"), _CTX
    )
    assert 0.0 <= result.score <= 1.0

def test_ttr_more_repetition_lower_score():
    varied   = TypeTokenRatioMetric().compute(_q("cat sat on hat mat fat"), _CTX)
    repeated = TypeTokenRatioMetric().compute(_q("cat cat cat cat cat cat"), _CTX)
    assert varied.score > repeated.score


# ---------------------------------------------------------------------------
# Name uniqueness
# ---------------------------------------------------------------------------

def test_all_metric_names_distinct():
    names = [m().name for m in _ALL_METRICS]
    assert len(names) == len(set(names)), f"Duplicate names: {names}"
