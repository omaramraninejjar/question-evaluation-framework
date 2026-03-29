"""
Tests for Linguistic Complexity metrics:
    ZipfWordFrequencyMetric  (requires wordfreq)
    ParseTreeDepthMetric     (requires spacy + en_core_web_sm)
    DependencyDistanceMetric (requires spacy + en_core_web_sm)

Each metric is tested independently with its own skip guard.
"""

from __future__ import annotations
import pytest
from src.models import Question, EvaluationContext

_CTX = EvaluationContext(learning_objectives=[], course_content=None)


def _q(text: str) -> Question:
    return Question(id="q1", text=text)


_SIMPLE  = "What is a cat?"
_COMPLEX = (
    "In the context of thermodynamic equilibrium, elucidate the intrinsic "
    "relationship between entropy, enthalpy, and the Gibbs free energy "
    "formulation as it pertains to spontaneous electrochemical reactions."
)


# ---------------------------------------------------------------------------
# ZipfWordFrequencyMetric
# ---------------------------------------------------------------------------

from src.metrics.zipf_frequency import ZipfWordFrequencyMetric, _WORDFREQ_AVAILABLE

@pytest.mark.skipif(not _WORDFREQ_AVAILABLE, reason="wordfreq not installed")
class TestZipfWordFrequency:
    def test_score_in_unit_interval(self):
        r = ZipfWordFrequencyMetric().compute(_q(_SIMPLE), _CTX)
        assert 0.0 <= r.score <= 1.0

    def test_empty_text_zero_flagged(self):
        r = ZipfWordFrequencyMetric().compute(_q(""), _CTX)
        assert r.score == 0.0 and r.flagged is True

    def test_common_words_higher_score(self):
        common = ZipfWordFrequencyMetric().compute(_q("what is a cat"), _CTX)
        rare   = ZipfWordFrequencyMetric().compute(_q(_COMPLEX), _CTX)
        assert common.score >= rare.score

    def test_metadata_has_avg_zipf(self):
        r = ZipfWordFrequencyMetric().compute(_q(_SIMPLE), _CTX)
        assert "avg_zipf" in r.metadata

    def test_context_ignored(self):
        ctx_full = EvaluationContext(
            learning_objectives=["understand entropy"],
            course_content="Thermodynamics and entropy.",
        )
        m = ZipfWordFrequencyMetric()
        assert m.compute(_q(_SIMPLE), _CTX).score == m.compute(_q(_SIMPLE), ctx_full).score


# ---------------------------------------------------------------------------
# ParseTreeDepthMetric
# ---------------------------------------------------------------------------

from src.metrics.parse_tree_depth import ParseTreeDepthMetric, _SPACY_AVAILABLE

@pytest.mark.skipif(not _SPACY_AVAILABLE, reason="spacy not installed")
class TestParseTreeDepth:
    def test_score_in_unit_interval(self):
        r = ParseTreeDepthMetric().compute(_q(_SIMPLE), _CTX)
        assert 0.0 <= r.score <= 1.0

    def test_empty_text_zero_flagged(self):
        r = ParseTreeDepthMetric().compute(_q(""), _CTX)
        assert r.score == 0.0 and r.flagged is True

    def test_simple_question_higher_score(self):
        simple  = ParseTreeDepthMetric().compute(_q(_SIMPLE), _CTX)
        complex_ = ParseTreeDepthMetric().compute(_q(_COMPLEX), _CTX)
        assert simple.score >= complex_.score

    def test_metadata_has_tree_depth(self):
        r = ParseTreeDepthMetric().compute(_q(_SIMPLE), _CTX)
        assert "tree_depth" in r.metadata
        assert isinstance(r.metadata["tree_depth"], int)

    def test_custom_max_depth(self):
        strict  = ParseTreeDepthMetric(max_depth=3).compute(_q(_COMPLEX), _CTX)
        relaxed = ParseTreeDepthMetric(max_depth=20).compute(_q(_COMPLEX), _CTX)
        assert relaxed.score >= strict.score


# ---------------------------------------------------------------------------
# DependencyDistanceMetric
# ---------------------------------------------------------------------------

from src.metrics.dependency_distance import DependencyDistanceMetric

@pytest.mark.skipif(not _SPACY_AVAILABLE, reason="spacy not installed")
class TestDependencyDistance:
    def test_score_in_unit_interval(self):
        r = DependencyDistanceMetric().compute(_q(_SIMPLE), _CTX)
        assert 0.0 <= r.score <= 1.0

    def test_empty_text_zero_flagged(self):
        r = DependencyDistanceMetric().compute(_q(""), _CTX)
        assert r.score == 0.0 and r.flagged is True

    def test_simple_question_higher_or_equal_score(self):
        simple   = DependencyDistanceMetric().compute(_q(_SIMPLE), _CTX)
        complex_ = DependencyDistanceMetric().compute(_q(_COMPLEX), _CTX)
        assert simple.score >= complex_.score

    def test_metadata_has_mdd(self):
        r = DependencyDistanceMetric().compute(_q(_SIMPLE), _CTX)
        assert "mean_dependency_distance" in r.metadata
        assert r.metadata["mean_dependency_distance"] >= 0.0
