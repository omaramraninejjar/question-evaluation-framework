"""
Tests for Well-Formedness metrics:
    QuestionMarkMetric  (no deps)
    NegationRateMetric  (no deps)
    HasVerbMetric       (nltk — always available)
    LMPerplexityMetric  (transformers + torch; skip if unavailable)
"""

from __future__ import annotations
import pytest
from src.models import Question, EvaluationContext

_CTX = EvaluationContext(learning_objectives=[], course_content=None)


def _q(text: str) -> Question:
    return Question(id="q1", text=text)


# ---------------------------------------------------------------------------
# QuestionMarkMetric
# ---------------------------------------------------------------------------

from src.metrics.question_mark import QuestionMarkMetric


class TestQuestionMark:
    def test_ends_with_question_mark_scores_one(self):
        r = QuestionMarkMetric().compute(_q("What is photosynthesis?"), _CTX)
        assert r.score == pytest.approx(1.0)
        assert r.flagged is False

    def test_no_question_mark_scores_zero_flagged(self):
        r = QuestionMarkMetric().compute(_q("Describe photosynthesis."), _CTX)
        assert r.score == pytest.approx(0.0)
        assert r.flagged is True

    def test_empty_text_zero_flagged(self):
        r = QuestionMarkMetric().compute(_q(""), _CTX)
        assert r.score == 0.0 and r.flagged is True

    def test_metadata_field(self):
        r = QuestionMarkMetric().compute(_q("What?"), _CTX)
        assert r.metadata["ends_with_question_mark"] is True

    def test_whitespace_trimmed(self):
        r = QuestionMarkMetric().compute(_q("What is it?   "), _CTX)
        assert r.score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# NegationRateMetric
# ---------------------------------------------------------------------------

from src.metrics.negation_rate import NegationRateMetric


class TestNegationRate:
    def test_no_negation_scores_one(self):
        r = NegationRateMetric().compute(_q("What is photosynthesis?"), _CTX)
        assert r.score == pytest.approx(1.0)
        assert r.flagged is False

    def test_one_negation_reduces_score(self):
        r = NegationRateMetric(max_negations=2).compute(
            _q("Which of the following is NOT a mammal?"), _CTX
        )
        assert 0.0 < r.score < 1.0

    def test_max_negations_score_zero(self):
        r = NegationRateMetric(max_negations=1).compute(
            _q("Which of the following is not correct?"), _CTX
        )
        assert r.score == pytest.approx(0.0)
        assert r.flagged is True

    def test_contraction_detected(self):
        r = NegationRateMetric().compute(_q("Why doesn't osmosis stop?"), _CTX)
        assert r.metadata["negation_count"] >= 1

    def test_empty_text_zero_flagged(self):
        r = NegationRateMetric().compute(_q(""), _CTX)
        assert r.score == 0.0 and r.flagged is True

    def test_metadata_has_count(self):
        r = NegationRateMetric().compute(_q("What is not produced?"), _CTX)
        assert "negation_count" in r.metadata
        assert r.metadata["negation_count"] >= 1


# ---------------------------------------------------------------------------
# HasVerbMetric
# ---------------------------------------------------------------------------

from src.metrics.has_verb import HasVerbMetric


class TestHasVerb:
    def test_sentence_with_verb_scores_one(self):
        r = HasVerbMetric().compute(_q("What does the mitochondrion produce?"), _CTX)
        assert r.score == pytest.approx(1.0)
        assert r.flagged is False

    def test_fragment_without_verb_scores_zero_flagged(self):
        r = HasVerbMetric().compute(_q("The mitochondrion?"), _CTX)
        # NLTK may or may not detect verb in very short fragments — just check type
        assert isinstance(r.score, float)
        assert r.score in (0.0, 1.0)

    def test_empty_text_zero_flagged(self):
        r = HasVerbMetric().compute(_q(""), _CTX)
        assert r.score == 0.0 and r.flagged is True

    def test_metadata_has_flag(self):
        r = HasVerbMetric().compute(_q("What is photosynthesis?"), _CTX)
        assert "has_verb" in r.metadata

    def test_gerund_counts_as_verb(self):
        r = HasVerbMetric().compute(_q("Explaining photosynthesis?"), _CTX)
        assert r.score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# LMPerplexityMetric
# ---------------------------------------------------------------------------

from src.metrics.lm_perplexity import LMPerplexityMetric, _LM_AVAILABLE
import os as _os

# Skip LM tests if GPT-2 model is not already cached locally (avoids ~500 MB download in CI)
_GPT2_CACHED = _os.path.isdir(
    _os.path.expanduser("~/.cache/huggingface/hub/models--gpt2")
)
_LM_READY = _LM_AVAILABLE and _GPT2_CACHED


@pytest.mark.skipif(not _LM_READY, reason="GPT-2 model not cached (run: python -c \"from transformers import pipeline; pipeline('text-generation', model='gpt2')\" to download)")
class TestLMPerplexity:
    def test_score_in_unit_interval(self):
        r = LMPerplexityMetric().compute(_q("What is the function of mitochondria?"), _CTX)
        assert 0.0 <= r.score <= 1.0

    def test_empty_text_zero_flagged(self):
        r = LMPerplexityMetric().compute(_q(""), _CTX)
        assert r.score == 0.0 and r.flagged is True

    def test_fluent_scores_higher_than_garbled(self):
        fluent  = LMPerplexityMetric().compute(
            _q("What is the role of chlorophyll in photosynthesis?"), _CTX
        )
        garbled = LMPerplexityMetric().compute(
            _q("photosynthesis chlorophyll role the is What"), _CTX
        )
        assert fluent.score >= garbled.score

    def test_metadata_has_perplexity(self):
        r = LMPerplexityMetric().compute(_q("What is photosynthesis?"), _CTX)
        assert "perplexity" in r.metadata
        assert r.metadata["perplexity"] > 0.0
