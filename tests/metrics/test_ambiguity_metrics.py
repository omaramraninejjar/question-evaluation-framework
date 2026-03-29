"""
Tests for Ambiguity metrics:
    WHWordTypeMetric     (no deps)
    PronounRatioMetric   (nltk — always available)
    PolysemyScoreMetric  (nltk + wordnet corpus)
"""

from __future__ import annotations
import pytest
from src.models import Question, EvaluationContext

_CTX = EvaluationContext(learning_objectives=[], course_content=None)


def _q(text: str) -> Question:
    return Question(id="q1", text=text)


# ---------------------------------------------------------------------------
# WHWordTypeMetric
# ---------------------------------------------------------------------------

from src.metrics.wh_word_type import WHWordTypeMetric


class TestWHWordType:
    def test_score_in_unit_interval(self):
        r = WHWordTypeMetric().compute(_q("What is photosynthesis?"), _CTX)
        assert 0.0 <= r.score <= 1.0

    def test_empty_text_zero_flagged(self):
        r = WHWordTypeMetric().compute(_q(""), _CTX)
        assert r.score == 0.0 and r.flagged is True

    def test_specific_wh_scores_high(self):
        # who/where/when/which → 1.0
        r = WHWordTypeMetric().compute(_q("Who discovered penicillin?"), _CTX)
        assert r.score == pytest.approx(1.0)

    def test_open_ended_wh_scores_lower(self):
        why = WHWordTypeMetric().compute(_q("Why does osmosis occur?"), _CTX)
        who = WHWordTypeMetric().compute(_q("Who discovered penicillin?"), _CTX)
        assert who.score > why.score

    def test_no_wh_word_scores_low(self):
        r = WHWordTypeMetric().compute(_q("Explain photosynthesis."), _CTX)
        assert r.score == pytest.approx(0.2)

    def test_compound_question_lower_than_single(self):
        single   = WHWordTypeMetric().compute(_q("What is osmosis?"), _CTX)
        compound = WHWordTypeMetric().compute(_q("What and why does osmosis occur?"), _CTX)
        assert single.score >= compound.score

    def test_metadata_has_wh_type(self):
        r = WHWordTypeMetric().compute(_q("Where does photosynthesis occur?"), _CTX)
        assert "wh_type" in r.metadata
        assert r.metadata["wh_type"] == "where"

    def test_what_scores_0_7(self):
        r = WHWordTypeMetric().compute(_q("What is photosynthesis?"), _CTX)
        assert r.score == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# PronounRatioMetric
# ---------------------------------------------------------------------------

from src.metrics.pronoun_ratio import PronounRatioMetric


class TestPronounRatio:
    def test_score_in_unit_interval(self):
        r = PronounRatioMetric().compute(_q("What is photosynthesis?"), _CTX)
        assert 0.0 <= r.score <= 1.0

    def test_empty_text_zero_flagged(self):
        r = PronounRatioMetric().compute(_q(""), _CTX)
        assert r.score == 0.0 and r.flagged is True

    def test_no_pronouns_score_one(self):
        r = PronounRatioMetric().compute(
            _q("What is the function of mitochondria?"), _CTX
        )
        assert r.score == pytest.approx(1.0)

    def test_pronoun_heavy_lower_score(self):
        clear    = PronounRatioMetric().compute(
            _q("What does the mitochondrion produce?"), _CTX
        )
        ambig = PronounRatioMetric().compute(_q("What does it produce?"), _CTX)
        assert clear.score >= ambig.score

    def test_metadata_has_ratio(self):
        r = PronounRatioMetric().compute(_q("What is it?"), _CTX)
        assert "pronoun_ratio" in r.metadata
        assert 0.0 <= r.metadata["pronoun_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# PolysemyScoreMetric
# ---------------------------------------------------------------------------

from src.metrics.polysemy import PolysemyScoreMetric, _NLTK_WN_AVAILABLE


@pytest.mark.skipif(not _NLTK_WN_AVAILABLE, reason="nltk wordnet corpus not available")
class TestPolysemyScore:
    def test_score_in_unit_interval(self):
        r = PolysemyScoreMetric().compute(_q("What is photosynthesis?"), _CTX)
        assert 0.0 <= r.score <= 1.0

    def test_empty_text_zero_flagged(self):
        r = PolysemyScoreMetric().compute(_q(""), _CTX)
        assert r.score == 0.0 and r.flagged is True

    def test_specific_technical_term_higher_score(self):
        """'mitochondrion' has fewer senses than 'set' — should score higher."""
        specific = PolysemyScoreMetric().compute(
            _q("What does the mitochondrion produce?"), _CTX
        )
        ambiguous = PolysemyScoreMetric().compute(
            _q("What does the bank run set?"), _CTX
        )
        assert specific.score >= ambiguous.score

    def test_metadata_has_avg_synsets(self):
        r = PolysemyScoreMetric().compute(_q("What is photosynthesis?"), _CTX)
        assert "avg_synsets" in r.metadata
        assert r.metadata["avg_synsets"] >= 0.0
