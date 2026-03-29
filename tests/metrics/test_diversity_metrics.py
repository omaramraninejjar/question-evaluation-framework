"""
Tests for Diversity metrics:
    DistinctNMetric        (no deps)
    VocabularyNoveltyMetric (no deps — uses context.course_content)
"""

from __future__ import annotations
import pytest
from src.models import Question, EvaluationContext

_CTX      = EvaluationContext(learning_objectives=[], course_content=None)
_CTX_FULL = EvaluationContext(
    learning_objectives=[],
    course_content="Photosynthesis converts light energy into chemical energy stored in glucose.",
)


def _q(text: str) -> Question:
    return Question(id="q1", text=text)


# ---------------------------------------------------------------------------
# DistinctNMetric
# ---------------------------------------------------------------------------

from src.metrics.distinct_n import DistinctNMetric


class TestDistinctN:
    def test_score_in_unit_interval(self):
        r = DistinctNMetric(n=2).compute(_q("What is the main function of mitochondria?"), _CTX)
        assert 0.0 <= r.score <= 1.0

    def test_empty_text_zero_flagged(self):
        r = DistinctNMetric(n=2).compute(_q(""), _CTX)
        assert r.score == 0.0 and r.flagged is True

    def test_all_unique_bigrams_score_one(self):
        # "cat sat on mat" → bigrams: (cat,sat),(sat,on),(on,mat) — all unique
        r = DistinctNMetric(n=2).compute(_q("cat sat on mat"), _CTX)
        assert r.score == pytest.approx(1.0)

    def test_repeated_tokens_lower_score(self):
        diverse  = DistinctNMetric(n=2).compute(_q("cat sat on mat"), _CTX)
        repeated = DistinctNMetric(n=2).compute(_q("cat cat cat cat"), _CTX)
        assert diverse.score > repeated.score

    def test_unigram_variant(self):
        r = DistinctNMetric(n=1).compute(_q("cat sat on mat"), _CTX)
        assert r.score == pytest.approx(1.0)  # all unique unigrams

    def test_name_reflects_n(self):
        assert DistinctNMetric(n=2).name == "distinct_2"
        assert DistinctNMetric(n=3).name == "distinct_3"

    def test_metadata_has_n(self):
        r = DistinctNMetric(n=2).compute(_q("What is photosynthesis?"), _CTX)
        assert r.metadata["n"] == 2

    def test_context_ignored(self):
        m = DistinctNMetric(n=2)
        r1 = m.compute(_q("What is photosynthesis?"), _CTX)
        r2 = m.compute(_q("What is photosynthesis?"), _CTX_FULL)
        assert r1.score == r2.score


# ---------------------------------------------------------------------------
# VocabularyNoveltyMetric
# ---------------------------------------------------------------------------

from src.metrics.vocabulary_novelty import VocabularyNoveltyMetric


class TestVocabularyNovelty:
    def test_no_course_content_zero_flagged(self):
        r = VocabularyNoveltyMetric().compute(_q("What is photosynthesis?"), _CTX)
        assert r.score == 0.0 and r.flagged is True

    def test_empty_text_zero_flagged(self):
        r = VocabularyNoveltyMetric().compute(_q(""), _CTX_FULL)
        assert r.score == 0.0 and r.flagged is True

    def test_score_in_unit_interval(self):
        r = VocabularyNoveltyMetric().compute(_q("What is photosynthesis?"), _CTX_FULL)
        assert 0.0 <= r.score <= 1.0

    def test_all_novel_words_score_one(self):
        ctx = EvaluationContext(course_content="apple banana cherry")
        r = VocabularyNoveltyMetric().compute(_q("dogs cats birds?"), ctx)
        assert r.score == pytest.approx(1.0)

    def test_all_shared_words_score_zero(self):
        ctx = EvaluationContext(course_content="what is photosynthesis")
        r = VocabularyNoveltyMetric().compute(_q("what is photosynthesis"), ctx)
        assert r.score == pytest.approx(0.0)

    def test_partial_overlap(self):
        ctx = EvaluationContext(course_content="photosynthesis converts light energy")
        r = VocabularyNoveltyMetric().compute(
            _q("How does photosynthesis produce oxygen?"), ctx
        )
        assert 0.0 < r.score < 1.0

    def test_metadata_fields(self):
        r = VocabularyNoveltyMetric().compute(_q("What is photosynthesis?"), _CTX_FULL)
        assert "novelty_ratio" in r.metadata
        assert "novel_word_count" in r.metadata
        assert "question_word_count" in r.metadata
