"""
Tests for ConceptCoverageDimension (src/dimensions/pedagogical/concept_coverage.py).

Structure tests (fast — no model inference):
    Verify dimension identity, metric presence, and reference_source assignment.

Edge-case scoring tests (fast — models exit before inference on empty/missing inputs):
    Empty question and missing course_content hit early-exit paths in all metrics.

Core scoring test (runs BLEU/ROUGE-L/METEOR only):
    Only the three unconditional metrics are exercised for speed; all optional
    metrics (chrF, TF-IDF, BM25, BERTScore, BLEURT, BARTScore, COMET, SBERT,
    NLI) are tested in their own test files.
"""

from __future__ import annotations

import pytest

from src.models import Question, EvaluationContext, DimensionName
from src.dimensions.pedagogical.concept_coverage import ConceptCoverageDimension
from src.metrics.chrf import _CHRF_AVAILABLE
from src.metrics.tfidf import _TFIDF_AVAILABLE
from src.metrics.bm25 import _BM25_AVAILABLE
from src.metrics.bertscore import _BERT_SCORE_AVAILABLE
from src.metrics.bleurt import _BLEURT_AVAILABLE
from src.metrics.bartscore import _BART_SCORE_AVAILABLE
from src.metrics.comet import _COMET_AVAILABLE
from src.metrics.sentencebert import _SBERT_AVAILABLE
from src.metrics.nli import _NLI_AVAILABLE

_CORE_METRICS = {"bleu", "rouge_l", "meteor"}  # unconditionally loaded


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _q(text: str) -> Question:
    return Question(id="q1", text=text)

def _ctx(objectives: list[str] | None = None, course_content: str | None = None) -> EvaluationContext:
    return EvaluationContext(
        learning_objectives=objectives or [],
        course_content=course_content,
    )


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

def test_dimension_name():
    assert ConceptCoverageDimension().name == DimensionName.CONCEPT_COVERAGE

def test_core_metrics_always_present():
    d = ConceptCoverageDimension()
    names = {m.name for m in d.metrics}
    assert _CORE_METRICS.issubset(names)

def test_optional_metrics_present_when_available():
    d = ConceptCoverageDimension()
    names = {m.name for m in d.metrics}
    if _CHRF_AVAILABLE:
        assert "chrf" in names
    if _TFIDF_AVAILABLE:
        assert "tfidf" in names
    if _BM25_AVAILABLE:
        assert "bm25" in names
    if _BERT_SCORE_AVAILABLE:
        assert "bertscore" in names
    if _BLEURT_AVAILABLE:
        assert "bleurt" in names
    if _BART_SCORE_AVAILABLE:
        assert "bartscore" in names
    if _COMET_AVAILABLE:
        assert "comet" in names
    if _SBERT_AVAILABLE:
        assert "sbert" in names
    if _NLI_AVAILABLE:
        assert "nli_entailment" in names

def test_all_metrics_use_course_content():
    d = ConceptCoverageDimension()
    for m in d.metrics:
        assert m.reference_source == "course_content", (
            f"{m.name} has reference_source={m.reference_source!r}, expected 'course_content'"
        )

def test_repr_includes_class_and_metric_names():
    d = ConceptCoverageDimension()
    r = repr(d)
    assert "ConceptCoverageDimension" in r
    for name in _CORE_METRICS:
        assert name in r

def test_each_instantiation_gets_its_own_metrics_list():
    """Instance-level list must not be shared across instances."""
    d1 = ConceptCoverageDimension()
    d2 = ConceptCoverageDimension()
    assert d1.metrics is not d2.metrics


# ---------------------------------------------------------------------------
# Edge-case scoring (fast — early exit before any model is loaded)
# ---------------------------------------------------------------------------

def test_no_course_content_all_metrics_flagged_zero():
    """All metrics exit before inference when course_content is None."""
    d = ConceptCoverageDimension()
    result = d.score(_q("What is osmosis?"), _ctx())
    assert result.dimension == DimensionName.CONCEPT_COVERAGE
    for name, mr in result.scores.items():
        assert mr.score == 0.0, f"{name}: expected score=0.0, got {mr.score}"
        assert mr.flagged is True, f"{name}: expected flagged=True"

def test_empty_question_all_metrics_flagged_zero():
    """All metrics exit before inference when question text is empty."""
    d = ConceptCoverageDimension()
    result = d.score(_q(""), _ctx(course_content="Photosynthesis converts light energy."))
    for name, mr in result.scores.items():
        assert mr.score == 0.0, f"{name}: expected score=0.0, got {mr.score}"
        assert mr.flagged is True, f"{name}: expected flagged=True"

def test_dimension_result_keys_match_loaded_metrics():
    """DimensionResult.scores keys must equal the set of loaded metric names."""
    d = ConceptCoverageDimension()
    result = d.score(_q("What is osmosis?"), _ctx())   # early exit — no model load
    assert set(result.scores.keys()) == {m.name for m in d.metrics}


# ---------------------------------------------------------------------------
# Core scoring (BLEU / ROUGE-L / METEOR only)
# ---------------------------------------------------------------------------

def test_core_metrics_return_nonzero_on_related_question():
    """Core metrics should score > 0 when question relates to the course content."""
    d = ConceptCoverageDimension()
    result = d.score(
        _q("What is photosynthesis?"),
        _ctx(course_content="Photosynthesis is the process by which plants convert light energy into chemical energy."),
    )
    for name in _CORE_METRICS:
        assert result.scores[name].score > 0.0, f"{name}: expected score > 0.0"

def test_core_metrics_score_higher_for_relevant_than_unrelated():
    """Relevant question should score higher than a completely off-topic one."""
    d = ConceptCoverageDimension()
    content = "Photosynthesis converts solar energy into chemical energy stored in glucose."
    ctx = _ctx(course_content=content)
    relevant  = d.score(_q("How do plants use sunlight to produce glucose?"), ctx)
    unrelated = d.score(_q("What is the formula for the area of a circle?"), ctx)
    for name in _CORE_METRICS:
        assert relevant.scores[name].score >= unrelated.scores[name].score, (
            f"{name}: relevant score {relevant.scores[name].score:.4f} should be >= "
            f"unrelated score {unrelated.scores[name].score:.4f}"
        )

def test_flagging_propagates_to_result():
    """A weak question should have at least one flagged metric in core set."""
    d = ConceptCoverageDimension()
    result = d.score(
        _q("x"),   # near-empty — will score very low on all core metrics
        _ctx(course_content="Photosynthesis converts solar energy into chemical energy stored in glucose."),
    )
    core_flagged = [result.scores[n].flagged for n in _CORE_METRICS]
    assert any(core_flagged), "Expected at least one core metric to be flagged for a near-empty question"

def test_distinct_from_curriculum_alignment():
    """ConceptCoverage uses course_content; CurriculumAlignment uses learning_objectives."""
    from src.dimensions.pedagogical.curriculum_alignment import CurriculumAlignmentDimension
    ca = CurriculumAlignmentDimension()
    cc = ConceptCoverageDimension()
    assert all(m.reference_source == "learning_objectives" for m in ca.metrics)
    assert all(m.reference_source == "course_content" for m in cc.metrics)
