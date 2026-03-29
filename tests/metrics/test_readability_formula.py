"""
Tests for textstat formula readability metrics.

Covers: FleschReadingEaseMetric, FleschKincaidMetric, DaleChallMetric,
        GunningFogMetric, ColemanLiauMetric, ARIMetric, SMOGMetric,
        LinsearWriteMetric.

Requires:
    pip install textstat

All tests are skipped gracefully when textstat is not installed.

Score convention (all formula metrics):
    Higher score → more accessible (easier) text.
    Score is in [0.0, 1.0].
    The test text "What is the main function of photosynthesis?" is a
    simple educational question expected to sit in a mid-to-high
    accessibility range (≥ 0.3) for most grade-level formulas.
"""

from __future__ import annotations

import pytest

from src.models import Question, EvaluationContext
from src.metrics.flesch_ease import FleschReadingEaseMetric, _TEXTSTAT_AVAILABLE
from src.metrics.flesch_kincaid import FleschKincaidMetric
from src.metrics.dale_chall import DaleChallMetric
from src.metrics.gunning_fog import GunningFogMetric
from src.metrics.coleman_liau import ColemanLiauMetric
from src.metrics.ari import ARIMetric
from src.metrics.smog import SMOGMetric
from src.metrics.linsear_write import LinsearWriteMetric

if not _TEXTSTAT_AVAILABLE:
    pytest.skip(
        "textstat unavailable — install: pip install textstat",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIMPLE   = "What is the main function of photosynthesis?"
_COMPLEX  = (
    "In the context of thermodynamic equilibrium, elucidate the "
    "intrinsic relationship between entropy, enthalpy, and the Gibbs "
    "free energy formulation as it pertains to spontaneous reactions."
)
_EMPTY    = ""

def _q(text: str) -> Question:
    return Question(id="q1", text=text)

_CTX = EvaluationContext(learning_objectives=[], course_content=None)

# All metric classes under test
_FORMULA_METRICS = [
    FleschReadingEaseMetric,
    FleschKincaidMetric,
    DaleChallMetric,
    GunningFogMetric,
    ColemanLiauMetric,
    ARIMetric,
    SMOGMetric,
    LinsearWriteMetric,
]


# ---------------------------------------------------------------------------
# Shared contract tests (parametrised over all formula metrics)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("MetricClass", _FORMULA_METRICS)
def test_score_in_unit_interval(MetricClass):
    result = MetricClass().compute(_q(_SIMPLE), _CTX)
    assert 0.0 <= result.score <= 1.0, (
        f"{MetricClass.__name__}: score {result.score} not in [0, 1]"
    )

@pytest.mark.parametrize("MetricClass", _FORMULA_METRICS)
def test_empty_text_returns_zero_flagged(MetricClass):
    result = MetricClass().compute(_q(_EMPTY), _CTX)
    assert result.score == 0.0, f"{MetricClass.__name__}: expected score=0.0 on empty text"
    assert result.flagged is True, f"{MetricClass.__name__}: expected flagged=True on empty text"

@pytest.mark.parametrize("MetricClass", _FORMULA_METRICS)
def test_result_has_required_fields(MetricClass):
    result = MetricClass().compute(_q(_SIMPLE), _CTX)
    assert result.metric_name == MetricClass().name
    assert isinstance(result.rationale, str) and result.rationale
    assert isinstance(result.metadata, dict)

@pytest.mark.parametrize("MetricClass", _FORMULA_METRICS)
def test_flagging_threshold_respected(MetricClass):
    """With flag_below=1.0 every score should be flagged."""
    # Most formula metrics accept flag_below as __init__ kwarg
    try:
        m = MetricClass(flag_below=1.0)
    except TypeError:
        pytest.skip(f"{MetricClass.__name__} does not accept flag_below kwarg")
    result = m.compute(_q(_SIMPLE), _CTX)
    assert result.flagged is True, (
        f"{MetricClass.__name__}: expected flagged=True with flag_below=1.0"
    )

@pytest.mark.parametrize("MetricClass", _FORMULA_METRICS)
def test_context_is_ignored(MetricClass):
    """Readability metrics should produce the same score regardless of context."""
    ctx_empty = EvaluationContext(learning_objectives=[], course_content=None)
    ctx_full  = EvaluationContext(
        learning_objectives=["understand photosynthesis"],
        course_content="Photosynthesis converts light energy.",
    )
    m = MetricClass()
    assert m.compute(_q(_SIMPLE), ctx_empty).score == m.compute(_q(_SIMPLE), ctx_full).score


# ---------------------------------------------------------------------------
# Per-metric spot checks
# ---------------------------------------------------------------------------

def test_flesch_ease_simple_text_accessible():
    """A simple question should score > 0 on Flesch ease."""
    result = FleschReadingEaseMetric().compute(_q(_SIMPLE), _CTX)
    assert result.score > 0.0

def test_flesch_ease_metadata_has_label():
    result = FleschReadingEaseMetric().compute(_q(_SIMPLE), _CTX)
    assert "label" in result.metadata

def test_flesch_kincaid_metadata_has_raw_grade():
    result = FleschKincaidMetric().compute(_q(_SIMPLE), _CTX)
    assert "raw_grade" in result.metadata

def test_dale_chall_metadata_has_grade_label():
    result = DaleChallMetric().compute(_q(_SIMPLE), _CTX)
    assert "grade_label" in result.metadata

def test_gunning_fog_metadata_has_raw_grade():
    result = GunningFogMetric().compute(_q(_SIMPLE), _CTX)
    assert "raw_grade" in result.metadata

def test_coleman_liau_metadata_has_raw_grade():
    result = ColemanLiauMetric().compute(_q(_SIMPLE), _CTX)
    assert "raw_grade" in result.metadata

def test_ari_metadata_has_raw_grade():
    result = ARIMetric().compute(_q(_SIMPLE), _CTX)
    assert "raw_grade" in result.metadata

def test_smog_metadata_has_raw_grade():
    result = SMOGMetric().compute(_q(_SIMPLE), _CTX)
    assert "raw_grade" in result.metadata

def test_linsear_write_metadata_has_raw_grade():
    result = LinsearWriteMetric().compute(_q(_SIMPLE), _CTX)
    assert "raw_grade" in result.metadata

def test_complex_text_not_more_accessible_than_simple():
    """Complex academic text should score ≤ simple question on FK grade."""
    simple_r  = FleschKincaidMetric().compute(_q(_SIMPLE), _CTX)
    complex_r = FleschKincaidMetric().compute(_q(_COMPLEX), _CTX)
    assert simple_r.score >= complex_r.score, (
        f"FK: simple {simple_r.score:.4f} should be >= complex {complex_r.score:.4f}"
    )

def test_custom_max_grade_affects_normalisation():
    """A lower max_grade ceiling should yield a lower (or equal) score."""
    default = FleschKincaidMetric(max_grade=18.0).compute(_q(_SIMPLE), _CTX)
    strict  = FleschKincaidMetric(max_grade=8.0).compute(_q(_SIMPLE), _CTX)
    assert strict.score <= default.score + 1e-9

def test_metric_names_are_distinct():
    names = [m().name for m in _FORMULA_METRICS]
    assert len(names) == len(set(names)), f"Duplicate metric names: {names}"
