"""
Shared data models for the evaluation framework.
All layers (Aspect, Dimension, Metric) communicate through these types.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AspectName(str, Enum):
    PEDAGOGICAL            = "pedagogical"
    PSYCHOMETRIC           = "psychometric"
    LINGUISTIC_STRUCTURAL  = "linguistic_structural"
    FAIRNESS_ETHICS        = "fairness_ethics"


class DimensionName(str, Enum):
    # Pedagogical
    CURRICULUM_ALIGNMENT   = "curriculum_alignment"
    COGNITIVE_DEMAND       = "cognitive_demand"
    CONCEPT_COVERAGE       = "concept_coverage"
    RESPONSE_BURDEN        = "response_burden"
    # Psychometric
    DIFFICULTY             = "difficulty"
    DISCRIMINATION         = "discrimination"
    GUESSING_CARELESS      = "guessing_careless"
    DISTRACTOR_FUNCTIONING = "distractor_functioning"
    ITEM_FIT               = "item_fit"
    DIMENSIONALITY         = "dimensionality"
    RELIABILITY            = "reliability"
    # Linguistic and Structural
    READABILITY            = "readability"
    LINGUISTIC_COMPLEXITY  = "linguistic_complexity"
    AMBIGUITY              = "ambiguity"
    WELL_FORMEDNESS        = "well_formedness"
    DIVERSITY              = "diversity"
    # Fairness and Ethics
    GROUP_BIAS             = "group_bias"
    MEASUREMENT_INVARIANCE = "measurement_invariance"
    CONTENT_SENSITIVITY    = "content_sensitivity"
    HARMFUL_CONTENT_RISK   = "harmful_content_risk"
    PRIVACY_RISK           = "privacy_risk"


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

@dataclass
class Question:
    """
    A single assessment item.
    Contains only what belongs to the item itself — no curriculum context.
    """
    id: str
    text: str
    options: list[str] | None = None        # MCQ choices, if any
    correct_answer: str | None = None
    subject: str | None = None
    grade_level: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationContext:
    """
    Curriculum and course context used as reference material by metrics.

    Kept separate from Question so that:
      - Question stays lean (only item-level data)
      - Reference sources can grow without polluting the item schema
      - The same Question can be re-evaluated against different contexts

    Fields:
        learning_objectives : Reference texts for Curriculum Alignment.
                              Typically one string per objective, but a list
                              allows multiple acceptable phrasings.
        course_content      : Raw or summarised course text for Concept Coverage.
        rubric              : Scoring rubric text, used by rubric-based metrics.
        metadata            : Escape hatch for any future context fields a
                              specific metric may need.
    """
    learning_objectives: list[str] = field(default_factory=list)
    course_content: str | None = None
    rubric: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Result models  (Metric → Dimension → Aspect → Evaluation)
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    """Atomic output from a single metric computation."""
    metric_name: str
    score: float                            # Normalised 0.0–1.0
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    flagged: bool = False


@dataclass
class DimensionResult:
    """
    Result for one dimension.
    `scores` maps metric_name → MetricResult (no pre-aggregation).
    """
    dimension: DimensionName
    scores: dict[str, MetricResult] = field(default_factory=dict)
    notes: str = ""

    def metric_names(self) -> list[str]:
        return list(self.scores.keys())

    def flagged_metrics(self) -> list[MetricResult]:
        return [r for r in self.scores.values() if r.flagged]


@dataclass
class AspectResult:
    """
    Result for one aspect.
    `scores` maps dimension_name → DimensionResult (no pre-aggregation).
    """
    aspect: AspectName
    scores: dict[str, DimensionResult] = field(default_factory=dict)

    def dimension_names(self) -> list[str]:
        return list(self.scores.keys())

    def flagged_dimensions(self) -> list[DimensionResult]:
        return [d for d in self.scores.values() if d.flagged_metrics()]


@dataclass
class EvaluationResult:
    """
    Top-level output for a single (Question, EvaluationContext) pair.
    `scores` maps aspect_name → AspectResult (no pre-aggregation).
    Use Scorer to derive any aggregate numbers downstream.
    """
    question_id: str
    scores: dict[str, AspectResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def aspect_names(self) -> list[str]:
        return list(self.scores.keys())

    def flat_scores(self) -> dict[str, dict[str, dict[str, float]]]:
        """
        Fully-flat view for quick inspection or export.
        Shape: {aspect: {dimension: {metric: score}}}
        """
        return {
            a: {
                d: {m: r.score for m, r in dim.scores.items()}
                for d, dim in asp.scores.items()
            }
            for a, asp in self.scores.items()
        }