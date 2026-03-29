"""Shared pytest fixtures."""

import nltk
import pytest


@pytest.fixture(scope="session", autouse=True)
def download_nltk_data():
    """Download required NLTK corpora once per test session."""
    for resource in ("punkt_tab", "wordnet"):
        nltk.download(resource, quiet=True)
from src.models import (
    Question, EvaluationContext,
    MetricResult, DimensionResult, AspectResult, EvaluationResult,
    AspectName, DimensionName,
)


@pytest.fixture
def sample_question() -> Question:
    return Question(
        id="q_fixture",
        text="Which of the following best describes photosynthesis?",
        options=[
            "A. Conversion of light energy to chemical energy",
            "B. Breakdown of glucose to release energy",
        ],
        correct_answer="A. Conversion of light energy to chemical energy",
        subject="biology",
        grade_level="grade_9",
    )


@pytest.fixture
def sample_context() -> EvaluationContext:
    return EvaluationContext(
        learning_objectives=["Understand the process of photosynthesis"],
        course_content="Chapter 6: Photosynthesis converts solar energy into chemical energy.",
        rubric="Award 1 point for identifying light as the energy source.",
    )


@pytest.fixture
def sample_metric_result() -> MetricResult:
    return MetricResult(metric_name="sample_metric", score=0.75, rationale="Meets standard.")


@pytest.fixture
def sample_evaluation_result(sample_metric_result) -> EvaluationResult:
    dim = DimensionResult(DimensionName.READABILITY, scores={"sample_metric": sample_metric_result})
    asp = AspectResult(AspectName.LINGUISTIC_STRUCTURAL, scores={"readability": dim})
    return EvaluationResult(question_id="q_fixture", scores={"linguistic_structural": asp})