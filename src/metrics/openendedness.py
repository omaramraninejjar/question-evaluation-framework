"""
Open-Endedness metric (Response Burden).

Reference:
    Tourangeau, R., Rips, L. J., & Rasinski, K. (2000). The Psychology of
    Survey Response. Cambridge University Press. Chapter 2: Response formats
    and response burden.

    Graesser, A. C., Cai, Z., Louwerse, M. M., & Daniel, F. (2006).
    Question Understanding Aid (QUAID). Public Opinion Quarterly, 70(1), 3–22.

What it measures:
    Degree to which the question requires an open-ended, elaborated response
    as opposed to a brief, closed-ended answer.

    Open-endedness is a source of response burden that is independent of the
    construct being measured: the effort required to formulate a full explanation
    is greater than locating a single fact, even if both tap the same knowledge.

    Classification heuristics (ordered by burden level):

        Very low  (score 0.9) — binary: "Is …?", "Does …?", "Can …?"
        Low       (score 0.7) — closed WH: who, what, when, where, which
        Moderate  (score 0.4) — explain, describe, summarize, outline
        High      (score 0.2) — why, how, justify, discuss, analyze, evaluate
        Very high (score 0.05) — write, create, design, construct, compose

    For compound questions (multiple types detected), the highest burden
    category governs.

Score:
    score = openendedness_score ∈ [0.05, 0.9]
    CONVENTION: higher score → LOWER burden → better item efficiency.
    flag_below default 0.3 (flags items requiring full essays/designs).

Dependency:
    None — pattern matching, no external packages required.
"""

from __future__ import annotations
import re
import logging
from src.models import MetricResult
from src.metrics.base_readability import BaseReadabilityMetric

logger = logging.getLogger(__name__)

# Patterns ordered from MOST to LEAST burdensome
_PATTERNS: list[tuple[float, re.Pattern]] = [
    # Very high burden — creative/extended writing
    (0.05, re.compile(
        r"\b(write|create|design|construct|compose|invent|produce|build|"
        r"generate|develop|formulate|plan|propose)\b",
        re.IGNORECASE,
    )),
    # High burden — explanatory / analytical
    (0.2, re.compile(
        r"\b(why|how|justify|argue|discuss|evaluate|assess|analyze|analyse|"
        r"critique|investigate|examine|compare|contrast|synthesize|synthesise|"
        r"interpret|reflect|elaborate)\b",
        re.IGNORECASE,
    )),
    # Moderate burden — descriptive / summarizing
    (0.4, re.compile(
        r"\b(explain|describe|summarize|summarise|outline|illustrate|"
        r"demonstrate|clarify|define|characterize|characterise)\b",
        re.IGNORECASE,
    )),
    # Low burden — closed WH
    (0.7, re.compile(
        r"\b(what|who|which|when|where)\b",
        re.IGNORECASE,
    )),
    # Very low burden — binary yes/no
    (0.9, re.compile(
        r"^(is|are|was|were|do|does|did|can|could|will|would|should|has|have)\b",
        re.IGNORECASE,
    )),
]


def _openendedness_score(text: str) -> tuple[float, str]:
    """Return (score, category_label) for the highest-burden category found."""
    text_stripped = text.strip()
    for score, pattern in _PATTERNS:
        if pattern.search(text_stripped):
            return score, f"score={score}"
    return 0.7, "default (closed WH)"  # fallback: treat as closed WH


class OpenEndednessMetric(BaseReadabilityMetric):
    """
    Open-endedness score: estimated response format burden.

    Higher score → more closed-ended → lower response burden.
    Lower score → more open-ended → higher response production demand.

    Args:
        flag_below : Score threshold below which flagged=True (default 0.3).
    """

    name = "openendedness"
    description = (
        "Open-endedness score (expected response format burden) "
        "(higher score = closed-ended = lower response burden)."
    )

    def __init__(self, flag_below: float = 0.3):
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        score, _ = _openendedness_score(text)
        return score

    def _normalize(self, raw: float) -> float:
        return raw  # already in [0, 1]

    def _build_result(self, score: float, raw: float) -> MetricResult:
        burden_label = (
            "very low" if score >= 0.8 else
            "low" if score >= 0.6 else
            "moderate" if score >= 0.35 else
            "high" if score >= 0.1 else
            "very high"
        )
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Open-endedness: {burden_label} response burden "
                f"(score={score:.4f})."
            ),
            flagged=score < self.flag_below,
            metadata={"openendedness_score": score, "burden_level": burden_label},
        )
