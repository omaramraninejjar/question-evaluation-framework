"""
Survey Quality Predictor (SQP) Score metric (Response Burden).

Reference:
    Saris, W. E., & Gallhofer, I. N. (2007). Design, Evaluation, and
    Analysis of Questionnaires for Survey Research. Wiley.
    https://doi.org/10.1002/9780470165096

    Saris, W. E., Revilla, M., Krosnick, J. A., & Shaeffer, E. M. (2010).
    Comparing questions with agree/disagree response options to questions
    with item-specific response options. Survey Research Methods, 4(1), 61–79.

    Oberski, D. L., Kirchner, A., Mayerl, J., & Saris, W. E. (2012).
    SQP: Survey Quality Predictor, a free software program for the design
    and analysis of survey questions. ESRA 2012.

What it measures:
    A simplified text-based implementation of the Survey Quality Predictor,
    which estimates question quality by detecting known sources of systematic
    error in item wording. The SQP system codes structural question features
    to predict reliability and validity; this metric operationalises the
    text-detectable subset:

        1. Double-barrel detection — single item asking about two distinct
           constructs connected by "and" / "or" (reduces validity)
        2. Leading question detection — stem presupposes or steers toward
           a particular answer (reduces validity)
        3. Loaded/emotionally-charged language — words with strong affective
           valence that may anchor responses (reduces reliability)
        4. Vague frequency anchors — "often", "sometimes", "rarely" as
           quantifier descriptors in the item itself (reduces reliability)

    Each violation adds a penalty. The composite score reflects the
    predicted absence of these error sources.

    penalty = Σ(weight_i × violation_i)
    score   = max(0.0, 1.0 − penalty / max_penalty)

    Weights: double_barrel=0.4, leading=0.35, loaded=0.15, vague_freq=0.10

Score:
    score ∈ [0.0, 1.0]
    Higher score → fewer detected error sources → higher predicted quality.
    flag_below default 0.5 (any single major violation → flag).

Dependency:
    None — pattern matching, no external packages required.
"""

from __future__ import annotations
import re
import logging
from src.models import MetricResult
from src.metrics.base_readability import BaseReadabilityMetric

logger = logging.getLogger(__name__)

# --- Double-barrel detection ---
# Two distinct imperative/wh clauses connected by "and" or "or"
_DOUBLE_BARREL = re.compile(
    r"\b(and|or)\b.{5,50}\b(and|or)\b",  # simple: 2+ connectors in a question
    re.IGNORECASE,
)
_AND_SPLIT = re.compile(r"\b(and|or)\b", re.IGNORECASE)

# --- Leading question patterns ---
_LEADING_PATTERNS = [
    re.compile(r"\bdon't you (think|agree|believe|feel)\b", re.IGNORECASE),
    re.compile(r"\bwouldn't you (say|agree|expect)\b", re.IGNORECASE),
    re.compile(r"\bisn't it (true|the case|obvious)\b", re.IGNORECASE),
    re.compile(r"\bsurely\b", re.IGNORECASE),
    re.compile(r"\bobviously\b", re.IGNORECASE),
    re.compile(r"\bclearly\b", re.IGNORECASE),
    re.compile(r"\bof course\b", re.IGNORECASE),
]

# --- Loaded / emotionally-charged language ---
_LOADED_WORDS = frozenset({
    "terrible", "awful", "horrible", "disgusting", "outrageous", "shameful",
    "wonderful", "magnificent", "excellent", "superb", "perfect", "fantastic",
    "ridiculous", "absurd", "unacceptable", "devastating", "catastrophic",
    "revolutionary", "groundbreaking", "disastrous", "tragic",
})

# --- Vague frequency anchors ---
_VAGUE_FREQ = frozenset({
    "often", "sometimes", "rarely", "occasionally", "frequently",
    "usually", "generally", "seldom", "regularly", "commonly",
})

_W_DOUBLE_BARREL = 0.40
_W_LEADING = 0.35
_W_LOADED = 0.15
_W_VAGUE = 0.10
_MAX_PENALTY = _W_DOUBLE_BARREL + _W_LEADING + _W_LOADED + _W_VAGUE  # = 1.0


def _detect_double_barrel(text: str) -> bool:
    """True if question appears to ask about two distinct things."""
    # More robust: count number of verbs/clauses split by "and"
    parts = _AND_SPLIT.split(text)
    # If splitting by and/or gives 3+ meaningful parts (>4 words each), double barrel
    meaningful_parts = [p.strip() for p in parts if len(re.findall(r"[a-zA-Z]+", p)) >= 4]
    return len(meaningful_parts) >= 3


def _detect_leading(text: str) -> bool:
    return any(p.search(text) for p in _LEADING_PATTERNS)


def _detect_loaded(text: str) -> bool:
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    return bool(words & _LOADED_WORDS)


def _detect_vague_freq(text: str) -> bool:
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    return bool(words & _VAGUE_FREQ)


class SQPScoreMetric(BaseReadabilityMetric):
    """
    SQP-inspired quality score: detects known sources of question wording error.

    Higher score → fewer detected error sources → higher predicted reliability
    and validity of the item.

    Args:
        flag_below : Score threshold below which flagged=True (default 0.5).
    """

    name = "sqp_score"
    description = (
        "Survey Quality Predictor score (absence of double-barrel, leading, "
        "loaded, and vague-frequency wording flaws) "
        "(higher score = cleaner item wording = lower measurement error)."
    )

    def __init__(self, flag_below: float = 0.5):
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        """Return penalty [0, 1]."""
        penalty = 0.0
        if _detect_double_barrel(text):
            penalty += _W_DOUBLE_BARREL
        if _detect_leading(text):
            penalty += _W_LEADING
        if _detect_loaded(text):
            penalty += _W_LOADED
        if _detect_vague_freq(text):
            penalty += _W_VAGUE
        return min(penalty, _MAX_PENALTY)

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - raw / _MAX_PENALTY)

    def compute(self, question, context):
        if not question.text.strip():
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                rationale="Question text is empty.",
                flagged=True,
                metadata={"reason": "empty_text"},
            )
        text = question.text
        double_barrel = _detect_double_barrel(text)
        leading = _detect_leading(text)
        loaded = _detect_loaded(text)
        vague = _detect_vague_freq(text)

        penalty = (
            (_W_DOUBLE_BARREL if double_barrel else 0.0)
            + (_W_LEADING if leading else 0.0)
            + (_W_LOADED if loaded else 0.0)
            + (_W_VAGUE if vague else 0.0)
        )
        score = max(0.0, 1.0 - penalty / _MAX_PENALTY)

        violations = []
        if double_barrel:
            violations.append("double-barrel")
        if leading:
            violations.append("leading question")
        if loaded:
            violations.append("loaded language")
        if vague:
            violations.append("vague frequency anchor")

        rationale = (
            f"SQP violations detected: {violations} (penalty={penalty:.2f}; score={score:.4f})."
            if violations
            else f"No SQP wording flaws detected (score={score:.4f})."
        )
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=rationale,
            flagged=score < self.flag_below,
            metadata={
                "double_barrel": double_barrel,
                "leading_question": leading,
                "loaded_language": loaded,
                "vague_freq_anchor": vague,
                "penalty": penalty,
            },
        )

    def _build_result(self, score: float, raw: float) -> MetricResult:  # pragma: no cover
        raise NotImplementedError
