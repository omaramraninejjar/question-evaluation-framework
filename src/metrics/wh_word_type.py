"""
WH-Word Type metric (Ambiguity).

Reference:
    Graesser, A. C., Cai, Z., Louwerse, M. M., & Daniel, F. (2006).
    Question Understanding Aid (QUAID): A web facility that tests
    problem comprehensibility of survey questions.
    Public Opinion Quarterly, 70(1), 3–22.
    https://doi.org/10.1093/poq/nfj007

What it measures:
    Detects the leading WH-word of a question and classifies it.

    WH-words signal the cognitive operation required:
        what  → object / definition  (moderately specific)
        who   → agent / person       (specific)
        where → location             (specific)
        when  → time                 (specific)
        which → selection            (specific)
        why   → reason / cause       (open-ended — more ambiguous)
        how   → process / manner     (open-ended — more ambiguous)
        none  → non-interrogative or imperative (potentially ambiguous)

Scoring:
    +1.0  specific WH-words: who, where, when, which
    +0.7  moderately specific: what
    +0.4  open-ended: why, how
    +0.2  no WH-word detected (not a direct question)

    Multiple WH-words (compound question): score / 2
    (Compound questions are harder to answer and more ambiguous.)

Higher score → more specific, less ambiguous question type.

Dependency:
    None — pure regex, no external packages.
"""

from __future__ import annotations
import re
import logging
from src.models import MetricResult
from src.metrics.base_readability import BaseReadabilityMetric

logger = logging.getLogger(__name__)

_WH_SCORES: dict[str, float] = {
    "who":   1.0,
    "where": 1.0,
    "when":  1.0,
    "which": 1.0,
    "what":  0.7,
    "why":   0.4,
    "how":   0.4,
}

_WH_PATTERN = re.compile(
    r"\b(what|who|where|when|which|why|how)\b", re.IGNORECASE
)


class WHWordTypeMetric(BaseReadabilityMetric):
    """
    WH-word type score: specificity of the interrogative form.

    Higher score → more specific WH-word → lower ambiguity.

    Args:
        flag_below : Score threshold below which flagged=True (default 0.3).
    """

    name = "wh_word_type"
    description = (
        "WH-word type specificity "
        "(higher score = more specific interrogative = lower structural ambiguity)."
    )

    def __init__(self, flag_below: float = 0.3):
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        # Returns a tuple packed as float for compatibility; decoded in _build_result
        matches = _WH_PATTERN.findall(text)
        return float(len(matches))  # raw = count of WH-words found

    def _normalize(self, raw: float) -> float:
        # Not used — we override compute-level logic in _build_result
        return 0.0

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
        matches = [m.lower() for m in _WH_PATTERN.findall(text)]
        unique = list(dict.fromkeys(matches))  # preserve order, deduplicate

        if not unique:
            score = 0.2
            wh_type = "none"
        else:
            base = max(_WH_SCORES.get(w, 0.5) for w in unique)
            score = base / 2.0 if len(unique) > 1 else base
            wh_type = unique[0] if len(unique) == 1 else "+".join(unique)

        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"WH-word(s): {wh_type!r} → score {score:.2f} "
                f"({'compound question' if len(unique) > 1 else 'single WH-word'})."
            ),
            flagged=score < self.flag_below,
            metadata={
                "wh_type": wh_type,
                "wh_words_found": unique,
                "n_wh_words": len(unique),
            },
        )

    # Required by abstract base — not called (compute is overridden)
    def _build_result(self, score: float, raw: float) -> MetricResult:  # pragma: no cover
        raise NotImplementedError
