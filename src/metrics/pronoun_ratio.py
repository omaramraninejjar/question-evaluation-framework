"""
Pronoun Ratio metric (Ambiguity).

Reference:
    Hirst, G., & Gerber, L. (1995). Approximate coreference resolution.
    Proceedings of AAAI, 95, 156–163.

    Mitkov, R. (2002). Anaphora Resolution. Pearson Education.
    Chapter 2: Sources of ambiguity in pronoun interpretation.

What it measures:
    Fraction of word tokens that are pronouns (PRP, PRP$, WP, WP$).

    Pronouns introduce potential coreference ambiguity: without a passage or
    other context, a question containing "it", "they", "he", "she", or "this"
    may be unclear about the referent. High pronoun ratio → higher ambiguity
    risk.

    Example ambiguous: "What does it produce?"
    Example clear:     "What does the mitochondrion produce?"

Score normalisation:
    score = max(0.0, 1.0 − ratio)
    Higher score → fewer pronouns → lower coreference-ambiguity risk.
    flag_below default 0.5 (ratio > 0.5 is flagged as very pronoun-heavy).

Dependency:
    nltk >= 3.8   (already a core requirement)
    nltk.download('averaged_perceptron_tagger')
"""

from __future__ import annotations
import logging
from src.models import MetricResult
from src.metrics.base_readability import BaseReadabilityMetric

logger = logging.getLogger(__name__)

import nltk as _nltk
for _res in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "punkt",
             "punkt_tab"):
    try:
        _nltk.download(_res, quiet=True)
    except Exception:
        pass

from nltk import pos_tag as _pos_tag, word_tokenize as _word_tokenize

_PRONOUN_TAGS = frozenset({"PRP", "PRP$"})  # WP/WP$ are WH-pronouns (structural, not coreference-ambiguous)


class PronounRatioMetric(BaseReadabilityMetric):
    """
    Pronoun ratio: fraction of tokens that are pronouns.

    Higher score → fewer pronouns → lower coreference-ambiguity risk.

    Args:
        flag_below : Score threshold below which flagged=True (default 0.5).
    """

    name = "pronoun_ratio"
    description = (
        "Pronoun ratio (fraction of pronoun tokens) "
        "(higher score = fewer pronouns = lower coreference ambiguity)."
    )

    def __init__(self, flag_below: float = 0.5):
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        tokens = _word_tokenize(text)
        words = [t for t in tokens if t.isalpha()]
        if not words:
            return 0.0
        tagged = _pos_tag(words)
        n_pronouns = sum(1 for _, tag in tagged if tag in _PRONOUN_TAGS)
        return n_pronouns / len(words)

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - raw)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Pronoun ratio {raw:.4f} "
                f"(normalised {score:.4f})."
            ),
            flagged=score < self.flag_below,
            metadata={"pronoun_ratio": raw},
        )
