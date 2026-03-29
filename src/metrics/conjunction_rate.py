"""
Conjunction Rate metric (Linguistic Complexity).

Reference:
    Halliday, M. A. K., & Hasan, R. (1976). Cohesion in English.
    Longman. Chapter 5: Conjunctive Cohesion.

    McNamara, D. S., Graesser, A. C., McCarthy, P. M., & Cai, Z. (2014).
    Automated Evaluation of Text and Discourse with Coh-Metrix.
    Cambridge University Press. Section 7.3: Connectives and conjunctions.

What it measures:
    Ratio of conjunction tokens (coordinating and subordinating) to total
    word tokens.

    Conjunctions link clauses and phrases:
      Coordinating (CC): and, but, or, nor, for, so, yet
      Subordinating:     although, because, since, while, if, unless, until,
                         after, before, as, though, whenever, wherever, once,
                         provided, whereas, whether

    A high conjunction rate signals clause-coordination complexity — the
    respondent must parse multiple linked propositions simultaneously.

Score normalisation:
    score = max(0.0, 1.0 − rate / max_rate)
    Default max_rate = 0.3 (30 % of tokens being conjunctions is very high).
    Higher score → fewer conjunctions → simpler clause structure.
    flag_below default 0.3.

Dependency:
    nltk >= 3.8   (already a core requirement)
    nltk.download('averaged_perceptron_tagger')
"""

from __future__ import annotations
import re
import logging
from src.models import MetricResult
from src.metrics.base_readability import BaseReadabilityMetric

logger = logging.getLogger(__name__)

import nltk as _nltk
for _res in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
             "punkt", "punkt_tab"):
    try:
        _nltk.download(_res, quiet=True)
    except Exception:
        pass

from nltk import pos_tag as _pos_tag, word_tokenize as _word_tokenize

_SUBORDINATING_CONJ = frozenset({
    "although", "because", "since", "while", "if", "unless", "until",
    "after", "before", "as", "though", "whenever", "wherever", "once",
    "provided", "whereas", "whether", "even",
})


def _conjunction_count(text: str) -> tuple[int, int]:
    """Return (n_conjunctions, n_total_words)."""
    tokens = _word_tokenize(text)
    words = [t for t in tokens if t.isalpha()]
    if not words:
        return 0, 0
    tagged = _pos_tag(words)
    count = 0
    for word, tag in tagged:
        if tag == "CC":  # coordinating conjunction
            count += 1
        elif tag == "IN" and word.lower() in _SUBORDINATING_CONJ:
            count += 1
    return count, len(words)


class ConjunctionRateMetric(BaseReadabilityMetric):
    """
    Conjunction rate: coordinating + subordinating conjunctions / total words.

    Higher score → fewer conjunctions → simpler clause coordination.

    Args:
        max_rate   : Rate that maps to score 0.0 (default 0.3).
        flag_below : Score threshold below which flagged=True (default 0.3).
    """

    name = "conjunction_rate"
    description = (
        "Conjunction rate (conjunctions / total words) "
        "(higher score = fewer conjunctions = simpler clause structure)."
    )

    def __init__(self, max_rate: float = 0.3, flag_below: float = 0.3):
        self.max_rate = max_rate
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        count, total = _conjunction_count(text)
        return count / total if total else 0.0

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - raw / self.max_rate)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Conjunction rate {raw:.4f} "
                f"(normalised {score:.4f}; max_rate={self.max_rate})."
            ),
            flagged=score < self.flag_below,
            metadata={"conjunction_rate": raw, "max_rate": self.max_rate},
        )
