"""
Distinct-N metric (Diversity).

Original paper (corpus-level):
    Li, J., Galley, M., Brockett, C., Gao, J., & Dolan, B. (2016).
    A diversity-promoting objective function for neural conversation models.
    NAACL-HLT 2016, 110–119. https://aclanthology.org/N16-1014/

What it measures (per-question adaptation):
    Ratio of distinct n-grams to total n-grams within the question text.

        distinct_n = |unique n-grams| / |total n-grams|

    For a single question this is a vocabulary-diversity measure analogous to
    Type-Token Ratio (TTR) but for n-gram sequences:
        Distinct-1 ≡ TTR (already in text_counts.py)
        Distinct-2 captures bigram diversity — low values indicate repetitive
                   phrase patterns or very short texts.

    Note: The original Li et al. metric is corpus-level (computed across a set
    of generated responses). The per-question version here is a local proxy
    useful for single-item analysis.

Score:
    score = distinct_n (already in [0, 1])
    Higher score → more unique n-gram sequences → richer lexical diversity.
    flag_below default 0.5 for bigrams.

Dependency:
    None — pure Python, no external packages required.
"""

from __future__ import annotations
import re
from typing import Sequence
from src.models import MetricResult
from src.metrics.base_readability import BaseReadabilityMetric


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def _distinct_n(text: str, n: int) -> float:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    grams = _ngrams(tokens, n)
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


class DistinctNMetric(BaseReadabilityMetric):
    """
    Distinct-N ratio: unique n-grams / total n-grams within the question.

    Higher score → more diverse n-gram vocabulary → greater lexical variety.

    Args:
        n          : N-gram order (default 2 — bigrams).
        flag_below : Score threshold below which flagged=True (default 0.5).
    """

    name = "distinct_n"
    description = (
        "Distinct-N ratio (unique n-grams / total n-grams) "
        "(higher score = more diverse n-gram vocabulary)."
    )

    def __init__(self, n: int = 2, flag_below: float = 0.5):
        self.n = n
        self.flag_below = flag_below
        # Make name reflect the chosen N
        self.name = f"distinct_{n}"

    def _compute_raw(self, text: str) -> float:
        return _distinct_n(text, self.n)

    def _normalize(self, raw: float) -> float:
        return raw  # already in [0, 1]

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Distinct-{self.n} ratio {raw:.4f} "
                f"(unique {self.n}-grams / total {self.n}-grams; score={score:.4f})."
            ),
            flagged=score < self.flag_below,
            metadata={"distinct_n": raw, "n": self.n},
        )
