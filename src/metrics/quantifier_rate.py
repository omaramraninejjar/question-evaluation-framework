"""
Quantifier Rate metric (Ambiguity).

Reference:
    Barwise, J., & Cooper, R. (1981). Generalized quantifiers and natural
    language. Linguistics and Philosophy, 4(2), 159–219.
    https://doi.org/10.1007/BF00350139

    Graesser, A. C., Cai, Z., Louwerse, M. M., & Daniel, F. (2006).
    Question Understanding Aid (QUAID): A web facility that tests problem
    comprehensibility of survey questions. Public Opinion Quarterly, 70(1),
    3–22. https://doi.org/10.1093/poq/nfj007

What it measures:
    Ratio of quantifier word tokens to total word tokens.

    Quantifiers introduce logical scope that can be ambiguous without a
    precise answer target:
        "Which of the following is true for ALL cases?" — scope of "all"
        may differ by interpretation.

    Quantifier set: all, some, any, every, none, few, many, most, each,
                    both, either, neither, several, no

    High quantifier rate → higher potential for logical-scope ambiguity in
    the expected answer space.

Score normalisation:
    score = max(0.0, 1.0 − rate / max_rate)
    Default max_rate = 0.3.
    Higher score → fewer quantifiers → clearer logical scope.
    flag_below default 0.5.

Dependency:
    None — keyword list, no external packages required.
"""

from __future__ import annotations
import re
import logging
from src.models import MetricResult
from src.metrics.base_readability import BaseReadabilityMetric

logger = logging.getLogger(__name__)

_QUANTIFIERS = frozenset({
    "all", "some", "any", "every", "none", "few", "many", "most", "each",
    "both", "either", "neither", "several", "no",
})


class QuantifierRateMetric(BaseReadabilityMetric):
    """
    Quantifier rate: fraction of tokens that are logical quantifiers.

    Higher score → fewer quantifiers → clearer logical scope → less ambiguity.

    Args:
        max_rate   : Quantifier rate that maps to score 0.0 (default 0.3).
        flag_below : Score threshold below which flagged=True (default 0.5).
    """

    name = "quantifier_rate"
    description = (
        "Quantifier token rate (quantifiers / total words) "
        "(higher score = fewer quantifiers = clearer logical scope)."
    )

    def __init__(self, max_rate: float = 0.3, flag_below: float = 0.5):
        self.max_rate = max_rate
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        words = re.findall(r"[a-zA-Z]+", text.lower())
        if not words:
            return 0.0
        count = sum(1 for w in words if w in _QUANTIFIERS)
        return count / len(words)

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - raw / self.max_rate)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Quantifier rate {raw:.4f} "
                f"(normalised {score:.4f}; max_rate={self.max_rate})."
            ),
            flagged=score < self.flag_below,
            metadata={"quantifier_rate": raw, "max_rate": self.max_rate},
        )
