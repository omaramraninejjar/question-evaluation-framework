"""
Dependency Distance metric (Linguistic Complexity).

Original paper:
    Liu, H. (2008). Dependency distance as a metric of language comprehension difficulty.
    Journal of Cognitive Science, 9(2), 159–191.

    Futrell, R., Mahowald, K., & Gibson, E. (2015). Large-scale evidence of dependency
    length minimization in 37 languages. PNAS, 112(33), 10336–10341.
    https://doi.org/10.1073/pnas.1502134112

What it measures:
    Mean Dependency Distance (MDD): average linear distance between each
    dependent token and its syntactic head.

        MDD = mean(|index(token) − index(head)|)  for all non-root tokens

    Larger MDD → more distant dependencies → greater syntactic processing load.
    English prose averages MDD ≈ 2.0–3.5; complex academic text reaches 4–6.

Score normalisation:
    score = max(0.0, 1.0 − MDD / max_dist)
    Default max_dist = 8.0.
    Higher score → shorter dependencies → lower processing complexity.

Dependency:
    spacy >= 3.0          (pip install spacy)
    en_core_web_sm model  (python -m spacy download en_core_web_sm)
"""

from __future__ import annotations
import logging
from src.models import MetricResult
from src.metrics.base_readability import BaseReadabilityMetric

logger = logging.getLogger(__name__)

try:
    import spacy as _spacy
    _SPACY_AVAILABLE = True
except Exception:
    _spacy = None  # type: ignore[assignment]
    _SPACY_AVAILABLE = False

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        for model_name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
            try:
                _nlp = _spacy.load(model_name)
                return _nlp
            except OSError:
                continue
        raise RuntimeError(
            "No spaCy English model found.\n"
            "Run: python -m spacy download en_core_web_sm"
        )
    return _nlp


class DependencyDistanceMetric(BaseReadabilityMetric):
    """
    Mean Dependency Distance (MDD) for question.text.

    Higher score → shorter average arc length → lower syntactic complexity.

    Args:
        max_dist   : MDD value that maps to score 0.0 (default 8.0).
        flag_below : Score threshold below which flagged=True (default 0.3).
    """

    name = "dependency_distance"
    description = (
        "Mean Dependency Distance (avg arc length in dependency parse) "
        "(higher score = shorter arcs = lower syntactic complexity)."
    )

    def __init__(self, max_dist: float = 8.0, flag_below: float = 0.3):
        if not _SPACY_AVAILABLE:
            raise ImportError(
                "spacy is required for DependencyDistanceMetric.\n"
                "pip install spacy && python -m spacy download en_core_web_sm"
            )
        self.max_dist = max_dist
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        nlp = _get_nlp()
        doc = nlp(text)
        dists = [abs(tok.i - tok.head.i) for tok in doc if tok.dep_ != "ROOT"]
        return sum(dists) / len(dists) if dists else 0.0

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - raw / self.max_dist)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Mean Dependency Distance {raw:.2f} "
                f"(normalised {score:.4f}; max_dist={self.max_dist})."
            ),
            flagged=score < self.flag_below,
            metadata={"mean_dependency_distance": raw, "max_dist": self.max_dist},
        )
