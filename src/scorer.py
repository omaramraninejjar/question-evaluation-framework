"""
Scorer — optional post-processing utility for deriving aggregate numbers
from the granular dicts produced by EvaluationResult.

The Evaluator never calls Scorer. You call it explicitly when you need
a rolled-up value for reporting, ranking, or thresholding.

Usage examples:

    scorer  = Scorer()
    result  = evaluator.evaluate(question)

    # Roll up one dimension to a single number
    dim_score = scorer.aggregate_dimension(
        result.scores["linguistic_structural"].scores["readability"]
    )

    # Roll up one aspect
    asp_score = scorer.aggregate_aspect(
        result.scores["pedagogical"]
    )

    # Roll up the entire question
    overall = scorer.aggregate_evaluation(result)

    # Apply min-gate on specific metrics (e.g. safety)
    safety  = scorer.aggregate_dimension(
        result.scores["fairness_ethics"].scores["harmful_content_risk"],
        strategy="min_gate",
    )
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from src.models import DimensionResult, AspectResult, EvaluationResult


# ---------------------------------------------------------------------------
# Strategy base
# ---------------------------------------------------------------------------

class ScoringStrategy(ABC):

    @abstractmethod
    def aggregate(self, scores: list[float], weights: list[float] | None = None) -> float:
        """
        Combine a list of scores into a single value.

        Args:
            scores:  List of normalised scores (0.0–1.0).
            weights: Optional per-score weights (same length as scores).
                     If None, equal weights are used.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class WeightedAverageStrategy(ScoringStrategy):
    """Standard weighted mean. Default for all layers."""

    def aggregate(self, scores: list[float], weights: list[float] | None = None) -> float:
        if not scores:
            return 0.0
        w = weights if weights else [1.0] * len(scores)
        if len(w) != len(scores):
            raise ValueError("scores and weights must have the same length.")
        total_w = sum(w)
        return sum(s * wi for s, wi in zip(scores, w)) / total_w if total_w else 0.0


class MinGateStrategy(ScoringStrategy):
    """
    Returns the minimum score in the list.
    Recommended for risk/safety dimensions where one bad metric dominates.
    """

    def aggregate(self, scores: list[float], weights: list[float] | None = None) -> float:
        return min(scores) if scores else 0.0


class PercentileRankStrategy(ScoringStrategy):
    """Returns the score at a given percentile of the distribution."""

    def __init__(self, percentile: float = 50.0):
        if not (0.0 <= percentile <= 100.0):
            raise ValueError("percentile must be between 0 and 100.")
        self.percentile = percentile

    def aggregate(self, scores: list[float], weights: list[float] | None = None) -> float:
        if not scores:
            return 0.0
        s = sorted(scores)
        idx = (self.percentile / 100.0) * (len(s) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
        return s[lo] * (1 - (idx - lo)) + s[hi] * (idx - lo)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_STRATEGIES: dict[str, ScoringStrategy] = {
    "weighted_average": WeightedAverageStrategy(),
    "min_gate": MinGateStrategy(),
    "percentile_rank": PercentileRankStrategy(percentile=50.0),
}


# ---------------------------------------------------------------------------
# Scorer facade
# ---------------------------------------------------------------------------

class Scorer:
    """
    Post-processing utility that derives aggregate numbers from the granular
    dicts stored in DimensionResult / AspectResult / EvaluationResult.

    All three aggregate_* methods accept an optional `weights` dict that
    maps names to floats, and an optional `strategy` override.
    """

    def __init__(self, strategy: str = "weighted_average"):
        if strategy not in _STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {list(_STRATEGIES.keys())}"
            )
        self._default_strategy = strategy

    # ── Primitive ────────────────────────────────────────────────────────

    def aggregate(
        self,
        scores: list[float],
        weights: list[float] | None = None,
        strategy: str | None = None,
    ) -> float:
        """Aggregate a plain list of scores."""
        strat = _STRATEGIES[strategy or self._default_strategy]
        return strat.aggregate(scores, weights)

    # ── Dimension (metric scores → one number) ────────────────────────────

    def aggregate_dimension(
        self,
        dim_result: DimensionResult,
        weights: dict[str, float] | None = None,
        strategy: str | None = None,
    ) -> float:
        """
        Roll up all metric scores in a DimensionResult.

        Args:
            dim_result: The DimensionResult to aggregate.
            weights:    {metric_name: weight}. Missing names get weight 1.0.
            strategy:   Override the default strategy for this call.
        """
        items = list(dim_result.scores.items())  # (metric_name, MetricResult)
        if not items:
            return 0.0
        score_list = [r.score for _, r in items]
        weight_list = [
            (weights or {}).get(name, 1.0) for name, _ in items
        ]
        return self.aggregate(score_list, weight_list, strategy)

    # ── Aspect (dimension scores → one number) ────────────────────────────

    def aggregate_aspect(
        self,
        asp_result: AspectResult,
        weights: dict[str, float] | None = None,
        dim_strategy: str | None = None,
        agg_strategy: str | None = None,
    ) -> float:
        """
        Roll up all dimension scores in an AspectResult.

        Args:
            asp_result:   The AspectResult to aggregate.
            weights:      {dimension_name: weight}. Missing names get 1.0.
            dim_strategy: Strategy used when collapsing each dimension.
            agg_strategy: Strategy used to combine the per-dimension numbers.
        """
        items = list(asp_result.scores.items())  # (dim_name, DimensionResult)
        if not items:
            return 0.0
        dim_scores = [
            self.aggregate_dimension(dr, strategy=dim_strategy)
            for _, dr in items
        ]
        weight_list = [
            (weights or {}).get(name, 1.0) for name, _ in items
        ]
        return self.aggregate(dim_scores, weight_list, agg_strategy)

    # ── Evaluation (aspect scores → one number) ───────────────────────────

    def aggregate_evaluation(
        self,
        ev_result: EvaluationResult,
        weights: dict[str, float] | None = None,
        dim_strategy: str | None = None,
        agg_strategy: str | None = None,
    ) -> float:
        """
        Roll up all aspect scores in an EvaluationResult to a single number.

        Args:
            ev_result:    The EvaluationResult to aggregate.
            weights:      {aspect_name: weight}. Missing names get 1.0.
            dim_strategy: Strategy used when collapsing each dimension.
            agg_strategy: Strategy used to combine the per-aspect numbers.
        """
        items = list(ev_result.scores.items())  # (asp_name, AspectResult)
        if not items:
            return 0.0
        asp_scores = [
            self.aggregate_aspect(ar, dim_strategy=dim_strategy, agg_strategy=agg_strategy)
            for _, ar in items
        ]
        weight_list = [
            (weights or {}).get(name, 1.0) for name, _ in items
        ]
        return self.aggregate(asp_scores, weight_list, agg_strategy)

    @staticmethod
    def available_strategies() -> list[str]:
        return list(_STRATEGIES.keys())