"""
Cognitive Demand dimension (Pedagogical aspect).

Definition: Level of cognitive processing required to answer the item,
mapped to Bloom's revised taxonomy (Remember → Understand → Apply →
Analyze → Evaluate → Create) and Webb's Depth of Knowledge scale.

Metrics wired:
    Always available (no extra deps):
        bloom_level   — Bloom's revised taxonomy level (1–6) from action verbs
        dok_level     — Webb's Depth of Knowledge level (1–4) from keywords
        hots_keywords — HOTS/LOTS keyword ratio (higher-order vs. lower-order)
"""

from src.models import DimensionName
from src.dimensions.base import BaseDimension

from src.metrics.bloom_level import BloomLevelMetric
from src.metrics.dok_level import DOKLevelMetric
from src.metrics.hots_keywords import HOTSKeywordsMetric


class CognitiveDemandDimension(BaseDimension):
    name = DimensionName.COGNITIVE_DEMAND
    description = (
        "Level of cognitive processing required to answer the item, "
        "mapped to Bloom's revised taxonomy and Webb's Depth of Knowledge."
    )
    metrics = []

    def __init__(self):
        self.metrics = [
            BloomLevelMetric(),
            DOKLevelMetric(),
            HOTSKeywordsMetric(),
        ]
