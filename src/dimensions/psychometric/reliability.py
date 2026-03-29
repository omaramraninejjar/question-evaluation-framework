"""
Reliability dimension (Psychometric aspect).

Definition: Consistency of measurement provided by the item across
repeated administrations, parallel forms, or internal consistency
analysis. Estimated via item-level contributions to Cronbach's alpha,
test-retest correlation, or IRT information at the item level.
"""

from src.models import DimensionName
from src.dimensions.base import BaseDimension


class ReliabilityDimension(BaseDimension):
    name = DimensionName.RELIABILITY
    description = (
        "Consistency of measurement provided by the item, estimated via "
        "item-level contributions to alpha or IRT information."
    )
    metrics = []           # Populate once metrics are defined
