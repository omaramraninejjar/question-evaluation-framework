"""
Item Fit dimension (Psychometric aspect).

Definition: Degree to which observed item-response patterns match the
predictions of a psychometric model (e.g., Rasch, 2PL IRT). Poor fit
indicates that the item behaves unexpectedly relative to the modelled
construct, often signalled by infit/outfit mean-square statistics.
"""

from src.models import DimensionName
from src.dimensions.base import BaseDimension


class ItemFitDimension(BaseDimension):
    name = DimensionName.ITEM_FIT
    description = (
        "Degree to which observed response patterns match psychometric model "
        "predictions (e.g., Rasch infit/outfit statistics)."
    )
    metrics = []           # Populate once metrics are defined
