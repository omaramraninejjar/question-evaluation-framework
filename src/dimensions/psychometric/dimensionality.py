"""
Dimensionality dimension (Psychometric aspect).

Definition: Extent to which the item measures a single, well-defined
latent construct (unidimensionality). Violations — assessed via
exploratory/confirmatory factor analysis or residual analysis — suggest
the item taps multiple constructs, threatening score interpretability.
"""

from src.models import DimensionName
from src.dimensions.base import BaseDimension


class DimensionalityDimension(BaseDimension):
    name = DimensionName.DIMENSIONALITY
    description = (
        "Extent to which the item measures a single latent construct "
        "(unidimensionality), assessed via factor or residual analysis."
    )
    metrics = []           # Populate once metrics are defined
