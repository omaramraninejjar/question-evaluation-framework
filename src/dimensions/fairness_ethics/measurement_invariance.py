"""
Measurement Invariance dimension (Fairness & Ethics aspect).

Definition: Consistency of the item's psychometric properties (factor
loadings, intercepts, difficulty) across distinct subpopulations.
Violations mean that latent trait scores are not comparable across groups,
undermining the fairness of any score-based decisions.
"""

from src.models import DimensionName
from src.dimensions.base import BaseDimension


class MeasurementInvarianceDimension(BaseDimension):
    name = DimensionName.MEASUREMENT_INVARIANCE
    description = (
        "Consistency of the item's psychometric properties across "
        "subpopulations, ensuring latent trait scores are comparable."
    )
    metrics = []           # Populate once metrics are defined
