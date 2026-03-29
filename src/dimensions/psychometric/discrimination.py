"""
Discrimination dimension (Psychometric aspect).

Definition: Ability of the item to differentiate between respondents with
high versus low levels of the target construct. In CTT this is the
point-biserial correlation; in IRT it corresponds to the a-parameter.
"""

from src.models import DimensionName
from src.dimensions.base import BaseDimension


class DiscriminationDimension(BaseDimension):
    name = DimensionName.DISCRIMINATION
    description = (
        "Ability of the item to differentiate between high- and low-ability "
        "respondents (CTT point-biserial or IRT a-parameter)."
    )
    metrics = []           # Populate once metrics are defined
