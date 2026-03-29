"""
Difficulty dimension (Psychometric aspect).

Definition: Statistical likelihood that a respondent of average ability
answers the item correctly. In classical test theory this is the p-value
(proportion correct); in IRT it corresponds to the b-parameter (location
on the latent trait scale).
"""

from src.models import DimensionName
from src.dimensions.base import BaseDimension


class DifficultyDimension(BaseDimension):
    name = DimensionName.DIFFICULTY
    description = (
        "Statistical likelihood that a respondent of average ability answers "
        "the item correctly (CTT p-value or IRT b-parameter)."
    )
    metrics = []           # Populate once metrics are defined
