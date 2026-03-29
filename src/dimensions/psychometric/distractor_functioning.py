"""
Distractor Functioning dimension (Psychometric aspect).

Definition: Effectiveness of each incorrect answer option (distractor) in
attracting respondents with lower ability. A well-functioning distractor
is chosen more often by low scorers than high scorers and contributes to
item discrimination.
"""

from src.models import DimensionName
from src.dimensions.base import BaseDimension


class DistractorFunctioningDimension(BaseDimension):
    name = DimensionName.DISTRACTOR_FUNCTIONING
    description = (
        "Effectiveness of incorrect answer options in attracting lower-ability "
        "respondents, contributing to item discrimination."
    )
    metrics = []           # Populate once metrics are defined
