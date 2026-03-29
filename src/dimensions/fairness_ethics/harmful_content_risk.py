"""
Harmful Content Risk dimension (Fairness & Ethics aspect).

Definition: Risk that the item contains, promotes, or could elicit
harmful, offensive, dangerous, or legally problematic content. This
includes misinformation, hate speech, instructions for harmful acts,
and content inappropriate for the target age group.
"""

from src.models import DimensionName
from src.dimensions.base import BaseDimension


class HarmfulContentRiskDimension(BaseDimension):
    name = DimensionName.HARMFUL_CONTENT_RISK
    description = (
        "Risk that the item contains or could elicit harmful, offensive, "
        "dangerous, or age-inappropriate content."
    )
    metrics = []           # Populate once metrics are defined
