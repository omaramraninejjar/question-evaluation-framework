"""
Content Sensitivity dimension (Fairness & Ethics aspect).

Definition: Potential for item content to be culturally inappropriate,
distressing, or exclusionary for particular groups of respondents.
Sensitive content may affect motivation, anxiety, and performance in
ways unrelated to the target construct.
"""

from src.models import DimensionName
from src.dimensions.base import BaseDimension


class ContentSensitivityDimension(BaseDimension):
    name = DimensionName.CONTENT_SENSITIVITY
    description = (
        "Potential for item content to be culturally inappropriate, "
        "distressing, or exclusionary for particular respondent groups."
    )
    metrics = []           # Populate once metrics are defined
