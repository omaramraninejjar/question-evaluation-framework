"""
Group Bias dimension (Fairness & Ethics aspect).

Definition: Differential Item Functioning (DIF) — the extent to which
the item performs differently across demographic groups (e.g., gender,
ethnicity, socioeconomic status) after controlling for overall ability.
DIF indicates that the item measures something beyond the target construct
for certain groups.
"""

from src.models import DimensionName
from src.dimensions.base import BaseDimension


class GroupBiasDimension(BaseDimension):
    name = DimensionName.GROUP_BIAS
    description = (
        "Differential Item Functioning (DIF): extent to which the item "
        "performs differently across demographic groups after controlling "
        "for ability."
    )
    metrics = []           # Populate once metrics are defined
