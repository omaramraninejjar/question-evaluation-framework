"""
Fairness and Ethics aspect.

Definition: Equity and risk properties of items with respect to learners
and contexts.

Dimensions (5):
  - Group Bias, Measurement Invariance, Content Sensitivity,
    Harmful Content Risk, Privacy Risk
"""

from src.models import AspectName
from src.aspects.base import BaseAspect
from src.dimensions.fairness_ethics.group_bias import GroupBiasDimension
from src.dimensions.fairness_ethics.measurement_invariance import MeasurementInvarianceDimension
from src.dimensions.fairness_ethics.content_sensitivity import ContentSensitivityDimension
from src.dimensions.fairness_ethics.harmful_content_risk import HarmfulContentRiskDimension
from src.dimensions.fairness_ethics.privacy_risk import PrivacyRiskDimension


class FairnessEthicsAspect(BaseAspect):
    name = AspectName.FAIRNESS_ETHICS
    description = "Equity and risk properties of items with respect to learners and contexts."
    dimensions = [
        GroupBiasDimension(),
        MeasurementInvarianceDimension(),
        ContentSensitivityDimension(),
        HarmfulContentRiskDimension(),
        PrivacyRiskDimension(),
    ]