"""
Linguistic and Structural aspect.

Definition: Quality of item wording and format with respect to clarity
and interpretability.

Dimensions (5):
  - Readability, Linguistic Complexity, Ambiguity, Well-formedness, Diversity
"""

from src.models import AspectName
from src.aspects.base import BaseAspect
from src.dimensions.linguistic_structural.readability import ReadabilityDimension
from src.dimensions.linguistic_structural.linguistic_complexity import LinguisticComplexityDimension
from src.dimensions.linguistic_structural.ambiguity import AmbiguityDimension
from src.dimensions.linguistic_structural.well_formedness import WellFormednessDimension
from src.dimensions.linguistic_structural.diversity import DiversityDimension


class LinguisticStructuralAspect(BaseAspect):
    name = AspectName.LINGUISTIC_STRUCTURAL
    description = (
        "Quality of item wording and format with respect to clarity and interpretability."
    )
    dimensions = [
        ReadabilityDimension(),
        LinguisticComplexityDimension(),
        AmbiguityDimension(),
        WellFormednessDimension(),
        DiversityDimension(),
    ]