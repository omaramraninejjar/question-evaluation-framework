"""
Pedagogical aspect.

Definition: Educational quality of an item with respect to intended learning
goals and the thinking it is designed to elicit.

Dimensions (4):
  - Curriculum Alignment
  - Cognitive Demand
  - Concept Coverage
  - Response Burden
"""

from src.models import AspectName
from src.aspects.base import BaseAspect
from src.dimensions.pedagogical.curriculum_alignment import CurriculumAlignmentDimension
from src.dimensions.pedagogical.cognitive_demand import CognitiveDemandDimension
from src.dimensions.pedagogical.concept_coverage import ConceptCoverageDimension
from src.dimensions.pedagogical.response_burden import ResponseBurdenDimension


class PedagogicalAspect(BaseAspect):
    name = AspectName.PEDAGOGICAL
    description = (
        "Educational quality of an item with respect to intended learning goals "
        "and the thinking it is designed to elicit."
    )
    dimensions = [
        CurriculumAlignmentDimension(),
        CognitiveDemandDimension(),
        ConceptCoverageDimension(),
        ResponseBurdenDimension(),
    ]