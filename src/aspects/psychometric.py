"""
Psychometric aspect.

Definition: Measurement quality of an item as part of an assessment instrument.

Dimensions (7):
  - Difficulty, Discrimination, Guessing and Careless Responding,
    Distractor Functioning, Item Fit, Dimensionality, Reliability
"""

from src.models import AspectName
from src.aspects.base import BaseAspect
from src.dimensions.psychometric.difficulty import DifficultyDimension
from src.dimensions.psychometric.discrimination import DiscriminationDimension
from src.dimensions.psychometric.guessing_careless import GuessingCarelessDimension
from src.dimensions.psychometric.distractor_functioning import DistractorFunctioningDimension
from src.dimensions.psychometric.item_fit import ItemFitDimension
from src.dimensions.psychometric.dimensionality import DimensionalityDimension
from src.dimensions.psychometric.reliability import ReliabilityDimension


class PsychometricAspect(BaseAspect):
    name = AspectName.PSYCHOMETRIC
    description = "Measurement quality of an item as part of an assessment instrument."
    dimensions = [
        DifficultyDimension(),
        DiscriminationDimension(),
        GuessingCarelessDimension(),
        DistractorFunctioningDimension(),
        ItemFitDimension(),
        DimensionalityDimension(),
        ReliabilityDimension(),
    ]