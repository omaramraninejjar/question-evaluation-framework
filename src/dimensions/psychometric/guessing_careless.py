"""
Guessing & Careless Errors dimension (Psychometric aspect).

Definition: Probability that a low-ability respondent answers correctly
by guessing (c-parameter in 3PL IRT) or that a high-ability respondent
answers incorrectly due to careless error (d-parameter in 4PL IRT).
Both introduce noise that attenuates validity.
"""

from src.models import DimensionName
from src.dimensions.base import BaseDimension


class GuessingCarelessDimension(BaseDimension):
    name = DimensionName.GUESSING_CARELESS
    description = (
        "Probability of pseudo-guessing by low-ability respondents or "
        "careless errors by high-ability respondents (IRT c/d parameters)."
    )
    metrics = []           # Populate once metrics are defined
