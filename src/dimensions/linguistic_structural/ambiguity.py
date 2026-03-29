"""
Ambiguity dimension (Linguistic & Structural aspect).

Definition: Degree to which the item stem, options, or instructions
permit multiple valid interpretations, making it unclear what is being
asked or what constitutes a correct answer. Ambiguity reduces validity
and increases construct-irrelevant variance.

Metrics wired:
    Always available (no extra deps):
        wh_word_type, pronoun_ratio, quantifier_rate

    Optional — require nltk wordnet corpus:
        polysemy_score

    Optional — require spacy >= 3.0 + en_core_web_sm model:
        negation_ambiguity
"""

from src.models import DimensionName
from src.dimensions.base import BaseDimension

from src.metrics.wh_word_type import WHWordTypeMetric
from src.metrics.pronoun_ratio import PronounRatioMetric
from src.metrics.quantifier_rate import QuantifierRateMetric
from src.metrics.polysemy import PolysemyScoreMetric, _NLTK_WN_AVAILABLE
from src.metrics.negation_ambiguity import NegationAmbiguityScoreMetric, _SPACY_AVAILABLE


class AmbiguityDimension(BaseDimension):
    name = DimensionName.AMBIGUITY
    description = (
        "Degree to which the item permits multiple valid interpretations, "
        "obscuring what is being asked or what constitutes a correct answer."
    )
    metrics = []

    def __init__(self):
        _metrics = [
            WHWordTypeMetric(),
            PronounRatioMetric(),
            QuantifierRateMetric(),
        ]

        if _NLTK_WN_AVAILABLE:
            _metrics.append(PolysemyScoreMetric())

        if _SPACY_AVAILABLE:
            _metrics.append(NegationAmbiguityScoreMetric())

        self.metrics = _metrics
