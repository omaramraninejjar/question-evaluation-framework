"""
Linguistic Complexity dimension (Linguistic & Structural aspect).

Definition: Syntactic and lexical complexity of the item text, including
sentence length, subordinate clause depth, passive constructions, and
low-frequency vocabulary. Complexity beyond what the construct requires
introduces construct-irrelevant difficulty.

Metrics wired:
    Always available (nltk):
        conjunction_rate

    Optional — require wordfreq >= 2.5 (pip install wordfreq):
        zipf_word_frequency

    Optional — require spacy >= 3.0 + en_core_web_sm model:
        parse_tree_depth, dependency_distance, constituent_count,
        passive_voice_rate
"""

from src.models import DimensionName
from src.dimensions.base import BaseDimension

from src.metrics.conjunction_rate import ConjunctionRateMetric
from src.metrics.zipf_frequency import ZipfWordFrequencyMetric, _WORDFREQ_AVAILABLE
from src.metrics.parse_tree_depth import ParseTreeDepthMetric, _SPACY_AVAILABLE
from src.metrics.dependency_distance import DependencyDistanceMetric
from src.metrics.constituent_count import ConstituentCountMetric
from src.metrics.passive_voice_rate import PassiveVoiceRateMetric


class LinguisticComplexityDimension(BaseDimension):
    name = DimensionName.LINGUISTIC_COMPLEXITY
    description = (
        "Syntactic and lexical complexity of the item text beyond what "
        "the target construct requires."
    )
    metrics = []

    def __init__(self):
        # NLTK-based — always available
        _metrics = [
            ConjunctionRateMetric(),
        ]

        if _WORDFREQ_AVAILABLE:
            _metrics.append(ZipfWordFrequencyMetric())

        if _SPACY_AVAILABLE:
            _metrics.append(ParseTreeDepthMetric())
            _metrics.append(DependencyDistanceMetric())
            _metrics.append(ConstituentCountMetric())
            _metrics.append(PassiveVoiceRateMetric())

        self.metrics = _metrics
