"""
Readability dimension (Linguistic & Structural aspect).

Definition: Ease with which a respondent of the target population can
read and comprehend the item text. Operationalised through formula-based
indices (Flesch-Kincaid, Dale-Chall, etc.) and target grade-level norms.

Metrics wired:
    Always available (no extra deps):
        word_count, sentence_count, avg_word_length, avg_syllables,
        long_word_ratio, type_token_ratio

    Always available (nltk):
        lexical_density

    Optional — require textstat >= 0.7 (pip install textstat):
        flesch_ease, flesch_kincaid, dale_chall, gunning_fog,
        coleman_liau, ari, smog, linsear_write, spache_score
"""

from src.models import DimensionName
from src.dimensions.base import BaseDimension

# Count metrics — always available
from src.metrics.text_counts import (
    WordCountMetric,
    SentenceCountMetric,
    AvgWordLengthMetric,
    AvgSyllablesMetric,
    LongWordRatioMetric,
    TypeTokenRatioMetric,
)

# NLTK-based readability metrics — always available
from src.metrics.lexical_density import LexicalDensityMetric

# Textstat formula metrics — optional
from src.metrics.flesch_ease import FleschReadingEaseMetric, _TEXTSTAT_AVAILABLE
from src.metrics.flesch_kincaid import FleschKincaidMetric
from src.metrics.dale_chall import DaleChallMetric
from src.metrics.gunning_fog import GunningFogMetric
from src.metrics.coleman_liau import ColemanLiauMetric
from src.metrics.ari import ARIMetric
from src.metrics.smog import SMOGMetric
from src.metrics.linsear_write import LinsearWriteMetric
from src.metrics.spache import SpacheScoreMetric


class ReadabilityDimension(BaseDimension):
    name = DimensionName.READABILITY
    description = (
        "Ease with which a respondent of the target population can read and "
        "comprehend the item text, relative to grade-level norms."
    )
    metrics = []  # instance-level list built in __init__

    def __init__(self):
        # Count-based metrics — always loaded
        _metrics = [
            WordCountMetric(),
            SentenceCountMetric(),
            AvgWordLengthMetric(),
            AvgSyllablesMetric(),
            LongWordRatioMetric(),
            TypeTokenRatioMetric(),
            LexicalDensityMetric(),
        ]

        # Textstat formula metrics — loaded only when textstat is installed
        if _TEXTSTAT_AVAILABLE:
            _metrics += [
                FleschReadingEaseMetric(),
                FleschKincaidMetric(),
                DaleChallMetric(),
                GunningFogMetric(),
                ColemanLiauMetric(),
                ARIMetric(),
                SMOGMetric(),
                LinsearWriteMetric(),
                SpacheScoreMetric(),
            ]

        self.metrics = _metrics
