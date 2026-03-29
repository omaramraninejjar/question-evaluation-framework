"""
Zipf Word Frequency metric (Linguistic Complexity).

Reference:
    van Heuven, W. J. B., Mandera, P., Keuleers, E., & Brysbaert, M. (2014).
    SUBTLEX-UK: A new and improved word frequency database for British English.
    Quarterly Journal of Experimental Psychology, 67(6), 1176–1190.
    https://doi.org/10.1080/17470218.2013.850521

    Brysbaert, M., & New, B. (2009). Moving beyond Kucera and Francis: A critical
    evaluation of current word frequency norms and the introduction of a new and
    improved word frequency measure for American English.
    Behavior Research Methods, 41(4), 977–990.
    https://doi.org/10.3758/BRM.41.4.977

What it measures:
    Average Zipf frequency of content words in the question text.

    Zipf scale (van Heuven et al., 2014):
        1–2  extremely rare (< 1 per million)
        3    rare (1–10 per million)
        4    medium frequency (common English)
        5–6  high frequency (function words, everyday vocabulary)
        7    very high frequency (the, a, is, …)

    Lower average Zipf → rarer vocabulary → higher linguistic complexity.

Score normalisation:
    score = avg_zipf / max_zipf
    Default max_zipf = 7.0.
    Higher score → more common vocabulary → less complex.
    flag_below default 0.4 (~average Zipf < 2.8, i.e. many rare words).

Dependency:
    wordfreq >= 2.5   (pip install wordfreq)
"""

from __future__ import annotations
import re
import logging
from src.models import MetricResult
from src.metrics.base_readability import BaseReadabilityMetric

logger = logging.getLogger(__name__)

try:
    from wordfreq import zipf_frequency as _zipf_frequency
    _WORDFREQ_AVAILABLE = True
except ImportError:
    _zipf_frequency = None  # type: ignore[assignment]
    _WORDFREQ_AVAILABLE = False

# Closed-class words to skip (focus on content-word complexity)
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "that",
    "this", "these", "those", "it", "its", "i", "you", "he", "she", "we",
    "they", "what", "which", "who", "whom", "how", "why", "when", "where",
    "and", "but", "or", "nor", "not", "no", "so", "yet",
})


class ZipfWordFrequencyMetric(BaseReadabilityMetric):
    """
    Average Zipf word frequency of content words.

    Higher score → more common vocabulary → lower linguistic complexity.

    Args:
        max_zipf   : Zipf value that maps to score 1.0 (default 7.0).
        lang       : Language code passed to wordfreq (default "en").
        flag_below : Score threshold below which flagged=True (default 0.4).
    """

    name = "zipf_word_frequency"
    description = (
        "Average Zipf frequency of content words "
        "(higher = more common vocabulary = lower complexity)."
    )

    def __init__(self, max_zipf: float = 7.0, lang: str = "en", flag_below: float = 0.4):
        if not _WORDFREQ_AVAILABLE:
            raise ImportError(
                "wordfreq is required for ZipfWordFrequencyMetric.\n"
                "pip install wordfreq"
            )
        self.max_zipf = max_zipf
        self.lang = lang
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        words = [w.lower() for w in re.findall(r"[a-zA-Z]+", text)
                 if w.lower() not in _STOP_WORDS]
        if not words:
            # Fall back to all words if text is entirely stop words
            words = [w.lower() for w in re.findall(r"[a-zA-Z]+", text)]
        if not words:
            return 0.0
        freqs = [_zipf_frequency(w, self.lang) for w in words]
        return sum(freqs) / len(freqs)

    def _normalize(self, raw: float) -> float:
        return min(1.0, max(0.0, raw / self.max_zipf))

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Avg Zipf frequency {raw:.2f} "
                f"(normalised {score:.4f}; max_zipf={self.max_zipf})."
            ),
            flagged=score < self.flag_below,
            metadata={"avg_zipf": raw, "max_zipf": self.max_zipf, "lang": self.lang},
        )
