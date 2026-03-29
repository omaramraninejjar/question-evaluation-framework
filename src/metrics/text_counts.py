"""
Text count and ratio metrics for readability assessment.

These metrics do not require external dependencies beyond NLTK (already
a core requirement) and the Python standard library.

Metrics provided:
    WordCountMetric       — total word count (normalised by max_words).
    SentenceCountMetric   — total sentence count (normalised by max_sentences).
    AvgWordLengthMetric   — average characters per word (inverted normalisation).
    AvgSyllablesMetric    — average syllables per word (inverted normalisation;
                            uses textstat if available, else a vowel-run heuristic).
    LongWordRatioMetric   — fraction of words with > long_threshold characters
                            (inverted: fewer long words → more accessible).
    TypeTokenRatioMetric  — type–token ratio: unique_words / total_words
                            (a vocabulary diversity measure; higher = more diverse).

Score convention (consistent with all readability metrics):
    Higher score → more accessible / better for the target population.

References:
    Templin, M. C. (1957). Certain language skills in children.
        University of Minnesota Press.  [type–token ratio background]
    Flesch, R. (1948). A new readability yardstick.
        Journal of Applied Psychology, 32(3), 221–233.  [syllable counting context]
    textstat >= 0.7 used for syllable counts when available;
    falls back to a simple vowel-run heuristic otherwise.

Dependency (optional for syllable counting):
    textstat >= 0.7   (pip install textstat)
"""

from __future__ import annotations
import re
import logging
from src.models import Question, EvaluationContext, MetricResult
from src.metrics.base_readability import BaseReadabilityMetric

logger = logging.getLogger(__name__)

try:
    import textstat as _textstat
    _TEXTSTAT_AVAILABLE = True
except ImportError:
    _textstat = None  # type: ignore[assignment]
    _TEXTSTAT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenize_words(text: str) -> list[str]:
    """Return a list of lowercase alphabetic tokens (no punctuation)."""
    return re.findall(r"[a-zA-Z]+", text.lower())


def _count_sentences(text: str) -> int:
    """Count sentences by splitting on '.', '!', '?' (at least 1)."""
    parts = re.split(r"[.!?]+", text.strip())
    return max(1, sum(1 for p in parts if p.strip()))


def _syllable_count_heuristic(word: str) -> int:
    """Simple vowel-run syllable counter (fallback when textstat unavailable)."""
    word = word.lower().rstrip("e")
    count = len(re.findall(r"[aeiou]+", word))
    return max(1, count)


def _syllables_per_word(words: list[str]) -> float:
    """Average syllables per word over the token list."""
    if not words:
        return 0.0
    if _TEXTSTAT_AVAILABLE:
        total = sum(_textstat.syllable_count(w) for w in words)
    else:
        total = sum(_syllable_count_heuristic(w) for w in words)
    return total / len(words)


# ---------------------------------------------------------------------------
# WordCountMetric
# ---------------------------------------------------------------------------

class WordCountMetric(BaseReadabilityMetric):
    """
    Total word count, normalised to [0, 1] by max_words.

    A very short question may indicate insufficient content; a very long one
    may be unnecessarily complex.  The normalisation here treats word count as
    a proxy for question completeness — more words (up to max_words) → higher
    score.  Set max_words to the expected upper bound for your target domain.

    Args:
        max_words  : Word count that maps to score 1.0 (default 50).
        flag_below : Score threshold below which flagged=True (default 0.1,
                     corresponding to ~5 words).
    """

    name = "word_count"
    description = (
        "Word count normalised by max_words "
        "(higher score = more words, up to the cap)."
    )

    def __init__(self, max_words: int = 50, flag_below: float = 0.1):
        self.max_words = max_words
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        return float(len(_tokenize_words(text)))

    def _normalize(self, raw: float) -> float:
        return min(1.0, raw / self.max_words)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"{int(raw)} words "
                f"(normalised {score:.4f}; max_words={self.max_words})."
            ),
            flagged=score < self.flag_below,
            metadata={"word_count": int(raw), "max_words": self.max_words},
        )


# ---------------------------------------------------------------------------
# SentenceCountMetric
# ---------------------------------------------------------------------------

class SentenceCountMetric(BaseReadabilityMetric):
    """
    Total sentence count, normalised to [0, 1] by max_sentences.

    Single-sentence questions (the norm in educational testing) will score
    1/max_sentences.  Adjust max_sentences if multi-sentence items are expected.

    Args:
        max_sentences : Sentence count that maps to score 1.0 (default 5).
        flag_below    : Score threshold below which flagged=True (default 0.0 —
                        sentence count alone rarely flags items).
    """

    name = "sentence_count"
    description = (
        "Sentence count normalised by max_sentences "
        "(higher score = more sentences, up to the cap)."
    )

    def __init__(self, max_sentences: int = 5, flag_below: float = 0.0):
        self.max_sentences = max_sentences
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        return float(_count_sentences(text))

    def _normalize(self, raw: float) -> float:
        return min(1.0, raw / self.max_sentences)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"{int(raw)} sentence(s) "
                f"(normalised {score:.4f}; max_sentences={self.max_sentences})."
            ),
            flagged=score < self.flag_below,
            metadata={"sentence_count": int(raw), "max_sentences": self.max_sentences},
        )


# ---------------------------------------------------------------------------
# AvgWordLengthMetric
# ---------------------------------------------------------------------------

class AvgWordLengthMetric(BaseReadabilityMetric):
    """
    Average number of characters per word, inverted so that shorter words
    (more accessible vocabulary) yield a higher score.

        score = max(0.0, 1.0 − avg_chars / max_avg_chars)

    Args:
        max_avg_chars : Average character count that maps to score 0.0
                        (default 10.0 — words longer than this on average
                        are considered maximally complex).
        flag_below    : Score threshold below which flagged=True (default 0.3).
    """

    name = "avg_word_length"
    description = (
        "Average word length in characters, inverted "
        "(higher score = shorter average word = more accessible vocabulary)."
    )

    def __init__(self, max_avg_chars: float = 10.0, flag_below: float = 0.3):
        self.max_avg_chars = max_avg_chars
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        words = _tokenize_words(text)
        if not words:
            return 0.0
        return sum(len(w) for w in words) / len(words)

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - raw / self.max_avg_chars)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Avg word length {raw:.2f} chars "
                f"(normalised {score:.4f}; max_avg_chars={self.max_avg_chars})."
            ),
            flagged=score < self.flag_below,
            metadata={"avg_word_length_chars": raw, "max_avg_chars": self.max_avg_chars},
        )


# ---------------------------------------------------------------------------
# AvgSyllablesMetric
# ---------------------------------------------------------------------------

class AvgSyllablesMetric(BaseReadabilityMetric):
    """
    Average syllables per word, inverted so that simpler words yield a higher
    score.

        score = max(0.0, 1.0 − avg_syllables / max_avg_syllables)

    Uses textstat.syllable_count() when available; falls back to a vowel-run
    heuristic otherwise.

    Args:
        max_avg_syllables : Average that maps to score 0.0 (default 4.0).
        flag_below        : Score threshold below which flagged=True (default 0.3).
    """

    name = "avg_syllables"
    description = (
        "Average syllables per word, inverted "
        "(higher score = fewer syllables per word = simpler vocabulary)."
    )

    def __init__(self, max_avg_syllables: float = 4.0, flag_below: float = 0.3):
        self.max_avg_syllables = max_avg_syllables
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        return _syllables_per_word(_tokenize_words(text))

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - raw / self.max_avg_syllables)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Avg syllables/word {raw:.2f} "
                f"(normalised {score:.4f}; max={self.max_avg_syllables})."
            ),
            flagged=score < self.flag_below,
            metadata={
                "avg_syllables_per_word": raw,
                "max_avg_syllables": self.max_avg_syllables,
                "syllable_counter": "textstat" if _TEXTSTAT_AVAILABLE else "heuristic",
            },
        )


# ---------------------------------------------------------------------------
# LongWordRatioMetric
# ---------------------------------------------------------------------------

class LongWordRatioMetric(BaseReadabilityMetric):
    """
    Fraction of words exceeding long_threshold characters, inverted.

        score = max(0.0, 1.0 − ratio)

    A higher proportion of long words signals more complex vocabulary.

    Args:
        long_threshold : Character count above which a word is 'long'
                         (default 6, i.e. words with ≥ 7 chars are 'long').
        flag_below     : Score threshold below which flagged=True (default 0.5).
    """

    name = "long_word_ratio"
    description = (
        "Fraction of long words (> long_threshold chars), inverted "
        "(higher score = fewer long words = more accessible vocabulary)."
    )

    def __init__(self, long_threshold: int = 6, flag_below: float = 0.5):
        self.long_threshold = long_threshold
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        words = _tokenize_words(text)
        if not words:
            return 0.0
        long_count = sum(1 for w in words if len(w) > self.long_threshold)
        return long_count / len(words)

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - raw)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Long-word ratio {raw:.4f} "
                f"(normalised {score:.4f}; threshold>{self.long_threshold} chars)."
            ),
            flagged=score < self.flag_below,
            metadata={
                "long_word_ratio": raw,
                "long_threshold_chars": self.long_threshold,
            },
        )


# ---------------------------------------------------------------------------
# TypeTokenRatioMetric
# ---------------------------------------------------------------------------

class TypeTokenRatioMetric(BaseReadabilityMetric):
    """
    Type–Token Ratio (TTR): unique_words / total_words.

    TTR measures lexical diversity.  A score of 1.0 means every word is unique
    (maximal diversity); lower values indicate repetition.  Short questions
    naturally tend toward higher TTR.

    This metric is NOT inverted — higher TTR is treated as better (richer
    vocabulary diversity in the question).

    Args:
        flag_below : Score threshold below which flagged=True (default 0.4).
    """

    name = "type_token_ratio"
    description = (
        "Type–Token Ratio (unique / total words): lexical diversity "
        "(higher = more diverse vocabulary)."
    )

    def __init__(self, flag_below: float = 0.4):
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        words = _tokenize_words(text)
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    def _normalize(self, raw: float) -> float:
        return raw  # already in [0, 1]

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Type–Token Ratio {raw:.4f} "
                f"(unique/total words; score={score:.4f})."
            ),
            flagged=score < self.flag_below,
            metadata={"type_token_ratio": raw},
        )
