"""
Polysemy Score metric (Ambiguity).

Reference:
    Fellbaum, C. (Ed.). (1998). WordNet: An Electronic Lexical Database.
    MIT Press. https://wordnet.princeton.edu/

    Navigli, R. (2009). Word sense disambiguation: A survey.
    ACM Computing Surveys, 41(2), 1–69. https://doi.org/10.1145/1824795.1824799

What it measures:
    Average number of WordNet synsets (senses) per content word.

    A word with many senses (e.g. "bank", "set", "run") is more polysemous
    and thus more susceptible to lexical ambiguity. A high average across the
    question content words indicates a higher potential for misinterpretation.

    Content words: nouns, verbs, adjectives, adverbs (open-class POS tags).
    Function words are excluded (they carry grammatical, not conceptual, meaning).

Score normalisation:
    score = max(0.0, 1.0 − avg_synsets / max_synsets)
    Default max_synsets = 10.0.
    Higher score → fewer senses per word → lower lexical ambiguity.
    flag_below default 0.3 (avg > 7 senses).

Dependency:
    nltk >= 3.8 + wordnet corpus  (already a core requirement)
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
"""

from __future__ import annotations
import re
import logging
from src.models import MetricResult
from src.metrics.base_readability import BaseReadabilityMetric

logger = logging.getLogger(__name__)

import nltk as _nltk

for _res in ("wordnet", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
    try:
        _nltk.download(_res, quiet=True)
    except Exception:
        pass

try:
    from nltk.corpus import wordnet as _wn
    from nltk import pos_tag as _pos_tag, word_tokenize as _word_tokenize
    _NLTK_WN_AVAILABLE = True
except ImportError:
    _wn = None  # type: ignore[assignment]
    _pos_tag = None  # type: ignore[assignment]
    _word_tokenize = None  # type: ignore[assignment]
    _NLTK_WN_AVAILABLE = False

# Penn Treebank POS prefixes for open-class (content) words
_CONTENT_POS = frozenset({"NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG",
                           "VBN", "VBP", "VBZ", "JJ", "JJR", "JJS",
                           "RB", "RBR", "RBS"})

_WN_POS_MAP = {
    "N": _wn.NOUN if _NLTK_WN_AVAILABLE and _wn else None,
    "V": _wn.VERB if _NLTK_WN_AVAILABLE and _wn else None,
    "J": _wn.ADJ  if _NLTK_WN_AVAILABLE and _wn else None,
    "R": _wn.ADV  if _NLTK_WN_AVAILABLE and _wn else None,
}


def _synset_count(word: str, wn_pos) -> int:
    if wn_pos is None:
        return 0
    return len(_wn.synsets(word, pos=wn_pos))


class PolysemyScoreMetric(BaseReadabilityMetric):
    """
    Average WordNet synset count per content word.

    Higher score → fewer senses per word → lower lexical ambiguity.

    Args:
        max_synsets : Average that maps to score 0.0 (default 10.0).
        flag_below  : Score threshold below which flagged=True (default 0.3).
    """

    name = "polysemy_score"
    description = (
        "Average number of WordNet senses per content word "
        "(higher score = fewer senses = lower lexical ambiguity)."
    )

    def __init__(self, max_synsets: float = 10.0, flag_below: float = 0.3):
        if not _NLTK_WN_AVAILABLE:
            raise ImportError(
                "nltk with wordnet corpus is required for PolysemyScoreMetric.\n"
                "pip install nltk && python -c \"import nltk; nltk.download('wordnet')\""
            )
        self.max_synsets = max_synsets
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        tokens = _word_tokenize(text)
        tagged = _pos_tag(tokens)
        counts = []
        for word, tag in tagged:
            if tag not in _CONTENT_POS:
                continue
            wn_pos = _WN_POS_MAP.get(tag[0])
            n = _synset_count(word.lower(), wn_pos)
            if n > 0:
                counts.append(n)
        return sum(counts) / len(counts) if counts else 0.0

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - raw / self.max_synsets)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Avg synsets/content word {raw:.2f} "
                f"(normalised {score:.4f}; max_synsets={self.max_synsets})."
            ),
            flagged=score < self.flag_below,
            metadata={"avg_synsets": raw, "max_synsets": self.max_synsets},
        )
