"""
Lexical Density metric (Readability).

Reference:
    Ure, J. (1971). Lexical density and register differentiation. In
    G. Perren & J. L. M. Trim (Eds.), Applications of Linguistics
    (pp. 443–452). Cambridge University Press.

    Halliday, M. A. K. (1985). Spoken and Written Language. Deakin
    University Press.

What it measures:
    Ratio of content words (open-class: nouns, verbs, adjectives, adverbs)
    to total word tokens.

        lexical_density = content_words / total_words

    A higher density indicates an information-dense, substantive question —
    typical of well-formed educational items. A very low density suggests
    the question is dominated by function words and may lack specificity.

Score:
    score = lexical_density ∈ [0.0, 1.0] (no inversion — higher is better)
    flag_below default 0.3 (fewer than 30 % content words is unusually sparse).

Dependency:
    nltk >= 3.8   (already a core requirement)
    nltk.download('averaged_perceptron_tagger')
"""

from __future__ import annotations
import logging
from src.models import MetricResult
from src.metrics.base_readability import BaseReadabilityMetric

logger = logging.getLogger(__name__)

import nltk as _nltk
for _res in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
             "punkt", "punkt_tab"):
    try:
        _nltk.download(_res, quiet=True)
    except Exception:
        pass

from nltk import pos_tag as _pos_tag, word_tokenize as _word_tokenize

# Penn Treebank open-class POS prefixes
_CONTENT_PREFIXES = ("NN", "VB", "JJ", "RB")


def _is_content(tag: str) -> bool:
    return tag.startswith(_CONTENT_PREFIXES)


class LexicalDensityMetric(BaseReadabilityMetric):
    """
    Lexical density: fraction of word tokens that are content words.

    Higher score → more content words → more information-dense question.

    Args:
        flag_below : Score threshold below which flagged=True (default 0.3).
    """

    name = "lexical_density"
    description = (
        "Lexical density (content words / total words) "
        "(higher score = more content words = more informative question)."
    )

    def __init__(self, flag_below: float = 0.3):
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        tokens = _word_tokenize(text)
        words = [t for t in tokens if t.isalpha()]
        if not words:
            return 0.0
        tagged = _pos_tag(words)
        n_content = sum(1 for _, tag in tagged if _is_content(tag))
        return n_content / len(words)

    def _normalize(self, raw: float) -> float:
        return raw  # already in [0, 1]

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Lexical density {raw:.4f} "
                f"({score:.4f} content-word ratio)."
            ),
            flagged=score < self.flag_below,
            metadata={"lexical_density": raw},
        )
