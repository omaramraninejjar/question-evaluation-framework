"""
Stem Complexity metric (Response Burden).

Reference:
    Tourangeau, R., Rips, L. J., & Rasinski, K. (2000). The Psychology of
    Survey Response. Cambridge University Press. Chapter 3: Comprehension.

    Graesser, A. C., Cai, Z., Louwerse, M. M., & Daniel, F. (2006).
    Question Understanding Aid (QUAID). Public Opinion Quarterly, 70(1), 3–22.

    Haladyna, T. M. (2004). Developing and Validating Multiple-Choice Test
    Items (3rd ed.). Lawrence Erlbaum. Chapter 6: Item stem guidelines.

What it measures:
    Cognitive burden imposed by the structural complexity of the question
    stem, independently of its linguistic readability or vocabulary level.

    Composite of three sub-indicators:

        1. Stem word count — longer stems impose more processing load
           (Graesser et al. 2006: >25 words in a question stem is excessive)
        2. Preamble length — word count BEFORE the first WH-word or
           auxiliary verb; longer preambles delay the respondent in
           identifying what is being asked
        3. Embedded clause count — approximate count of subordinate clauses
           detected by commas + subordinating conjunctions; each clause
           introduces a parsing step

    composite = (w_words * word_count + w_preamble * preamble_len
                  + w_clauses * clause_count) / normaliser

Score:
    score = max(0.0, 1.0 − composite / max_composite)
    Default max_composite = 1.0; individual sub-scores are normalised first.
    Higher score → simpler stem → lower processing burden.
    flag_below default 0.3.

Dependency:
    nltk >= 3.8   (already a core requirement)
"""

from __future__ import annotations
import re
import logging
from src.models import MetricResult
from src.metrics.base_readability import BaseReadabilityMetric

logger = logging.getLogger(__name__)

import nltk as _nltk
for _res in ("punkt", "punkt_tab"):
    try:
        _nltk.download(_res, quiet=True)
    except Exception:
        pass

from nltk import word_tokenize as _word_tokenize

_WH_AUX = re.compile(
    r"\b(what|who|which|when|where|why|how|is|are|was|were|do|does|did|"
    r"can|could|will|would|should|has|have)\b",
    re.IGNORECASE,
)

_SUBORD_CONJ = re.compile(
    r"\b(although|because|since|while|if|unless|until|after|before|as|"
    r"though|whenever|wherever|once|whereas|whether)\b",
    re.IGNORECASE,
)

# Normalisation constants
_MAX_WORDS = 40       # >40 words = maximum word-count burden
_MAX_PREAMBLE = 20    # >20 words before the question = excessive preamble
_MAX_CLAUSES = 4      # >4 subordinate clauses = excessive structure

# Sub-score weights (sum = 1.0)
_W_WORDS = 0.4
_W_PREAMBLE = 0.35
_W_CLAUSES = 0.25


def _stem_features(text: str) -> tuple[int, int, int]:
    """Return (word_count, preamble_len, clause_count)."""
    try:
        tokens = _word_tokenize(text)
    except Exception:
        tokens = text.split()
    words = [t for t in tokens if t.isalpha()]
    word_count = len(words)

    # Preamble: words before first WH-word / auxiliary
    match = _WH_AUX.search(text)
    preamble_len = len(re.findall(r"[a-zA-Z]+", text[:match.start()])) if match else word_count

    # Clause count: subordinating conjunctions + comma-separated phrases
    clause_count = len(_SUBORD_CONJ.findall(text)) + max(0, text.count(",") - 1)

    return word_count, preamble_len, clause_count


class StemComplexityMetric(BaseReadabilityMetric):
    """
    Stem complexity composite: word count + preamble length + clause count.

    Higher score → simpler stem → lower processing burden.

    Args:
        max_words     : Word count mapped to score 0.0 on that sub-dimension (default 40).
        max_preamble  : Preamble length mapped to score 0.0 (default 20).
        max_clauses   : Clause count mapped to score 0.0 (default 4).
        flag_below    : Score threshold below which flagged=True (default 0.3).
    """

    name = "stem_complexity"
    description = (
        "Stem complexity composite (word count + preamble + embedded clauses) "
        "(higher score = simpler stem = lower processing burden)."
    )

    def __init__(
        self,
        max_words: int = 40,
        max_preamble: int = 20,
        max_clauses: int = 4,
        flag_below: float = 0.3,
    ):
        self.max_words = max_words
        self.max_preamble = max_preamble
        self.max_clauses = max_clauses
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        wc, pre, cl = _stem_features(text)
        s_wc = min(1.0, wc / self.max_words)
        s_pre = min(1.0, pre / self.max_preamble)
        s_cl = min(1.0, cl / self.max_clauses)
        return _W_WORDS * s_wc + _W_PREAMBLE * s_pre + _W_CLAUSES * s_cl

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - raw)

    def compute(self, question, context):
        if not question.text.strip():
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                rationale="Question text is empty.",
                flagged=True,
                metadata={"reason": "empty_text"},
            )
        wc, pre, cl = _stem_features(question.text)
        raw = self._compute_raw(question.text)
        score = self._normalize(raw)
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Stem: {wc} words, {pre}-word preamble, {cl} embedded clause(s) "
                f"(complexity={raw:.4f}; score={score:.4f})."
            ),
            flagged=score < self.flag_below,
            metadata={
                "word_count": wc,
                "preamble_length": pre,
                "clause_count": cl,
                "complexity_composite": raw,
            },
        )

    def _build_result(self, score: float, raw: float) -> MetricResult:  # pragma: no cover
        raise NotImplementedError
