"""
Has Verb metric (Well-Formedness).

Reference:
    Haladyna, T. M., Downing, S. M., & Rodriguez, M. C. (2002).
    A review of multiple-choice item-writing guidelines for classroom assessment.
    Applied Measurement in Education, 15(3), 309–334.
    https://doi.org/10.1207/S15324818AME1503_5

What it measures:
    Binary indicator: does the question contain at least one verb?

    A syntactically complete question must contain a predicate (verb phrase).
    Questions lacking a finite verb may be:
        - Noun-phrase fragments: "The mitochondria?"
        - Incomplete stems: "Photosynthesis in plants:"
        - Copy-paste or generation artefacts

    Verb detection uses NLTK POS tagging; any VB* tag counts (base, past,
    gerund, participle, present, past-participle, 3rd-person singular).

Score:
    1.0 if at least one verb token is found
    0.0 if no verb is found

Higher score → question has a predicate → more structurally well-formed.

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
for _res in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "punkt",
             "punkt_tab"):
    try:
        _nltk.download(_res, quiet=True)
    except Exception:
        pass

from nltk import pos_tag as _pos_tag, word_tokenize as _word_tokenize

_VERB_TAGS = frozenset({"VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"})


class HasVerbMetric(BaseReadabilityMetric):
    """
    Binary check: does the question contain at least one verb?

    Score 1.0 → verb present; 0.0 → no verb detected.

    Args:
        flag_below : Score threshold below which flagged=True (default 0.5,
                     so absence of any verb always triggers a flag).
    """

    name = "has_verb"
    description = (
        "Binary: 1.0 if question contains at least one verb (VB*), else 0.0 "
        "(predicate completeness well-formedness)."
    )

    def __init__(self, flag_below: float = 0.5):
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        tokens = _word_tokenize(text)
        words = [t for t in tokens if t.isalpha()]
        if not words:
            return 0.0
        tagged = _pos_tag(words)
        return 1.0 if any(tag in _VERB_TAGS for _, tag in tagged) else 0.0

    def _normalize(self, raw: float) -> float:
        return raw

    def _build_result(self, score: float, raw: float) -> MetricResult:
        has_verb = bool(raw)
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                "At least one verb detected — question has a predicate."
                if has_verb
                else "No verb detected — question may be a fragment."
            ),
            flagged=score < self.flag_below,
            metadata={"has_verb": has_verb},
        )
