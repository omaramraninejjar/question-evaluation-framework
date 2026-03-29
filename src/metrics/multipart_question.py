"""
Multi-Part Question metric (Response Burden).

Reference:
    Haladyna, T. M., Downing, S. M., & Rodriguez, M. C. (2002). A review
    of multiple-choice item-writing guidelines for classroom assessment.
    Applied Measurement in Education, 15(3), 309–333.
    https://doi.org/10.1207/S15324818AME1503_5

    Tourangeau, R., Rips, L. J., & Rasinski, K. (2000). The Psychology of
    Survey Response. Cambridge University Press. Chapter 3: Response burden.

What it measures:
    Number of distinct sub-tasks embedded in a single question stem.

    Multi-part questions ("Explain X AND describe Y AND provide Z") impose
    compounded cognitive and response burden that is independent of the
    difficulty of each individual part. Standards for educational and
    psychological testing (AERA/APA/NCME, 2014) recommend single-focus items.

    Detection heuristics:
      1. Coordinating conjunctions (CC) between imperative verbs
         (e.g., "Analyze A AND compare B")
      2. Enumeration patterns ("first … second … third", "a) … b)")
      3. Multiple question marks within the stem
      4. Explicit part markers ("Part A", "(i)", "1.")

    part_count = max(1, heuristic_estimate)

Score:
    score = max(0.0, 1.0 − (part_count − 1) / max_extra_parts)
    Default max_extra_parts = 2 (i.e., 3-part question → score 0.0).
    Higher score → fewer sub-parts → lower structural response burden.
    flag_below default 0.5 (flags items with 2+ parts).

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
for _res in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
             "punkt", "punkt_tab"):
    try:
        _nltk.download(_res, quiet=True)
    except Exception:
        pass

from nltk import pos_tag as _pos_tag, word_tokenize as _word_tokenize

# Regex patterns for explicit part markers
_PART_MARKER = re.compile(
    r"\b(part\s+[a-z]|part\s+\d+|step\s+\d+|first[,\s]|second[,\s]|"
    r"third[,\s]|finally[,\s]|\([a-z]\)|\([ivx]+\)|\d+\.)\b",
    re.IGNORECASE,
)
_QUESTION_MARK = re.compile(r"\?")

# Imperative-like verb tags
_VERB_TAGS = frozenset({"VB", "VBP", "VBZ"})


def _count_parts(text: str) -> int:
    """Estimate number of distinct sub-tasks in the question stem."""
    count = 1  # baseline: at least one part

    # Heuristic 1: multiple question marks
    n_qmarks = len(_QUESTION_MARK.findall(text))
    if n_qmarks > 1:
        count = max(count, n_qmarks)

    # Heuristic 2: explicit part markers
    n_markers = len(_PART_MARKER.findall(text))
    if n_markers > 0:
        count = max(count, n_markers + 1)

    # Heuristic 3: coordinating conjunctions (CC) between verbs
    try:
        tokens = _word_tokenize(text)
        tagged = _pos_tag(tokens)
        cc_between_verbs = 0
        seen_verb = False
        for _, tag in tagged:
            if tag in _VERB_TAGS:
                seen_verb = True
            elif tag == "CC" and seen_verb:
                cc_between_verbs += 1
                seen_verb = False  # reset; wait for next verb
        count = max(count, 1 + cc_between_verbs)
    except Exception:
        pass

    return count


class MultiPartQuestionMetric(BaseReadabilityMetric):
    """
    Multi-part question detector: estimated number of distinct sub-tasks.

    Higher score → single-focus question → lower structural response burden.

    Args:
        max_extra_parts : Extra parts beyond 1 that map to score 0.0 (default 2).
        flag_below      : Score threshold below which flagged=True (default 0.5).
    """

    name = "multipart_question"
    description = (
        "Multi-part question indicator (estimated sub-task count) "
        "(higher score = single-focus item = lower structural response burden)."
    )

    def __init__(self, max_extra_parts: int = 2, flag_below: float = 0.5):
        self.max_extra_parts = max_extra_parts
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        return float(_count_parts(text))

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - (raw - 1.0) / self.max_extra_parts)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        parts = int(raw)
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Estimated {parts} sub-task(s) detected "
                f"(normalised {score:.4f}; max_extra_parts={self.max_extra_parts})."
            ),
            flagged=score < self.flag_below,
            metadata={"estimated_parts": parts, "max_extra_parts": self.max_extra_parts},
        )
