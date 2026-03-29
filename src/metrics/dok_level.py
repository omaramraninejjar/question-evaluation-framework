"""
Webb's Depth of Knowledge (DOK) Level metric (Cognitive Demand).

Reference:
    Webb, N. L. (1997). Research Monograph Number 6: Criteria for Alignment
    of Expectations and Assessments in Mathematics and Science Education.
    CCSSO and NISE.

    Webb, N. L. (2002). Depth-of-Knowledge Levels for Four Content Areas.
    Language Arts, Mathematics, Science, and Social Studies. CCSSO.

    Hess, K. K. (2009). Cognitive Rigor: Blending the Strengths of Bloom's
    Taxonomy and Webb's Depth of Knowledge to Enhance Classroom-Level
    Processes. ERIC ED517804.

What it measures:
    Complexity level on Webb's four-level Depth of Knowledge (DOK) scale:

        DOK 1 — Recall & Reproduction
                 (recall facts, apply simple procedures, define, identify)
        DOK 2 — Skills & Concepts
                 (use information, explain, classify, summarize, interpret)
        DOK 3 — Strategic Thinking
                 (analyze, develop a plan, justify, support, investigate)
        DOK 4 — Extended Thinking
                 (synthesize information across sources, design, produce
                  complex arguments, conduct extended research)

    DOK differs from Bloom's by focusing on the DEPTH of thinking required
    rather than the type of cognitive process. Detection is keyword-based,
    targeting verbs and phrases that anchor each DOK level.

Score:
    score = dok_level / 4.0 ∈ [0.25, 1.0]
    Higher score → higher DOK level → deeper cognitive engagement.
    flag_below default 0.35 (DOK 1 only; purely recall item).

Dependency:
    None — keyword mapping, no external packages required.
"""

from __future__ import annotations
import re
import logging
from src.models import MetricResult
from src.metrics.base_readability import BaseReadabilityMetric

logger = logging.getLogger(__name__)

# Keyword lists per Webb's DOK level
_DOK_KEYWORDS: dict[int, frozenset[str]] = {
    1: frozenset({  # Recall & Reproduction
        "define", "list", "name", "identify", "recall", "recognize", "state",
        "repeat", "match", "label", "memorize", "locate", "find", "count",
        "compute", "recite", "tell", "who", "what", "when", "where",
    }),
    2: frozenset({  # Skills & Concepts
        "describe", "explain", "summarize", "classify", "compare", "interpret",
        "use", "organize", "estimate", "infer", "predict", "show", "apply",
        "solve", "calculate", "illustrate", "sort", "distinguish", "categorize",
        "relate", "extend", "make",
    }),
    3: frozenset({  # Strategic Thinking
        "analyze", "evaluate", "justify", "critique", "formulate", "investigate",
        "assess", "develop", "revise", "support", "argue", "differentiate",
        "construct", "examine", "hypothesize", "determine", "cite", "explain",
        "phenomena", "draw", "conclusions", "synthesize",
    }),
    4: frozenset({  # Extended Thinking
        "design", "create", "produce", "compose", "generate", "propose",
        "conduct", "research", "synthesize", "integrate", "prove", "invent",
        "document", "gather", "plan", "debate", "collaborate", "critique",
        "across", "multiple",
    }),
}

_VERB_DOK: dict[str, int] = {}
for _dok, _kws in _DOK_KEYWORDS.items():
    for _kw in _kws:
        if _kw not in _VERB_DOK or _VERB_DOK[_kw] < _dok:
            _VERB_DOK[_kw] = _dok

_DOK_LABELS = {
    1: "Recall & Reproduction",
    2: "Skills & Concepts",
    3: "Strategic Thinking",
    4: "Extended Thinking",
}


def _detect_dok(text: str) -> tuple[int, str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    best = 0
    best_kw = ""
    for w in words:
        lvl = _VERB_DOK.get(w, 0)
        if lvl > best:
            best = lvl
            best_kw = w
    return best, best_kw


class DOKLevelMetric(BaseReadabilityMetric):
    """
    Webb's Depth of Knowledge level detected from question keywords.

    Higher score → higher DOK level → deeper cognitive engagement.
    Defaults to DOK 1 when no keyword is found.

    Args:
        flag_below : Score threshold below which flagged=True (default 0.35,
                     corresponding to DOK 1 — recall only).
    """

    name = "dok_level"
    description = (
        "Webb's Depth of Knowledge level (1=Recall → 4=Extended Thinking) "
        "(higher score = deeper cognitive engagement)."
    )

    def __init__(self, flag_below: float = 0.35):
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        level, _ = _detect_dok(text)
        return float(max(level, 1))

    def _normalize(self, raw: float) -> float:
        return raw / 4.0

    def compute(self, question, context):
        if not question.text.strip():
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                rationale="Question text is empty.",
                flagged=True,
                metadata={"reason": "empty_text"},
            )
        level, keyword = _detect_dok(question.text)
        level = max(level, 1)
        score = level / 4.0
        label = _DOK_LABELS[level]
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"DOK Level {level} — {label}"
                + (f" (keyword: '{keyword}')" if keyword else " (no signal keyword; defaulting to DOK 1)")
                + f" (score={score:.4f})."
            ),
            flagged=score < self.flag_below,
            metadata={"dok_level": level, "dok_label": label, "signal_keyword": keyword},
        )

    def _build_result(self, score: float, raw: float) -> MetricResult:  # pragma: no cover
        raise NotImplementedError
