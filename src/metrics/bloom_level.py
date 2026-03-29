"""
Bloom's Taxonomy Level metric (Cognitive Demand).

Reference:
    Anderson, L. W., & Krathwohl, D. R. (Eds.). (2001). A Taxonomy for
    Learning, Teaching, and Assessing: A Revision of Bloom's Educational
    Objectives. Longman.

    Abdelghany, M., et al. (2024). Automated Analysis of Learning Outcomes
    and Exam Questions Based on Bloom's Taxonomy. arXiv:2511.10903.

What it measures:
    Cognitive level of the item on Bloom's revised taxonomy (6-level scale):

        Level 1 — Remember    (recall, retrieve, recognize, define, list)
        Level 2 — Understand  (explain, summarize, paraphrase, classify)
        Level 3 — Apply       (use, solve, implement, demonstrate)
        Level 4 — Analyze     (analyze, differentiate, compare, examine)
        Level 5 — Evaluate    (judge, assess, critique, justify, argue)
        Level 6 — Create      (design, construct, formulate, generate, invent)

    Detection is verb-based: the first action verb (or verb after the WH-word)
    is matched against per-level verb lists derived from Anderson & Krathwohl
    (2001). If multiple level-signal verbs appear, the highest level is used.

Score:
    score = level / 6.0 ∈ [0.17, 1.0]
    Higher score → higher cognitive level → greater cognitive demand.
    flag_below default 0.4 (Level 1–2; item only requires recall/comprehension).

Dependency:
    None — keyword mapping, no external packages required.
"""

from __future__ import annotations
import re
import logging
from src.models import MetricResult
from src.metrics.base_readability import BaseReadabilityMetric

logger = logging.getLogger(__name__)

# Verb lists per Bloom's revised taxonomy level (Anderson & Krathwohl, 2001)
_BLOOM_VERBS: dict[int, frozenset[str]] = {
    1: frozenset({  # Remember
        "define", "duplicate", "list", "memorize", "recall", "repeat",
        "reproduce", "state", "name", "identify", "recognize", "retrieve",
        "locate", "match", "label", "recite", "record", "relate", "underline",
    }),
    2: frozenset({  # Understand
        "classify", "describe", "discuss", "explain", "identify", "indicate",
        "locate", "recognize", "report", "select", "translate", "paraphrase",
        "summarize", "interpret", "compare", "infer", "illustrate", "outline",
        "restate", "review", "rewrite", "show", "give",
    }),
    3: frozenset({  # Apply
        "apply", "choose", "demonstrate", "dramatize", "employ", "illustrate",
        "operate", "schedule", "sketch", "solve", "use", "implement", "execute",
        "practice", "calculate", "compute", "construct", "manipulate", "modify",
        "produce", "relate", "show", "write",
    }),
    4: frozenset({  # Analyze
        "analyze", "appraise", "calculate", "categorize", "compare", "contrast",
        "criticize", "differentiate", "discriminate", "distinguish", "examine",
        "experiment", "question", "test", "break", "inspect", "investigate",
        "separate", "subdivide", "survey", "deduce", "deconstruct",
    }),
    5: frozenset({  # Evaluate
        "appraise", "argue", "assess", "attach", "choose", "compare", "defend",
        "estimate", "judge", "predict", "rate", "support", "value", "critique",
        "evaluate", "justify", "prioritize", "conclude", "recommend", "select",
        "rank", "weigh", "measure", "verify",
    }),
    6: frozenset({  # Create
        "arrange", "assemble", "collect", "compose", "construct", "create",
        "design", "develop", "formulate", "manage", "organize", "plan",
        "prepare", "propose", "generate", "produce", "hypothesize", "invent",
        "author", "combine", "compile", "devise", "integrate", "revise",
    }),
}

# Flat lookup: verb → highest level that verb signals
_VERB_LEVEL: dict[str, int] = {}
for _lvl, _verbs in _BLOOM_VERBS.items():
    for _v in _verbs:
        # Keep the highest level if a verb appears in multiple levels
        if _v not in _VERB_LEVEL or _VERB_LEVEL[_v] < _lvl:
            _VERB_LEVEL[_v] = _lvl

_LEVEL_LABELS = {
    1: "Remember",
    2: "Understand",
    3: "Apply",
    4: "Analyze",
    5: "Evaluate",
    6: "Create",
}


def _detect_level(text: str) -> tuple[int, str]:
    """Return (bloom_level, matched_verb). Level 0 if no verb matched."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    best_level = 0
    best_verb = ""
    for word in words:
        lvl = _VERB_LEVEL.get(word, 0)
        if lvl > best_level:
            best_level = lvl
            best_verb = word
    return best_level, best_verb


class BloomLevelMetric(BaseReadabilityMetric):
    """
    Bloom's revised taxonomy level detected from the question's action verbs.

    Higher score → higher cognitive level → greater cognitive demand.
    Returns score=0.17 (Level 1) when no level-signal verb is found.

    Args:
        flag_below : Score threshold below which flagged=True (default 0.4,
                     corresponding to Level 1–2 — recall/comprehension only).
    """

    name = "bloom_level"
    description = (
        "Bloom's revised taxonomy level (1=Remember → 6=Create) "
        "(higher score = higher cognitive demand)."
    )

    def __init__(self, flag_below: float = 0.4):
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        level, _ = _detect_level(text)
        return float(max(level, 1))  # default to Level 1 if nothing matched

    def _normalize(self, raw: float) -> float:
        return raw / 6.0

    def compute(self, question, context):
        if not question.text.strip():
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                rationale="Question text is empty.",
                flagged=True,
                metadata={"reason": "empty_text"},
            )
        level, verb = _detect_level(question.text)
        level = max(level, 1)
        score = level / 6.0
        label = _LEVEL_LABELS[level]
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Bloom's Level {level} — {label}"
                + (f" (verb: '{verb}')" if verb else " (no signal verb found; defaulting to Level 1)")
                + f" (score={score:.4f})."
            ),
            flagged=score < self.flag_below,
            metadata={"bloom_level": level, "bloom_label": label, "signal_verb": verb},
        )

    def _build_result(self, score: float, raw: float) -> MetricResult:  # pragma: no cover
        raise NotImplementedError
