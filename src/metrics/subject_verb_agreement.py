"""
Subject-Verb Agreement Score metric (Well-Formedness).

Reference:
    Chomsky, N. (1965). Aspects of the Theory of Syntax. MIT Press.
    Chapter 2: Deep structures and grammatical relations.

    Bender, E. M. (2013). Linguistic Fundamentals for Natural Language
    Processing. Morgan & Claypool. Chapter 4: Morphosyntax.

What it measures:
    Proportion of subject–verb pairs where the grammatical number
    (singular / plural) of the subject matches that of the main verb.

    For each token with dependency "nsubj" or "nsubjpass" whose head is a
    VERB or AUX:
        agree = subject.morph["Number"] == head.morph["Number"]

    A mismatch (e.g., "The results *was* unclear?") is a clear grammatical
    error that lowers item quality and distracts respondents.

    agreement_rate = matching_pairs / total_pairs

    When no subject–verb pairs are found (e.g., imperative, fragment), a
    neutral score of 0.5 is returned (cannot evaluate).

Score:
    score = agreement_rate ∈ [0.0, 1.0]
    Higher score → better subject–verb agreement → more grammatically well-formed.
    flag_below default 0.7 (any single disagreement in a short question → flag).

Dependency:
    spacy >= 3.0          (pip install spacy)
    en_core_web_sm model  (python -m spacy download en_core_web_sm)
"""

from __future__ import annotations
import logging
from src.models import MetricResult
from src.metrics.base_readability import BaseReadabilityMetric

logger = logging.getLogger(__name__)

try:
    import spacy as _spacy
    _SPACY_AVAILABLE = True
except Exception:
    _spacy = None  # type: ignore[assignment]
    _SPACY_AVAILABLE = False

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        for model_name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
            try:
                _nlp = _spacy.load(model_name)
                return _nlp
            except OSError:
                continue
        raise RuntimeError(
            "No spaCy English model found.\n"
            "Run: python -m spacy download en_core_web_sm"
        )
    return _nlp


_SUBJ_DEPS = frozenset({"nsubj", "nsubjpass"})


def _sv_agreement_rate(doc) -> tuple[float, int, int]:
    """
    Return (agreement_rate, n_agree, n_total).
    agreement_rate is 0.5 when n_total == 0 (undetermined).
    """
    n_agree = 0
    n_total = 0
    for tok in doc:
        if tok.dep_ not in _SUBJ_DEPS:
            continue
        head = tok.head
        if head.pos_ not in {"VERB", "AUX"}:
            continue
        subj_num = tok.morph.get("Number")
        verb_num = head.morph.get("Number")
        if subj_num and verb_num:
            n_total += 1
            if subj_num == verb_num:
                n_agree += 1
    if n_total == 0:
        return 0.5, 0, 0
    return n_agree / n_total, n_agree, n_total


class SubjectVerbAgreementMetric(BaseReadabilityMetric):
    """
    Subject–verb agreement rate: fraction of SV pairs with matching number.

    Higher score → better agreement → more grammatically well-formed question.
    Returns 0.5 when no SV pairs are detected (cannot evaluate).

    Args:
        flag_below : Score threshold below which flagged=True (default 0.7).
    """

    name = "subject_verb_agreement"
    description = (
        "Subject–verb number agreement rate "
        "(higher score = better grammatical agreement = more well-formed)."
    )

    def __init__(self, flag_below: float = 0.7):
        if not _SPACY_AVAILABLE:
            raise ImportError(
                "spacy is required for SubjectVerbAgreementMetric.\n"
                "pip install spacy && python -m spacy download en_core_web_sm"
            )
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        nlp = _get_nlp()
        doc = nlp(text)
        rate, _, _ = _sv_agreement_rate(doc)
        return rate

    def _normalize(self, raw: float) -> float:
        return raw  # already in [0, 1]

    def compute(self, question, context):
        if not question.text.strip():
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                rationale="Question text is empty.",
                flagged=True,
                metadata={"reason": "empty_text"},
            )
        nlp = _get_nlp()
        doc = nlp(question.text)
        rate, n_agree, n_total = _sv_agreement_rate(doc)
        score = rate
        if n_total == 0:
            rationale = "No subject–verb pairs with number morphology found (score=0.5, undetermined)."
        else:
            rationale = (
                f"{n_agree}/{n_total} subject–verb pair(s) agree in number "
                f"(agreement_rate={score:.4f})."
            )
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=rationale,
            flagged=score < self.flag_below,
            metadata={
                "agreement_rate": score,
                "pairs_agree": n_agree,
                "pairs_total": n_total,
            },
        )

    # Required by abstract base — not called when compute() is overridden above
    def _build_result(self, score: float, raw: float) -> MetricResult:  # pragma: no cover
        raise NotImplementedError
