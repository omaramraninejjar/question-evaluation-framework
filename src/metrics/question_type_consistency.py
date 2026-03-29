"""
Question Type Consistency metric (Well-Formedness).

Reference:
    Graesser, A. C., Cai, Z., Louwerse, M. M., & Daniel, F. (2006).
    Question Understanding Aid (QUAID): A web facility that tests problem
    comprehensibility of survey questions. Public Opinion Quarterly, 70(1),
    3–22. https://doi.org/10.1093/poq/nfj007

    Rus, V., Graesser, A. C., & Susarla, H. S. (2010). Question generation:
    Example of a multi-year evaluation campaign. WS on the People's Web
    Meets NLP (ACL 2010).

What it measures:
    Consistency between the WH-interrogative word and the expected answer
    type inferred from named entities present in the question.

    Mapping used:
        who   → expects a PERSON entity
        where → expects a GPE, LOC, or FAC entity
        when  → expects a DATE or TIME entity

    If the WH-word is "what", "which", "why", "how", or absent, the answer
    type is unconstrained and no consistency check is performed (score = 1.0).

    Scoring:
        WH-word with matching entity present → 1.0 (consistent)
        WH-word with no matching entity      → 0.5 (potentially inconsistent)
        WH-word unconstrained / unknown      → 1.0 (cannot evaluate)

    A score of 0.5 is flagged because it may indicate the question stem
    uses a specific WH-word without the corresponding conceptual anchor
    (e.g., "Who invented the transistor?" where "transistor" is PRODUCT,
    not a PERSON — suggesting the answer type might mismatch the phrasing).

    flag_below default 0.6 (flags uncertain consistency).

Dependency:
    spacy >= 3.0          (pip install spacy)
    en_core_web_sm model  (python -m spacy download en_core_web_sm)
"""

from __future__ import annotations
import re
import logging
from src.models import Question, EvaluationContext, MetricResult
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


_WH_PATTERN = re.compile(
    r"\b(what|who|where|when|which|why|how)\b", re.IGNORECASE
)

# WH-words with constrained answer types
_WH_ENTITY_MAP: dict[str, frozenset[str]] = {
    "who":   frozenset({"PERSON"}),
    "where": frozenset({"GPE", "LOC", "FAC"}),
    "when":  frozenset({"DATE", "TIME"}),
}


class QuestionTypeConsistencyMetric(BaseReadabilityMetric):
    """
    WH-word / answer-type consistency check using named entity recognition.

    Higher score → WH-word matches entity type present in question → more
    coherently framed item.

    Args:
        flag_below : Score threshold below which flagged=True (default 0.6).
    """

    name = "question_type_consistency"
    description = (
        "WH-word to answer-type consistency "
        "(higher score = WH-word matches entities in question = clearer item framing)."
    )

    def __init__(self, flag_below: float = 0.6):
        if not _SPACY_AVAILABLE:
            raise ImportError(
                "spacy is required for QuestionTypeConsistencyMetric.\n"
                "pip install spacy && python -m spacy download en_core_web_sm"
            )
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        return 0.0  # unused — logic is in compute()

    def _normalize(self, raw: float) -> float:
        return raw  # unused

    def compute(self, question: Question, context: EvaluationContext) -> MetricResult:
        if not question.text.strip():
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                rationale="Question text is empty.",
                flagged=True,
                metadata={"reason": "empty_text"},
            )
        text = question.text
        wh_matches = [m.lower() for m in _WH_PATTERN.findall(text)]
        wh_word = wh_matches[0] if wh_matches else "none"

        expected_types = _WH_ENTITY_MAP.get(wh_word)
        if expected_types is None:
            # Unconstrained WH-word — no check needed
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                rationale=(
                    f"WH-word '{wh_word}' has no constrained answer type "
                    f"— consistency check not applicable (score=1.0)."
                ),
                flagged=False,
                metadata={
                    "wh_word": wh_word,
                    "constrained": False,
                    "entities_found": [],
                },
            )

        nlp = _get_nlp()
        doc = nlp(text)
        entity_labels = [ent.label_ for ent in doc.ents]
        matching = [lbl for lbl in entity_labels if lbl in expected_types]

        if matching:
            score = 1.0
            rationale = (
                f"WH-word '{wh_word}' expects {sorted(expected_types)} entity — "
                f"found {matching} (consistent; score=1.0)."
            )
        else:
            score = 0.5
            rationale = (
                f"WH-word '{wh_word}' expects {sorted(expected_types)} entity — "
                f"none found in question (uncertain; score=0.5)."
            )

        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=rationale,
            flagged=score < self.flag_below,
            metadata={
                "wh_word": wh_word,
                "expected_entity_types": sorted(expected_types),
                "entities_found": entity_labels,
                "matching_entities": matching,
                "constrained": True,
            },
        )

    # Required by abstract base — not called when compute() is overridden above
    def _build_result(self, score: float, raw: float) -> MetricResult:  # pragma: no cover
        raise NotImplementedError
