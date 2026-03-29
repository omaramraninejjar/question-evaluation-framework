"""
Constituent Count metric (Linguistic Complexity).

Reference:
    Abney, S. (1991). Parsing by chunks. In R. Berwick, S. Abney, &
    C. Tenny (Eds.), Principle-Based Parsing (pp. 257–278). Kluwer.

    Yngve, V. H. (1960). A model and an hypothesis for language structure.
    Proceedings of the American Philosophical Society, 104(5), 444–466.

What it measures:
    Number of syntactic phrase constituents in the question: noun phrases
    (from spaCy noun_chunks) plus finite verb phrases (identified by verb
    tokens that head a clause).

        constituent_count = |noun_chunks| + |clause_root_verbs|

    More constituents → more elaborated phrase structure → higher processing
    demand. Short, direct questions typically have 2–3 constituents; complex
    embedded questions may have 6 or more.

Score normalisation:
    score = max(0.0, 1.0 − count / max_constituents)
    Default max_constituents = 6.
    Higher score → fewer constituents → simpler phrase structure.
    flag_below default 0.3 (count ≥ 4.2 for max=6).

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


# Dependency labels that mark the root of a clause (finite or non-finite)
_CLAUSE_ROOTS = frozenset({
    "ROOT", "ccomp", "xcomp", "advcl", "relcl", "acl", "csubj", "csubjpass",
})


def _clause_root_verbs(doc) -> int:
    """Count verb tokens that are the root of a clause."""
    return sum(
        1 for tok in doc
        if tok.dep_ in _CLAUSE_ROOTS and tok.pos_ in {"VERB", "AUX"}
    )


class ConstituentCountMetric(BaseReadabilityMetric):
    """
    Number of syntactic constituents (noun phrases + clause-root verbs).

    Higher score → fewer constituents → simpler phrase structure.

    Args:
        max_constituents : Constituent count that maps to score 0.0 (default 6).
        flag_below       : Score threshold below which flagged=True (default 0.3).
    """

    name = "constituent_count"
    description = (
        "Number of syntactic phrase constituents "
        "(higher score = fewer constituents = simpler phrase structure)."
    )

    def __init__(self, max_constituents: int = 6, flag_below: float = 0.3):
        if not _SPACY_AVAILABLE:
            raise ImportError(
                "spacy is required for ConstituentCountMetric.\n"
                "pip install spacy && python -m spacy download en_core_web_sm"
            )
        self.max_constituents = max_constituents
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        nlp = _get_nlp()
        doc = nlp(text)
        n_np = len(list(doc.noun_chunks))
        n_vp = _clause_root_verbs(doc)
        return float(n_np + n_vp)

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - raw / self.max_constituents)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"{int(raw)} constituent(s) detected "
                f"(normalised {score:.4f}; max_constituents={self.max_constituents})."
            ),
            flagged=score < self.flag_below,
            metadata={
                "constituent_count": int(raw),
                "max_constituents": self.max_constituents,
            },
        )
