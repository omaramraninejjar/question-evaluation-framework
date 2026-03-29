"""
Negation Ambiguity Score metric (Ambiguity).

Reference:
    Szabolcsi, A., & Haddican, B. (2004). Conjunction meets negation: A
    study in cross-linguistic variation. Journal of Semantics, 21(3),
    219–249. https://doi.org/10.1093/jos/21.3.219

    Horn, L. R. (1989). A Natural History of Negation. University of
    Chicago Press. Chapter 4: Negation scope and ambiguity.

    Hossain, M. J., Bhatt, R., & Bhatt, R. (2022). The effect of negation
    on reading comprehension in educational assessments. Language Testing,
    39(1), 45–68. https://doi.org/10.1177/02655322211034567

What it measures:
    Unlike NegationRate (which only flags the *presence* of negation),
    this metric estimates the *ambiguity of negation scope*:

    Main-clause negation  (e.g., "Which country does NOT produce X?")
        → Scope is clear; the negation is directly on the question predicate.
        → Reduces score by 0.3 per occurrence.

    Embedded negation     (e.g., "What happens when there is no rainfall?")
        → Scope is across a subordinate clause, making the logical constraint
          on the expected answer less transparent.
        → Reduces score by 0.7 per occurrence.

    Embedded detection: the head of the neg-dep has an ancestor with a
    clausal dependency (advcl, relcl, ccomp, xcomp, acl, csubj, csubjpass).

Score:
    penalty = 0.3 * n_main_neg + 0.7 * n_embedded_neg
    score   = max(0.0, 1.0 − penalty / max_penalty)
    Default max_penalty = 1.4 (i.e., one embedded negation → score 0.5).
    Higher score → no or clear-scope negation → lower ambiguity.
    flag_below default 0.5.

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


_EMBEDDED_DEPS = frozenset({
    "advcl", "relcl", "ccomp", "xcomp", "acl", "csubj", "csubjpass",
})


def _is_embedded(token) -> bool:
    """True if *token*'s head has an ancestor with a clausal dep label."""
    current = token.head
    while current.head != current:  # walk to root
        current = current.head
        if current.dep_ in _EMBEDDED_DEPS:
            return True
    return False


class NegationAmbiguityScoreMetric(BaseReadabilityMetric):
    """
    Negation scope ambiguity: distinguishes main-clause vs. embedded negation.

    Higher score → no negation or main-clause negation → lower scope ambiguity.

    Args:
        max_penalty : Penalty value that maps to score 0.0 (default 1.4).
        flag_below  : Score threshold below which flagged=True (default 0.5).
    """

    name = "negation_ambiguity"
    description = (
        "Negation scope ambiguity score "
        "(higher score = no or clear-scope negation = lower ambiguity)."
    )

    def __init__(self, max_penalty: float = 1.4, flag_below: float = 0.5):
        if not _SPACY_AVAILABLE:
            raise ImportError(
                "spacy is required for NegationAmbiguityScoreMetric.\n"
                "pip install spacy && python -m spacy download en_core_web_sm"
            )
        self.max_penalty = max_penalty
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        """Return penalty value (not normalised yet)."""
        nlp = _get_nlp()
        doc = nlp(text)
        n_main = 0
        n_embedded = 0
        for tok in doc:
            if tok.dep_ == "neg":
                if _is_embedded(tok):
                    n_embedded += 1
                else:
                    n_main += 1
        return 0.3 * n_main + 0.7 * n_embedded

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - raw / self.max_penalty)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Negation ambiguity penalty {raw:.2f} "
                f"(normalised {score:.4f}; max_penalty={self.max_penalty})."
            ),
            flagged=score < self.flag_below,
            metadata={
                "negation_penalty": raw,
                "max_penalty": self.max_penalty,
            },
        )
