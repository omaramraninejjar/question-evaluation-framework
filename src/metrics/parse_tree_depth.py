"""
Parse Tree Depth metric (Linguistic Complexity).

Reference:
    Gibson, E. (1998). Linguistic complexity: Locality of syntactic dependencies.
    Cognition, 68(1), 1–76. https://doi.org/10.1016/S0010-0277(98)00034-1

    Liu, H. (2008). Dependency distance as a metric of language comprehension difficulty.
    Journal of Cognitive Science, 9(2), 159–191.

What it measures:
    Maximum depth of the dependency parse tree. A deeper tree indicates more
    centre-embedded or subordinate clause structures, which impose greater
    working-memory load on the reader.

        depth(token) = number of edges from token to root
        tree_depth   = max(depth(token) for all tokens)

    Shallow trees (depth ≤ 2) = simple, direct questions.
    Deep trees (depth ≥ 6)    = complex, embedded constructions.

Score normalisation:
    score = max(0.0, 1.0 − tree_depth / max_depth)
    Default max_depth = 10.
    Higher score → shallower tree → lower syntactic complexity.

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

_nlp = None  # lazy-loaded spaCy pipeline


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


def _token_depth(token) -> int:
    depth = 0
    while token.head != token:
        depth += 1
        token = token.head
    return depth


class ParseTreeDepthMetric(BaseReadabilityMetric):
    """
    Maximum dependency parse tree depth for question.text.

    Higher score → shallower tree → simpler syntactic structure.

    Args:
        max_depth  : Tree depth that maps to score 0.0 (default 10).
        flag_below : Score threshold below which flagged=True (default 0.3).
    """

    name = "parse_tree_depth"
    description = (
        "Maximum dependency parse tree depth "
        "(higher score = shallower tree = lower syntactic complexity)."
    )

    def __init__(self, max_depth: int = 10, flag_below: float = 0.3):
        if not _SPACY_AVAILABLE:
            raise ImportError(
                "spacy is required for ParseTreeDepthMetric.\n"
                "pip install spacy && python -m spacy download en_core_web_sm"
            )
        self.max_depth = max_depth
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        nlp = _get_nlp()
        doc = nlp(text)
        depths = [_token_depth(tok) for tok in doc]
        return float(max(depths)) if depths else 0.0

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - raw / self.max_depth)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Parse tree depth {int(raw)} "
                f"(normalised {score:.4f}; max_depth={self.max_depth})."
            ),
            flagged=score < self.flag_below,
            metadata={"tree_depth": int(raw), "max_depth": self.max_depth},
        )
