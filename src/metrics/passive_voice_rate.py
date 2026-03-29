"""
Passive Voice Rate metric (Linguistic Complexity).

Reference:
    Davison, A., & Lutz, R. (1985). Measuring syntactic complexity relative
    to discourse context. In D. R. Beyerstein (Ed.), Analyzing Written
    Discourse (pp. 177–210). Ablex.

    Rayner, K., Carlson, M., & Frazier, L. (1983). The interaction of syntax
    and semantics during sentence processing: Eye movements in the analysis
    of semantically biased sentences. Journal of Verbal Learning and Verbal
    Behavior, 22(3), 358–374. https://doi.org/10.1016/S0022-5371(83)90236-0

What it measures:
    Proportion of clauses that use passive voice constructions.
    Passive constructions are detected via spaCy dependency labels:
        "nsubjpass" — nominal subject in a passive clause
        "auxpass"   — passive auxiliary (be/get + past participle)

    Passive voice in test items is associated with higher processing load
    and construct-irrelevant difficulty (AERA/APA/NCME, 2014, Standards).
    Educational measurement guidelines recommend active-voice phrasing.

Score:
    score = max(0.0, 1.0 − passive_count / max_passive)
    Default max_passive = 1 (any passive construction = score 0.0).
    Higher score → no passive voice → more direct, active-voice phrasing.
    flag_below default 0.5 (flags any question with passive voice).

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


_PASSIVE_DEPS = frozenset({"nsubjpass", "auxpass"})


class PassiveVoiceRateMetric(BaseReadabilityMetric):
    """
    Passive voice indicator: count of passive dependency tokens.

    Higher score → no passive constructions → active voice → lower complexity.

    Args:
        max_passive : Passive token count that maps to score 0.0 (default 1).
        flag_below  : Score threshold below which flagged=True (default 0.5).
    """

    name = "passive_voice_rate"
    description = (
        "Passive voice indicator (passive dependency tokens) "
        "(higher score = active voice = lower processing complexity)."
    )

    def __init__(self, max_passive: int = 1, flag_below: float = 0.5):
        if not _SPACY_AVAILABLE:
            raise ImportError(
                "spacy is required for PassiveVoiceRateMetric.\n"
                "pip install spacy && python -m spacy download en_core_web_sm"
            )
        self.max_passive = max_passive
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        nlp = _get_nlp()
        doc = nlp(text)
        return float(sum(1 for tok in doc if tok.dep_ in _PASSIVE_DEPS))

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - raw / self.max_passive)

    def _build_result(self, score: float, raw: float) -> MetricResult:
        is_passive = int(raw) > 0
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"{'Passive' if is_passive else 'Active'} voice detected "
                f"({int(raw)} passive token(s); normalised {score:.4f})."
            ),
            flagged=score < self.flag_below,
            metadata={
                "passive_token_count": int(raw),
                "is_passive": is_passive,
                "max_passive": self.max_passive,
            },
        )
