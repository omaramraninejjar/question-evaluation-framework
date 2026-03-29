"""
Differential Privacy ε-Risk metric (Privacy Risk).

Reference:
    Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). Calibrating
    noise to sensitivity in private data analysis. Theory of Cryptography
    Conference (TCC), 265–284. https://doi.org/10.1007/11681878_14

    Dwork, C., & Roth, A. (2014). The Algorithmic Foundations of Differential
    Privacy. Foundations and Trends in Theoretical Computer Science, 9(3–4),
    211–407. https://doi.org/10.1561/0400000042

    Annamalai, M. S. M. S., Gadotti, A., & Rocher, L. (2024). A near-optimal
    mechanism for privacy measurement via differential privacy. arXiv:2405.04440.

What it measures:
    A text-level proxy for the privacy budget ε (epsilon) consumed by the
    identifying information content of a question.

    In formal differential privacy, ε bounds the log-likelihood ratio of any
    outcome for adjacent datasets (differing by one individual). Applied to
    text:

        ε_proxy = Σ_i  bits_i × weight_i

    where each detected identifying attribute type contributes an information
    estimate (in bits) reflecting how much it narrows the identification space:

        PERSON name      → 13 bits  (one in ~8,000 individuals)
        Specific age     →  7 bits  (one in ~100 year-group)
        Gender indicator →  1 bit   (binary attribute)
        Medical condition → 7 bits  (rare condition: ~1 in 100)
        Employer/Org     →  5 bits  (one in ~30 organisations in context)
        Geo-location     →  4 bits  (one in ~10 areas)
        Group membership →  3 bits  (one in ~8 demographic groups)

    These estimates follow from the population-frequency interpretation of
    self-information: bits = log2(1 / p_i).

    score = max(0.0, 1.0 − ε_proxy / max_epsilon)
    Default max_epsilon = 20 bits (≈ uniquely identifying one person in 1M).

    Higher ε_proxy → more information → smaller privacy budget remaining →
    lower score.

    INTERPRETATION NOTE:
    This is a *proxy* measure aligned with the spirit of DP, not a formal DP
    guarantee. It estimates the information content of the question's attributes
    as an upper bound on ε. A formal DP guarantee requires a randomisation
    mechanism applied to the data generating process, which is outside the
    scope of static item text analysis.

Score:
    score = max(0.0, 1.0 − ε_proxy / max_epsilon) ∈ [0.0, 1.0]
    Higher score → lower ε_proxy → lower estimated information leakage.
    flag_below default 0.5 (ε_proxy ≥ 10 bits → likely unique identification).

Dependency:
    spacy >= 3.0          (pip install spacy)
    en_core_web_sm model  (python -m spacy download en_core_web_sm)
"""

from __future__ import annotations
import re
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


# Bits of information per quasi-identifier type
# Derived from log2(1/p) where p = population frequency of the attribute
_BITS: dict[str, float] = {
    "person_name":        13.0,  # ~1/8000 — first + last name
    "age_specific":        7.0,  # ~1/100 — single year of age
    "gender":              1.0,  # ~1/2  — binary
    "medical_condition":   7.0,  # ~1/100 — specific diagnosis
    "employer":            5.0,  # ~1/32  — specific named organisation
    "location_specific":   4.0,  # ~1/16  — specific neighbourhood/facility
    "group_membership":    3.0,  # ~1/8   — demographic group
}

_GENDER_WORDS = frozenset({
    "he", "she", "his", "her", "him", "hers", "male", "female",
    "man", "woman", "boy", "girl",
})
_MEDICAL_WORDS = frozenset({
    "diagnosis", "diagnosed", "condition", "disease", "disorder", "syndrome",
    "patient", "symptom", "treatment", "medication", "therapy", "chronic",
    "cancer", "diabetes", "hypertension", "infection", "allergy", "allergic",
})
_AGE_RE = re.compile(
    r"\b(\d{1,3}[-–]\s*year[-–]\s*old|\d{1,3}\s*years?\s*old|age\s+\d{1,3}|"
    r"aged\s+\d{1,3})\b",
    re.IGNORECASE,
)


def _epsilon_components(text: str) -> dict[str, float]:
    """Return dict of attribute_type → bits contributed."""
    components: dict[str, float] = {}
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))

    if words & _GENDER_WORDS:
        components["gender"] = _BITS["gender"]
    if _AGE_RE.search(text):
        components["age_specific"] = _BITS["age_specific"]
    if words & _MEDICAL_WORDS:
        components["medical_condition"] = _BITS["medical_condition"]

    # NER-based
    if _SPACY_AVAILABLE:
        try:
            nlp = _get_nlp()
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON" and "person_name" not in components:
                    components["person_name"] = _BITS["person_name"]
                elif ent.label_ == "ORG" and "employer" not in components:
                    components["employer"] = _BITS["employer"]
                elif ent.label_ in {"GPE", "LOC", "FAC"} and "location_specific" not in components:
                    components["location_specific"] = _BITS["location_specific"]
                elif ent.label_ == "NORP" and "group_membership" not in components:
                    components["group_membership"] = _BITS["group_membership"]
        except Exception as exc:
            logger.warning("spaCy NER failed in DP epsilon metric: %s", exc)

    return components


class DPEpsilonRiskMetric(BaseReadabilityMetric):
    """
    Differential Privacy ε-proxy: estimates information content of identifying
    attributes in the question text as an upper-bound on ε.

    Higher score → lower ε_proxy → lower estimated information leakage.

    Args:
        max_epsilon : ε_proxy that maps to score 0.0 (default 20 bits).
        flag_below  : Score threshold below which flagged=True (default 0.5).
    """

    name = "dp_epsilon_risk"
    description = (
        "DP ε-proxy: information content of identifying attributes (bits) "
        "(higher score = lower ε_proxy = lower information leakage risk)."
    )

    def __init__(self, max_epsilon: float = 20.0, flag_below: float = 0.5):
        if not _SPACY_AVAILABLE:
            raise ImportError(
                "spacy is required for DPEpsilonRiskMetric.\n"
                "pip install spacy && python -m spacy download en_core_web_sm"
            )
        self.max_epsilon = max_epsilon
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        return sum(_epsilon_components(text).values())

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - raw / self.max_epsilon)

    def compute(self, question, context):
        if not question.text.strip():
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                rationale="Question text is empty.",
                flagged=True,
                metadata={"reason": "empty_text"},
            )
        components = _epsilon_components(question.text)
        epsilon = sum(components.values())
        score = max(0.0, 1.0 - epsilon / self.max_epsilon)

        if not components:
            rationale = f"No identifying attributes detected; ε_proxy=0 (score={score:.4f})."
        else:
            breakdown = ", ".join(f"{k}={v:.0f}b" for k, v in components.items())
            rationale = (
                f"ε_proxy={epsilon:.1f} bits [{breakdown}] "
                f"(max_epsilon={self.max_epsilon}; score={score:.4f})."
            )
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=rationale,
            flagged=score < self.flag_below,
            metadata={
                "epsilon_proxy": epsilon,
                "epsilon_components": components,
                "max_epsilon": self.max_epsilon,
            },
        )

    def _build_result(self, score: float, raw: float) -> MetricResult:  # pragma: no cover
        raise NotImplementedError
