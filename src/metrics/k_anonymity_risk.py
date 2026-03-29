"""
k-Anonymity Risk metric (Privacy Risk).

Reference:
    Sweeney, L. (2002). k-anonymity: A model for protecting privacy.
    International Journal of Uncertainty, Fuzziness and Knowledge-Based
    Systems, 10(5), 557–570. https://doi.org/10.1142/S0218488502001648

    Samarati, P., & Sweeney, L. (1998). Protecting privacy when disclosing
    information: k-anonymity and its enforcement through generalization and
    suppression. Technical Report SRI-CSL-98-04.

What it measures:
    Risk that the combination of quasi-identifier attributes present in the
    question text could uniquely identify an individual (k < threshold).

    In classical k-anonymity, a record is safe if at least k−1 other records
    share its exact combination of quasi-identifiers. Applied to question text,
    we estimate k from the number of distinct quasi-identifier attribute TYPES
    present:

        Quasi-identifier types detected (via spaCy NER + heuristics):
            PERSON   — named individual (k = 1; maximally identifying)
            AGE      — specific age or age range (CARDINAL + age context)
            GENDER   — gender-coded pronoun or explicit gender mention
            GPE/LOC  — specific geographic location
            NORP     — nationality, religious, or political group
            ORG      — employer or organisational membership
            MEDICAL  — medical condition keywords (symptom, diagnosis)

        k_estimate = N / 2^n_attributes
        where N = 10,000 (assumed population base, configurable) and
        n_attributes = count of distinct quasi-identifier types found.

        score = min(1.0, k_estimate / k_threshold)
        Default k_threshold = 5 (GDPR-inspired anonymisation guideline).

    A question containing only "a patient" (no name, age, location) has
    n_attributes ≈ 0 → k → ∞ → score = 1.0.
    A question describing a 42-year-old female nurse at a named hospital
    has n_attributes = 4–5 → k_estimate ≈ 625 or less → low score.

Score:
    score = min(1.0, k_estimate / k_threshold) ∈ [0.0, 1.0]
    Higher score → higher estimated k → lower re-identification risk.
    flag_below default 0.5 (k_estimate < k_threshold / 2).

Dependency:
    spacy >= 3.0          (pip install spacy)
    en_core_web_sm model  (python -m spacy download en_core_web_sm)
"""

from __future__ import annotations
import re
import math
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


_GENDER_WORDS = frozenset({
    "he", "she", "his", "her", "him", "hers", "male", "female",
    "man", "woman", "boy", "girl", "gentleman", "lady",
})

_MEDICAL_WORDS = frozenset({
    "diagnosis", "diagnosed", "condition", "disease", "disorder", "syndrome",
    "patient", "symptom", "treatment", "medication", "therapy", "chronic",
    "acute", "surgery", "procedure", "cancer", "diabetes", "hypertension",
    "infection", "allergy", "allergic", "prescription",
})

_AGE_RE = re.compile(
    r"\b(\d{1,3}[-–]\s*year[-–]\s*old|\d{1,3}\s*years?\s*old|age\s+\d{1,3}|"
    r"aged\s+\d{1,3})\b",
    re.IGNORECASE,
)


def _detect_quasi_identifiers(text: str) -> dict[str, bool]:
    """Return dict of quasi-identifier type → present."""
    qi: dict[str, bool] = {
        "person_name": False,
        "age": False,
        "gender": False,
        "location": False,
        "group_membership": False,
        "employer": False,
        "medical_condition": False,
    }

    # NER-based
    try:
        nlp = _get_nlp()
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                qi["person_name"] = True
            elif ent.label_ in {"GPE", "LOC", "FAC"}:
                qi["location"] = True
            elif ent.label_ == "NORP":
                qi["group_membership"] = True
            elif ent.label_ == "ORG":
                qi["employer"] = True
    except Exception as exc:
        logger.warning("spaCy NER failed in k-anonymity: %s", exc)

    # Age pattern
    if _AGE_RE.search(text):
        qi["age"] = True

    # Gender detection
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    if words & _GENDER_WORDS:
        qi["gender"] = True

    # Medical condition
    if words & _MEDICAL_WORDS:
        qi["medical_condition"] = True

    return qi


class KAnonymityRiskMetric(BaseReadabilityMetric):
    """
    k-Anonymity risk estimator: infers re-identification risk from the
    combination of quasi-identifier attribute types in the question text.

    Higher score → higher estimated k → lower re-identification risk.

    Args:
        k_threshold    : Minimum k considered safe (default 5, GDPR-inspired).
        population_n   : Assumed population base for k estimation (default 10000).
        flag_below     : Score threshold below which flagged=True (default 0.5).
    """

    name = "k_anonymity_risk"
    description = (
        "k-Anonymity risk from quasi-identifier combination in question text "
        "(higher score = higher estimated k = lower re-identification risk)."
    )

    def __init__(
        self,
        k_threshold: int = 5,
        population_n: int = 10_000,
        flag_below: float = 0.5,
    ):
        if not _SPACY_AVAILABLE:
            raise ImportError(
                "spacy is required for KAnonymityRiskMetric.\n"
                "pip install spacy && python -m spacy download en_core_web_sm"
            )
        self.k_threshold = k_threshold
        self.population_n = population_n
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        """Return estimated k value."""
        qi = _detect_quasi_identifiers(text)
        n_attrs = sum(qi.values())
        # k_estimate = N / 2^n_attrs; direct name = k=1 override
        if qi["person_name"]:
            return 1.0
        if n_attrs == 0:
            return float(self.population_n)
        return max(1.0, self.population_n / (2 ** n_attrs))

    def _normalize(self, raw: float) -> float:
        return min(1.0, raw / self.k_threshold)

    def compute(self, question, context):
        if not question.text.strip():
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                rationale="Question text is empty.",
                flagged=True,
                metadata={"reason": "empty_text"},
            )
        qi = _detect_quasi_identifiers(question.text)
        n_attrs = sum(qi.values())

        if qi["person_name"]:
            k_estimate = 1.0
        elif n_attrs == 0:
            k_estimate = float(self.population_n)
        else:
            k_estimate = max(1.0, self.population_n / (2 ** n_attrs))

        score = min(1.0, k_estimate / self.k_threshold)
        active = [attr for attr, present in qi.items() if present]

        rationale = (
            f"Quasi-identifiers detected: {active} (n={n_attrs}); "
            f"estimated k≈{k_estimate:.0f} (threshold={self.k_threshold}; score={score:.4f})."
            if active
            else f"No quasi-identifiers detected; k→∞ (score={score:.4f})."
        )
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=rationale,
            flagged=score < self.flag_below,
            metadata={
                "quasi_identifiers": qi,
                "n_attributes": n_attrs,
                "k_estimate": k_estimate,
                "k_threshold": self.k_threshold,
            },
        )

    def _build_result(self, score: float, raw: float) -> MetricResult:  # pragma: no cover
        raise NotImplementedError
