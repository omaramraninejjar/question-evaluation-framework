"""
PII Risk metric (Privacy Risk).

Reference:
    GDPR Article 4(1): "personal data means any information relating to an
    identified or identifiable natural person."

    Lison, P., Pilán, I., Sánchez, D., Romani, D., & Batet, M. (2021).
    Anonymisation Models for Clinical Notes: Evaluation of a Baseline System.
    Proceedings of ACL 2021.

    Presidio framework (Microsoft). https://microsoft.github.io/presidio/

What it measures:
    Presence and density of Personally Identifiable Information (PII) in the
    question text. Two detection layers are used:

    Layer 1 — Named Entity Recognition (spaCy):
        PERSON  — individual names
        GPE     — geopolitical entities (city/country level when combined)
        ORG     — organisational names that may identify context
        DATE    — specific dates (when combined with other PII, identifying)

    Layer 2 — Regex pattern matching:
        Email addresses    — \\S+@\\S+\\.\\S+
        Phone numbers      — common US/international formats
        Social Security Numbers — \\d{3}-\\d{2}-\\d{4}
        Student/employee IDs — digit sequences with context keywords

    pii_density = (n_ner_entities + n_regex_matches) / word_count

Score:
    score = max(0.0, 1.0 − pii_density / max_density)
    Default max_density = 0.3 (30 % PII tokens → score 0.0).
    Higher score → less PII content → lower privacy risk.
    flag_below default 0.7 (any PII detection above very low level → flag).

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


_PII_NER_LABELS = frozenset({"PERSON", "ORG"})  # GPE/DATE only PII in combination

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b")
_PHONE_RE = re.compile(
    r"\b(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_ID_RE = re.compile(
    r"\b(student|employee|id|ID|S/N|SID|EID|uid|user)\s*#?\s*\d{4,}\b",
    re.IGNORECASE,
)


def _detect_pii(text: str, use_spacy: bool = True) -> dict:
    """Return dict of detected PII counts and types."""
    results: dict = {
        "person_names": 0,
        "org_names": 0,
        "emails": 0,
        "phones": 0,
        "ssn": 0,
        "ids": 0,
        "ner_entities": [],
    }

    # Regex layer (always runs)
    results["emails"] = len(_EMAIL_RE.findall(text))
    results["phones"] = len(_PHONE_RE.findall(text))
    results["ssn"] = len(_SSN_RE.findall(text))
    results["ids"] = len(_ID_RE.findall(text))

    # NER layer
    if use_spacy and _SPACY_AVAILABLE:
        try:
            nlp = _get_nlp()
            doc = nlp(text)
            for ent in doc.ents:
                results["ner_entities"].append(ent.label_)
                if ent.label_ == "PERSON":
                    results["person_names"] += 1
                elif ent.label_ == "ORG":
                    results["org_names"] += 1
        except Exception as exc:
            logger.warning("spaCy NER failed: %s", exc)

    return results


class PIIRiskMetric(BaseReadabilityMetric):
    """
    PII density score: combined NER + regex detection of personal information.

    Higher score → less PII → lower privacy risk.
    Flag is raised on any meaningful PII presence.

    Args:
        max_density : PII token density mapped to score 0.0 (default 0.3).
        flag_below  : Score threshold below which flagged=True (default 0.7).
    """

    name = "pii_risk"
    description = (
        "PII presence score (named entities + regex patterns) "
        "(higher score = less PII detected = lower privacy risk)."
    )

    def __init__(self, max_density: float = 0.3, flag_below: float = 0.7):
        if not _SPACY_AVAILABLE:
            raise ImportError(
                "spacy is required for PIIRiskMetric.\n"
                "pip install spacy && python -m spacy download en_core_web_sm"
            )
        self.max_density = max_density
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        pii = _detect_pii(text)
        total_hits = (
            pii["person_names"] + pii["org_names"]
            + pii["emails"] + pii["phones"] + pii["ssn"] + pii["ids"]
        )
        word_count = max(1, len(re.findall(r"[a-zA-Z]+", text)))
        return total_hits / word_count

    def _normalize(self, raw: float) -> float:
        return max(0.0, 1.0 - raw / self.max_density)

    def compute(self, question, context):
        if not question.text.strip():
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                rationale="Question text is empty.",
                flagged=True,
                metadata={"reason": "empty_text"},
            )
        text = question.text
        pii = _detect_pii(text)
        total_hits = (
            pii["person_names"] + pii["org_names"]
            + pii["emails"] + pii["phones"] + pii["ssn"] + pii["ids"]
        )
        word_count = max(1, len(re.findall(r"[a-zA-Z]+", text)))
        density = total_hits / word_count
        score = max(0.0, 1.0 - density / self.max_density)

        if total_hits == 0:
            rationale = f"No PII detected (score={score:.4f})."
        else:
            types = []
            if pii["person_names"]: types.append(f"{pii['person_names']} name(s)")
            if pii["org_names"]: types.append(f"{pii['org_names']} org(s)")
            if pii["emails"]: types.append(f"{pii['emails']} email(s)")
            if pii["phones"]: types.append(f"{pii['phones']} phone(s)")
            if pii["ssn"]: types.append(f"{pii['ssn']} SSN(s)")
            if pii["ids"]: types.append(f"{pii['ids']} ID(s)")
            rationale = f"PII detected: {', '.join(types)} (density={density:.4f}; score={score:.4f})."

        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=rationale,
            flagged=score < self.flag_below,
            metadata={
                "pii_hits": total_hits,
                "pii_density": density,
                "person_names": pii["person_names"],
                "org_names": pii["org_names"],
                "emails": pii["emails"],
                "phones": pii["phones"],
                "ssn": pii["ssn"],
                "ids": pii["ids"],
            },
        )

    def _build_result(self, score: float, raw: float) -> MetricResult:  # pragma: no cover
        raise NotImplementedError
