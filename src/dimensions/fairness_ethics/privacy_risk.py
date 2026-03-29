"""
Privacy Risk dimension (Fairness & Ethics aspect).

Definition: Risk that the item reveals, requires disclosure of, or could
be used to infer personally identifiable information (PII) about the
respondent or third parties. Relevant under data-protection regulations
such as FERPA, GDPR, and COPPA.

Metrics wired:
    Optional — require spacy >= 3.0 + en_core_web_sm model:
        pii_risk         — PII density (NER + regex: names, emails, phones, SSNs)
        k_anonymity_risk — re-identification risk from quasi-identifier combination
        dp_epsilon_risk  — DP ε-proxy: information content of identifying attributes
"""

from src.models import DimensionName
from src.dimensions.base import BaseDimension

from src.metrics.pii_risk import PIIRiskMetric, _SPACY_AVAILABLE
from src.metrics.k_anonymity_risk import KAnonymityRiskMetric
from src.metrics.dp_epsilon_risk import DPEpsilonRiskMetric


class PrivacyRiskDimension(BaseDimension):
    name = DimensionName.PRIVACY_RISK
    description = (
        "Risk that the item reveals or requires disclosure of personally "
        "identifiable information (PII), subject to FERPA/GDPR/COPPA."
    )
    metrics = []

    def __init__(self):
        _metrics = []

        if _SPACY_AVAILABLE:
            _metrics.append(PIIRiskMetric())
            _metrics.append(KAnonymityRiskMetric())
            _metrics.append(DPEpsilonRiskMetric())

        self.metrics = _metrics
