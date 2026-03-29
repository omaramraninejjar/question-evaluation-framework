"""
Well-Formedness dimension (Linguistic & Structural aspect).

Definition: Grammatical, syntactic, and structural correctness of the
item. Covers agreement errors, punctuation, consistent option formatting,
absence of clang associations or grammatical cues that inadvertently
signal the correct answer.

Metrics wired:
    Always available (no extra deps):
        question_mark, negation_rate

    Core NLTK (always available):
        has_verb

    Optional — require spacy >= 3.0 + en_core_web_sm model:
        subject_verb_agreement, question_type_consistency

    Optional — require transformers + torch (gpt2 ~500 MB on first use):
        lm_perplexity
"""

from src.models import DimensionName
from src.dimensions.base import BaseDimension

from src.metrics.question_mark import QuestionMarkMetric
from src.metrics.negation_rate import NegationRateMetric
from src.metrics.has_verb import HasVerbMetric
from src.metrics.subject_verb_agreement import SubjectVerbAgreementMetric, _SPACY_AVAILABLE
from src.metrics.question_type_consistency import QuestionTypeConsistencyMetric
from src.metrics.lm_perplexity import LMPerplexityMetric, _LM_AVAILABLE


class WellFormednessDimension(BaseDimension):
    name = DimensionName.WELL_FORMEDNESS
    description = (
        "Grammatical, syntactic, and structural correctness of the item, "
        "including formatting consistency and absence of unintended cues."
    )
    metrics = []

    def __init__(self):
        _metrics = [
            QuestionMarkMetric(),
            NegationRateMetric(),
            HasVerbMetric(),
        ]

        if _SPACY_AVAILABLE:
            _metrics.append(SubjectVerbAgreementMetric())
            _metrics.append(QuestionTypeConsistencyMetric())

        if _LM_AVAILABLE:
            _metrics.append(LMPerplexityMetric())

        self.metrics = _metrics
