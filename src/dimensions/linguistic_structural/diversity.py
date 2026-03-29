"""
Diversity dimension (Linguistic & Structural aspect).

Definition: Representativeness of varied item formats, surface forms,
and content perspectives within a question set. Low diversity may
introduce test-taking fatigue, response sets, or systematic gaps in
construct coverage.

Metrics wired:
    Always available (no extra deps):
        distinct_1, distinct_2, vocabulary_novelty

    Context-dependent (requires context.metadata["question_batch"]):
        self_bleu — lexical overlap with a batch of peer questions;
                    returns score=1.0 when no batch is supplied.
"""

from src.models import DimensionName
from src.dimensions.base import BaseDimension

from src.metrics.distinct_n import DistinctNMetric
from src.metrics.vocabulary_novelty import VocabularyNoveltyMetric
from src.metrics.self_bleu import SelfBLEUMetric


class DiversityDimension(BaseDimension):
    name = DimensionName.DIVERSITY
    description = (
        "Representativeness of varied item formats, surface forms, and content "
        "perspectives within a question set."
    )
    metrics = []

    def __init__(self):
        self.metrics = [
            DistinctNMetric(n=1),        # unigram diversity within the question
            DistinctNMetric(n=2),        # bigram diversity within the question
            VocabularyNoveltyMetric(),   # fraction of words novel vs. course content
            SelfBLEUMetric(),            # overlap with peer questions (batch-level)
        ]
