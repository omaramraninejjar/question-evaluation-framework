"""
Response Burden dimension (Pedagogical aspect).

Definition: Cognitive and time load placed on the respondent to process
the item and produce an answer, independent of the construct being measured.
High burden may introduce construct-irrelevant variance.

Metrics wired:
    Always available (no extra deps):
        openendedness   — expected response format (closed vs. open-ended)
        sqp_score       — Survey Quality Predictor: double-barrel, leading,
                          loaded language, vague frequency anchors

    Always available (nltk):
        multipart_question — number of distinct sub-tasks in the stem
        stem_complexity    — word count + preamble length + embedded clauses
"""

from src.models import DimensionName
from src.dimensions.base import BaseDimension

from src.metrics.openendedness import OpenEndednessMetric
from src.metrics.sqp_score import SQPScoreMetric
from src.metrics.multipart_question import MultiPartQuestionMetric
from src.metrics.stem_complexity import StemComplexityMetric


class ResponseBurdenDimension(BaseDimension):
    name = DimensionName.RESPONSE_BURDEN
    description = (
        "Cognitive and time load placed on the respondent to process the item "
        "and produce an answer, independent of the construct being measured."
    )
    metrics = []

    def __init__(self):
        self.metrics = [
            OpenEndednessMetric(),
            SQPScoreMetric(),
            MultiPartQuestionMetric(),
            StemComplexityMetric(),
        ]
