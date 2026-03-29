"""
BM25 (Best Match 25) metric.

Original paper:
    Robertson & Walker (1994). Some simple effective approximations to the
    2-Poisson model for probabilistic weighted retrieval.
    SIGIR 1994.
    https://dl.acm.org/doi/10.1145/188490.188561

    Full BM25 formulation (with IDF):
    Robertson et al. (1995). Okapi at TREC-3. NIST Special Publication.

What it measures:
    Retrieval relevance of the candidate question against each reference text.
    BM25 (Okapi variant) combines term frequency saturation (terms that repeat
    many times in a document are up-weighted, but with diminishing returns) and
    inverse document frequency (rare terms in the corpus are more informative).
    It is widely used in information retrieval and outperforms raw TF-IDF for
    longer documents.

    In this framework the candidate question acts as the query and each
    reference text acts as a "document". A high score means the question
    contains the same key terms as the reference — a strong signal for concept
    coverage and curriculum alignment.

Score normalization:
    Raw BM25 scores are non-negative but unbounded. They are linearly mapped to
    [0, 1] using a configurable norm_factor:
        normalised = min(1.0, raw_score / norm_factor)
    The default norm_factor=10.0 is suitable for short-to-medium texts
    (typical learning objectives and course paragraphs). Increase it if your
    references are very long and scores regularly exceed 10.

Use in this framework:
    Curriculum Alignment — references = context.learning_objectives
    Concept Coverage     — references = [context.course_content]

Aggregation over multiple references:
    The candidate is scored against each reference independently;
    the maximum score across references is returned.

Dependency:
    rank-bm25 >= 0.2   (pip install rank-bm25)
"""

from __future__ import annotations
import logging
from src.models import Question, EvaluationContext, MetricResult
from src.metrics.base import BaseReferenceMetric

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25Okapi = None  # type: ignore[assignment,misc]
    _BM25_AVAILABLE = False


class BM25Metric(BaseReferenceMetric):
    """
    BM25 retrieval score between question.text and context references.

    Args:
        reference_source : Which context field to use as references.
            "learning_objectives" → context.learning_objectives  (list[str])
            "course_content"      → context.course_content       (str)
        norm_factor      : Divisor for raw BM25 score normalisation.
            raw scores >= norm_factor are clamped to 1.0.
            Default 10.0 — suitable for short educational texts.
        flag_below       : Score threshold below which flagged=True.
    """

    name = "bm25"
    description = (
        "BM25 retrieval relevance between question text and context references "
        "(Okapi BM25, max over references)."
    )

    def __init__(
        self,
        reference_source: str = "learning_objectives",
        norm_factor: float = 10.0,
        flag_below: float = 0.3,
    ):
        if not _BM25_AVAILABLE:
            raise ImportError(
                "rank-bm25 is required for BM25Metric.\n"
                "pip install rank-bm25"
            )
        if norm_factor <= 0:
            raise ValueError("norm_factor must be positive")
        self.reference_source = reference_source
        self.norm_factor = norm_factor
        self.flag_below = flag_below

    # ------------------------------------------------------------------

    def compute(self, question: Question, context: EvaluationContext) -> MetricResult:
        references = self._collect_references(context)

        if not references:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                rationale=f"No references found in context.{self.reference_source}.",
                flagged=True,
                metadata={"reason": "no_references", "reference_source": self.reference_source},
            )

        if not question.text.strip():
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                rationale="Question text is empty.",
                flagged=True,
                metadata={"reason": "empty_candidate"},
            )

        # Tokenize: simple whitespace split on lowercased text.
        tokenized_corpus = [ref.lower().split() for ref in references]
        query_tokens = question.text.lower().split()

        bm25 = _BM25Okapi(tokenized_corpus)
        raw_scores = bm25.get_scores(query_tokens)  # shape: (n_references,)

        best_raw = float(max(raw_scores))
        best_score = min(1.0, best_raw / self.norm_factor) if best_raw > 0.0 else 0.0

        quality = "strong" if best_score >= 0.6 else "moderate" if best_score >= 0.3 else "weak"

        return MetricResult(
            metric_name=self.name,
            score=best_score,
            rationale=(
                f"BM25 {best_score:.4f} (raw: {best_raw:.4f}, {quality} retrieval relevance) "
                f"vs {len(references)} reference(s) from context.{self.reference_source}."
            ),
            flagged=best_score < self.flag_below,
            metadata={
                "reference_source": self.reference_source,
                "n_references": len(references),
                "raw_score": best_raw,
                "norm_factor": self.norm_factor,
            },
        )
