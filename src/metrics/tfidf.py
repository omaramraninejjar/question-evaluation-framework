"""
TF-IDF Cosine Similarity metric.

Original reference:
    Salton & Buckley (1988). Term-weighting approaches in automatic text
    retrieval. Information Processing & Management, 24(5), 513–523.
    https://doi.org/10.1016/0306-4573(88)90021-0

What it measures:
    Cosine similarity between TF-IDF weighted bag-of-words vectors of the
    candidate question and each reference text.  TF-IDF down-weights common
    words (high document frequency) and up-weights rare, domain-specific terms,
    making it more discriminating than raw word-overlap metrics (BLEU, ROUGE)
    for vocabulary-heavy educational content.

    The vectorizer is fitted on the candidate + all references at compute time,
    so IDF weights are local to the comparison set.  Scores are in [0.0, 1.0];
    higher = greater keyword-weighted similarity.

Use in this framework:
    Curriculum Alignment — references = context.learning_objectives
    Concept Coverage     — references = [context.course_content]

Aggregation over multiple references:
    The candidate is scored against each reference independently;
    the maximum score across references is returned.

Dependency:
    scikit-learn >= 1.0   (pip install scikit-learn)
"""

from __future__ import annotations
import logging
from src.models import Question, EvaluationContext, MetricResult
from src.metrics.base import BaseReferenceMetric

logger = logging.getLogger(__name__)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
    _TFIDF_AVAILABLE = True
except ImportError:
    _TfidfVectorizer = None   # type: ignore[assignment,misc]
    _cosine_similarity = None  # type: ignore[assignment]
    _TFIDF_AVAILABLE = False


class TFIDFMetric(BaseReferenceMetric):
    """
    TF-IDF cosine similarity between question.text and context references.

    Args:
        reference_source : Which context field to use as references.
            "learning_objectives" → context.learning_objectives  (list[str])
            "course_content"      → context.course_content       (str)
        analyzer         : Tokenization unit passed to TfidfVectorizer.
            "word" (default) — word n-grams
            "char_wb"        — character n-grams within word boundaries
        ngram_range      : n-gram range for the vectorizer (default (1, 2),
                           unigrams + bigrams).
        flag_below       : Score threshold below which flagged=True.
    """

    name = "tfidf"
    description = (
        "TF-IDF cosine similarity between question text and context references "
        "(keyword-weighted, max over references)."
    )

    def __init__(
        self,
        reference_source: str = "learning_objectives",
        analyzer: str = "word",
        ngram_range: tuple[int, int] = (1, 2),
        flag_below: float = 0.3,
    ):
        if not _TFIDF_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for TFIDFMetric.\n"
                "pip install scikit-learn"
            )
        self.reference_source = reference_source
        self.analyzer = analyzer
        self.ngram_range = ngram_range
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

        # Fit the vectorizer on candidate + all references so IDF is local.
        corpus = [question.text] + references
        vectorizer = _TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)

        candidate_vec = tfidf_matrix[0:1]
        reference_vecs = tfidf_matrix[1:]
        sims = _cosine_similarity(candidate_vec, reference_vecs).flatten()
        best_score = float(sims.max())

        quality = "strong" if best_score >= 0.6 else "moderate" if best_score >= 0.3 else "weak"

        return MetricResult(
            metric_name=self.name,
            score=best_score,
            rationale=(
                f"TF-IDF cosine {best_score:.4f} ({quality} keyword-weighted similarity) "
                f"vs {len(references)} reference(s) from context.{self.reference_source}."
            ),
            flagged=best_score < self.flag_below,
            metadata={
                "reference_source": self.reference_source,
                "n_references": len(references),
                "analyzer": self.analyzer,
                "ngram_range": list(self.ngram_range),
            },
        )
