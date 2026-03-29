"""
Sentence-BERT (SBERT) Cosine Similarity metric.

Original paper:
    Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using
    Siamese BERT-Networks. EMNLP 2019.
    https://aclanthology.org/D19-1410/

What it measures:
    Cosine similarity between dense sentence-level embeddings produced by a
    Siamese BERT network fine-tuned for semantic textual similarity (STS).
    Unlike BERTScore (which operates token-by-token) or TF-IDF (which is
    bag-of-words), SBERT encodes entire sentences into a single fixed-size
    vector, capturing global meaning and paraphrase equivalence.

    This makes it particularly effective for Curriculum Alignment (does the
    question semantically match the learning objective?) and Concept Coverage
    (does the question relate to the course content at sentence level?).

Score range:
    Cosine similarity is in [−1, 1]; scores are clipped to [0.0, 1.0].
    Scores for thematically related educational texts typically fall in
    [0.3, 0.9]; identical sentences score 1.0.

Aggregation over multiple references:
    The candidate is encoded once; each reference is encoded once.
    The maximum cosine similarity across references is returned.

Model choice:
    model_name defaults to "all-MiniLM-L6-v2" (~80 MB, recommended for speed).
    Use "all-mpnet-base-v2" (~420 MB) for higher accuracy.
    Both are downloaded on first use and cached in ~/.cache/huggingface.

Dependency:
    sentence-transformers >= 2.2   (pip install sentence-transformers)
    torch >= 2.1                   (already required by bert-score)
"""

from __future__ import annotations
import logging
from src.models import Question, EvaluationContext, MetricResult
from src.metrics.base import BaseReferenceMetric

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    from sentence_transformers import util as _st_util
    _SBERT_AVAILABLE = True
except ImportError:
    _SentenceTransformer = None  # type: ignore[assignment,misc]
    _st_util = None              # type: ignore[assignment]
    _SBERT_AVAILABLE = False


class SentenceBERTMetric(BaseReferenceMetric):
    """
    Sentence-BERT cosine similarity between question.text and context references.

    Args:
        reference_source : Which context field to use as references.
            "learning_objectives" → context.learning_objectives  (list[str])
            "course_content"      → context.course_content       (str)
        model_name       : sentence-transformers model identifier.
            Default "all-MiniLM-L6-v2" (~80 MB, fast and accurate).
            "all-mpnet-base-v2" (~420 MB) gives higher accuracy.
        flag_below       : Score threshold below which flagged=True.
    """

    name = "sbert"
    description = (
        "Sentence-BERT cosine similarity between question text and context references "
        "(sentence-level semantic similarity, max over references)."
    )

    def __init__(
        self,
        reference_source: str = "learning_objectives",
        model_name: str = "all-MiniLM-L6-v2",
        flag_below: float = 0.5,
    ):
        if not _SBERT_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for SentenceBERTMetric.\n"
                "pip install sentence-transformers"
            )
        self.reference_source = reference_source
        self.model_name = model_name
        self.flag_below = flag_below
        self._model = None  # lazy-loaded on first compute()

    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            logger.info("Loading Sentence-BERT model %r …", self.model_name)
            self._model = _SentenceTransformer(self.model_name)
        return self._model

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

        model = self._get_model()

        # Encode candidate + all references in a single batch for efficiency.
        all_texts = [question.text] + references
        embeddings = model.encode(all_texts, convert_to_tensor=True)
        candidate_emb = embeddings[0:1]
        reference_embs = embeddings[1:]

        sims = _st_util.cos_sim(candidate_emb, reference_embs).flatten()
        best_raw = float(sims.max().item())
        best_score = min(max(best_raw, 0.0), 1.0)  # clip to [0, 1]

        quality = "strong" if best_score >= 0.8 else "moderate" if best_score >= 0.5 else "weak"

        return MetricResult(
            metric_name=self.name,
            score=best_score,
            rationale=(
                f"Sentence-BERT cosine {best_score:.4f} (raw: {best_raw:.4f}, "
                f"{quality} semantic similarity) "
                f"vs {len(references)} reference(s) from context.{self.reference_source} "
                f"[{self.model_name}]."
            ),
            flagged=best_score < self.flag_below,
            metadata={
                "reference_source": self.reference_source,
                "model_name": self.model_name,
                "n_references": len(references),
                "raw_score": best_raw,
            },
        )
