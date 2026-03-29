"""
COMET metric.

Original paper:
    Rei et al. (2020). COMET: A Neural Framework for MT Evaluation.
    EMNLP 2020.
    https://aclanthology.org/2020.emnlp-main.213

What it measures:
    Learned quality assessment between a candidate text and one or more
    reference texts, using a cross-lingual encoder (XLM-R) fine-tuned on
    human Direct Assessment (DA) scores. Unlike string-overlap metrics
    (BLEU, ROUGE) or contextual-embedding metrics (BERTScore), COMET is
    trained directly on human quality judgments and correlates strongly
    with them across diverse text types.

    In this framework the learning objective / course content serves as
    both the "source" and the "reference": we ask the model how well the
    candidate question expresses the target concept.

Score range:
    wmt22-comet-da scores are calibrated to the DA scale and are
    approximately in [0, 1]. Scores are clipped to [0.0, 1.0] for safety.

Aggregation over multiple references:
    The candidate is scored against each reference independently;
    the maximum score across references is returned.

Model choice:
    model_name defaults to "Unbabel/wmt22-comet-da" (~1.9 GB, downloaded
    on first use and cached in ~/.cache/huggingface). Use
    "Unbabel/wmt20-comet-da" as a lighter alternative (~1.6 GB).

Dependency:
    unbabel-comet >= 2.0   (pip install unbabel-comet)
    torch >= 2.1           (already required by bert-score)
"""

from __future__ import annotations
import logging
from src.models import Question, EvaluationContext, MetricResult
from src.metrics.base import BaseReferenceMetric

logger = logging.getLogger(__name__)

try:
    from comet import download_model as _comet_download, load_from_checkpoint as _comet_load
    _COMET_AVAILABLE = True
except Exception:  # ImportError or pkg_resources.DistributionNotFound (missing torchvision etc.)
    _comet_download = None  # type: ignore[assignment]
    _comet_load = None      # type: ignore[assignment]
    _COMET_AVAILABLE = False


class COMETMetric(BaseReferenceMetric):
    """
    COMET score between question.text and context references.

    Args:
        reference_source : Which context field to use as references.
            "learning_objectives" → context.learning_objectives  (list[str])
            "course_content"      → context.course_content       (str)
        model_name       : HuggingFace COMET model identifier.
            Default "Unbabel/wmt22-comet-da" (~1.9 GB, recommended).
            "Unbabel/wmt20-comet-da" (~1.6 GB) is a lighter alternative.
        batch_size       : Batch size for COMET inference (default 8).
        num_workers      : DataLoader workers passed to model.predict().
            Must be > 0 on macOS (pytorch-lightning 2.x sets
            multiprocessing_context='spawn', which conflicts with 0).
            Default 1; increase for faster CPU inference.
        flag_below       : Score threshold below which flagged=True.
    """

    name = "comet"
    description = (
        "Human-judgment-trained quality estimation via cross-lingual embeddings "
        "(COMET, max over references)."
    )

    def __init__(
        self,
        reference_source: str = "learning_objectives",
        model_name: str = "Unbabel/wmt22-comet-da",
        batch_size: int = 8,
        num_workers: int = 1,
        flag_below: float = 0.5,
    ):
        if not _COMET_AVAILABLE:
            raise ImportError(
                "unbabel-comet is required for COMETMetric.\n"
                "pip install unbabel-comet"
            )
        self.reference_source = reference_source
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.flag_below = flag_below
        self._model = None  # lazy-loaded on first compute()

    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            logger.info("Loading COMET model %r …", self.model_name)
            model_path = _comet_download(self.model_name)
            self._model = _comet_load(model_path)
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

        # COMET expects (src, mt, ref) triples.
        # For educational QG: reference acts as both source and reference;
        # question.text is the candidate ("machine translation").
        data = [
            {"src": ref, "mt": question.text, "ref": ref}
            for ref in references
        ]

        model_output = model.predict(
            data,
            batch_size=self.batch_size,
            gpus=0,                      # CPU-only
            num_workers=self.num_workers, # must be > 0 on macOS (PL 2.x + spawn)
            progress_bar=False,
        )

        best_raw = max(model_output.scores)
        best_score = min(max(best_raw, 0.0), 1.0)

        quality = "strong" if best_score >= 0.9 else "moderate" if best_score >= 0.75 else "weak"

        return MetricResult(
            metric_name=self.name,
            score=best_score,
            rationale=(
                f"COMET {best_score:.4f} (raw: {best_raw:.4f}, {quality} semantic quality) "
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
