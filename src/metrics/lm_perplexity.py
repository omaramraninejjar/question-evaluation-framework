"""
Language Model Perplexity metric (Well-Formedness).

Reference:
    Brown, T. B., Mann, B., Ryder, N., et al. (2020). Language Models are
    Few-Shot Learners. NeurIPS 2020. https://arxiv.org/abs/2005.14165

    Lau, J. H., & Baldwin, T. (2016). An empirical evaluation of various
    ranking techniques for unsupervised and supervised sentence acceptability
    prediction. ACL 2016 Workshop on Cognitive Aspects of Computational
    Language Learning. https://aclanthology.org/W16-1919/

What it measures:
    GPT-2 perplexity of the question text. Perplexity measures how
    "surprised" a language model is by the text:

        PPL = exp(−(1/N) × Σ log P(token_i | context))

    Lower perplexity → text is more consistent with the model's prior over
    natural English → more fluent, grammatically well-formed sentence.

    Typical GPT-2 perplexity ranges:
        ≤  50   Very fluent, natural English
        50–200  Normal prose, educational text
        200–500 Awkward phrasing, potential grammar issues
        ≥ 500   Likely malformed or highly unusual text

Score normalisation:
    score = max(0.0, 1.0 − log(ppl) / log(max_ppl))
    Default max_ppl = 1000.0.
    Higher score → lower perplexity → more well-formed text.
    flag_below default 0.3 (ppl > ~200 for max_ppl=1000).

Dependency:
    transformers >= 4.0   (already a core requirement)
    torch >= 2.1          (already a core requirement)
    Model gpt2 (~500 MB, downloaded from HuggingFace on first use)
"""

from __future__ import annotations
import math
import logging
from src.models import MetricResult
from src.metrics.base_readability import BaseReadabilityMetric

logger = logging.getLogger(__name__)

try:
    import torch as _torch
    from transformers import GPT2LMHeadModel as _GPT2LMHeadModel
    from transformers import GPT2TokenizerFast as _GPT2TokenizerFast
    _LM_AVAILABLE = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _GPT2LMHeadModel = None  # type: ignore[assignment]
    _GPT2TokenizerFast = None  # type: ignore[assignment]
    _LM_AVAILABLE = False

_tokenizer = None
_model = None


def _get_model():
    global _tokenizer, _model
    if _model is None:
        logger.info("Loading GPT-2 for LM perplexity (first use — ~500 MB download)…")
        _tokenizer = _GPT2TokenizerFast.from_pretrained("gpt2")
        _model = _GPT2LMHeadModel.from_pretrained("gpt2")
        _model.eval()
    return _tokenizer, _model


class LMPerplexityMetric(BaseReadabilityMetric):
    """
    GPT-2 perplexity as a well-formedness proxy.

    Higher score → lower perplexity → more fluent/grammatical text.
    Model gpt2 is downloaded on first use (~500 MB, cached in ~/.cache).

    Args:
        max_ppl    : Perplexity that maps to score 0.0 (default 1000.0).
        flag_below : Score threshold below which flagged=True (default 0.3).
    """

    name = "lm_perplexity"
    description = (
        "GPT-2 perplexity (log-normalised) "
        "(higher score = lower perplexity = more fluent, well-formed text)."
    )

    def __init__(self, max_ppl: float = 1000.0, flag_below: float = 0.3):
        if not _LM_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for LMPerplexityMetric.\n"
                "pip install transformers torch"
            )
        self.max_ppl = max_ppl
        self.flag_below = flag_below

    def _compute_raw(self, text: str) -> float:
        tokenizer, model = _get_model()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with _torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss  # average NLL per token
        return math.exp(loss.item())

    def _normalize(self, raw: float) -> float:
        if raw <= 1.0:
            return 1.0
        return max(0.0, 1.0 - math.log(raw) / math.log(self.max_ppl))

    def _build_result(self, score: float, raw: float) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"GPT-2 perplexity {raw:.1f} "
                f"(normalised {score:.4f}; max_ppl={self.max_ppl})."
            ),
            flagged=score < self.flag_below,
            metadata={"perplexity": raw, "max_ppl": self.max_ppl},
        )
