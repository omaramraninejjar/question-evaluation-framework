"""
Self-BLEU metric (Diversity).

Reference:
    Zhu, Y., Lu, S., Zheng, L., Shi, J., Zhang, R., Wang, Y., & Lyu, M. R.
    (2018). Texygen: A benchmarking platform for text generation models.
    SIGIR 2018, 1097–1100. https://doi.org/10.1145/3209978.3210080

    Li, J., Galley, M., Brockett, C., Gao, J., & Dolan, B. (2016).
    A diversity-promoting objective function for neural conversation models.
    NAACL-HLT 2016, 110–119. https://aclanthology.org/N16-1014/

What it measures:
    Lexical overlap between the current question and a batch of other
    generated questions supplied via context.metadata["question_batch"]
    (a list of question strings).

        self_bleu_i = avg BLEU(question_i, question_j)  for j ≠ i

    A high Self-BLEU value means the question is very similar to others in
    the batch — indicating low generation diversity. A low value means the
    question is lexically distinct from its peers.

    score = 1.0 − self_bleu   (higher score → more distinct → better diversity)

    When no batch is provided, the question is assumed to be fully distinct
    and score = 1.0 is returned (not flagged).

    BLEU implementation: 1-gram + 2-gram modified precision with brevity
    penalty (no external dependencies).

Score:
    score ∈ [0.0, 1.0]
    Higher score → lower overlap with batch → more lexically diverse question.
    flag_below default 0.3 (avg BLEU ≥ 0.7 with peers → very low diversity).

Dependency:
    None — pure Python, no external packages required.
    Requires context.metadata["question_batch"] (list[str]) at evaluation time.
"""

from __future__ import annotations
import re
import math
import logging
from collections import Counter
from src.models import Question, EvaluationContext, MetricResult
from src.metrics.base import BaseMetric

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def _modified_precision(hypothesis: list[str], reference: list[str], n: int) -> float:
    """Modified n-gram precision with clipping."""
    h_grams = [tuple(hypothesis[i: i + n]) for i in range(len(hypothesis) - n + 1)]
    r_grams = [tuple(reference[i: i + n]) for i in range(len(reference) - n + 1)]
    if not h_grams:
        return 0.0
    r_counts = Counter(r_grams)
    h_counts = Counter(h_grams)
    clipped = sum(min(cnt, r_counts[gram]) for gram, cnt in h_counts.items())
    return clipped / len(h_grams)


def _bleu_2(hypothesis: str, reference: str) -> float:
    """
    Sentence-level 2-gram BLEU (1-gram + 2-gram, equal weights) with brevity
    penalty.
    """
    h_tok = _tokenize(hypothesis)
    r_tok = _tokenize(reference)
    if not h_tok or not r_tok:
        return 0.0
    bp = min(1.0, len(h_tok) / len(r_tok))
    p1 = _modified_precision(h_tok, r_tok, 1)
    p2 = _modified_precision(h_tok, r_tok, 2)
    if p1 <= 0.0 or p2 <= 0.0:
        return 0.0
    return bp * math.exp(0.5 * math.log(p1) + 0.5 * math.log(p2))


class SelfBLEUMetric(BaseMetric):
    """
    Self-BLEU: average lexical overlap between the question and a batch of
    other generated questions.

    Higher score → more distinct from peers → greater generation diversity.
    Returns score=1.0 when no question_batch is provided.

    Args:
        flag_below : Score threshold below which flagged=True (default 0.3).
    """

    name = "self_bleu"
    description = (
        "Self-BLEU diversity: 1 − avg BLEU with question batch "
        "(higher score = more distinct = greater lexical diversity)."
    )

    def __init__(self, flag_below: float = 0.3):
        self.flag_below = flag_below

    def compute(self, question: Question, context: EvaluationContext) -> MetricResult:
        if not question.text.strip():
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                rationale="Question text is empty.",
                flagged=True,
                metadata={"reason": "empty_text"},
            )

        batch: list[str] = context.metadata.get("question_batch", [])
        batch = [q for q in batch if isinstance(q, str) and q.strip()
                 and q.strip() != question.text.strip()]

        if not batch:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                rationale="No question_batch provided — assuming fully distinct (score=1.0).",
                flagged=False,
                metadata={"batch_size": 0, "avg_bleu": 0.0, "self_bleu_score": 1.0},
            )

        bleu_scores = [_bleu_2(question.text, ref) for ref in batch]
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        score = max(0.0, 1.0 - avg_bleu)

        return MetricResult(
            metric_name=self.name,
            score=score,
            rationale=(
                f"Avg BLEU with {len(batch)} peer question(s): {avg_bleu:.4f} "
                f"→ Self-BLEU diversity score {score:.4f}."
            ),
            flagged=score < self.flag_below,
            metadata={
                "batch_size": len(batch),
                "avg_bleu": avg_bleu,
                "self_bleu_score": score,
            },
        )
