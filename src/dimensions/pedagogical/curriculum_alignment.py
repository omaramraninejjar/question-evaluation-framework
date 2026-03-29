"""
Curriculum Alignment dimension (Pedagogical aspect).

Definition:
    Extent to which the item matches intended learning objectives.

Metrics (all scored against context.learning_objectives):
    Always loaded:
        bleu      — n-gram overlap (NLTK)
        rouge_l   — LCS overlap (rouge-score)
        meteor    — alignment-based F-mean with fragmentation penalty (NLTK)
        chrf      — character n-gram F-score (sacrebleu)
        tfidf     — TF-IDF keyword-weighted cosine similarity (scikit-learn)
        bm25      — BM25 retrieval relevance (rank-bm25)
    Loaded when available:
        bertscore       — contextual token-level embedding similarity (bert-score + torch)
        bleurt          — learned similarity from human judgments (evaluate + tensorflow + bleurt)
        bartscore       — BART seq2seq log-likelihood (transformers + torch)
        comet           — cross-lingual human-judgment trained (unbabel-comet + torch)
        sbert           — sentence-level cosine similarity (sentence-transformers + torch)
        nli_entailment  — NLI entailment probability (sentence-transformers + torch)
"""

from src.models import DimensionName
from src.dimensions.base import BaseDimension
from src.metrics.bleu import BLEUMetric
from src.metrics.rouge import ROUGELMetric
from src.metrics.meteor import METEORMetric
from src.metrics.chrf import chrFMetric, _CHRF_AVAILABLE
from src.metrics.tfidf import TFIDFMetric, _TFIDF_AVAILABLE
from src.metrics.bm25 import BM25Metric, _BM25_AVAILABLE
from src.metrics.bertscore import BERTScoreMetric, _BERT_SCORE_AVAILABLE
from src.metrics.bleurt import BLEURTMetric, _BLEURT_AVAILABLE
from src.metrics.bartscore import BARTScoreMetric, _BART_SCORE_AVAILABLE
from src.metrics.comet import COMETMetric, _COMET_AVAILABLE
from src.metrics.sentencebert import SentenceBERTMetric, _SBERT_AVAILABLE
from src.metrics.nli import NLIEntailmentMetric, _NLI_AVAILABLE

_SRC = "learning_objectives"


class CurriculumAlignmentDimension(BaseDimension):
    name = DimensionName.CURRICULUM_ALIGNMENT
    description = "Extent to which the item matches intended learning objectives."
    metrics = []  # overridden per-instance in __init__

    def __init__(self) -> None:
        self.metrics = [
            BLEUMetric(reference_source=_SRC),
            ROUGELMetric(reference_source=_SRC),
            METEORMetric(reference_source=_SRC),
        ]
        if _CHRF_AVAILABLE:
            self.metrics.append(chrFMetric(reference_source=_SRC))
        if _TFIDF_AVAILABLE:
            self.metrics.append(TFIDFMetric(reference_source=_SRC))
        if _BM25_AVAILABLE:
            self.metrics.append(BM25Metric(reference_source=_SRC))
        if _BERT_SCORE_AVAILABLE:
            self.metrics.append(BERTScoreMetric(reference_source=_SRC))
        if _BLEURT_AVAILABLE:
            self.metrics.append(BLEURTMetric(reference_source=_SRC))
        if _BART_SCORE_AVAILABLE:
            self.metrics.append(BARTScoreMetric(reference_source=_SRC))
        if _COMET_AVAILABLE:
            self.metrics.append(COMETMetric(reference_source=_SRC))
        if _SBERT_AVAILABLE:
            self.metrics.append(SentenceBERTMetric(reference_source=_SRC))
        if _NLI_AVAILABLE:
            self.metrics.append(NLIEntailmentMetric(reference_source=_SRC))
