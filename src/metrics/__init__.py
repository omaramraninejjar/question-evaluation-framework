"""
Public API for all implemented metrics.

Core reference-based (always available — nltk, rouge-score, sacrebleu,
                       scikit-learn, rank-bm25 are required deps):
    BLEUMetric, ROUGELMetric, METEORMetric, chrFMetric, TFIDFMetric, BM25Metric

Optional reference-based (require extra deps; guarded by _AVAILABLE flag):
    BERTScoreMetric     — requires bert-score + torch
    BLEURTMetric        — requires evaluate + tensorflow + bleurt
    BARTScoreMetric     — requires transformers + torch
    COMETMetric         — requires unbabel-comet + torch
    SentenceBERTMetric  — requires sentence-transformers + torch
    NLIEntailmentMetric — requires sentence-transformers + torch

Readability — textstat formula-based (optional; guarded by _TEXTSTAT_AVAILABLE):
    FleschReadingEaseMetric, FleschKincaidMetric, DaleChallMetric,
    GunningFogMetric, ColemanLiauMetric, ARIMetric, SMOGMetric,
    LinsearWriteMetric, SpacheScoreMetric

Readability — count-based (no external deps):
    WordCountMetric, SentenceCountMetric, AvgWordLengthMetric,
    AvgSyllablesMetric, LongWordRatioMetric, TypeTokenRatioMetric

Readability — nltk-based (always available):
    LexicalDensityMetric

Linguistic Complexity (optional; guarded by _WORDFREQ_AVAILABLE / _SPACY_AVAILABLE):
    ZipfWordFrequencyMetric, ParseTreeDepthMetric, DependencyDistanceMetric,
    ConstituentCountMetric, PassiveVoiceRateMetric

Linguistic Complexity — nltk-based (always available):
    ConjunctionRateMetric

Ambiguity (nltk always available; wordnet corpus required for polysemy;
           spacy required for negation_ambiguity):
    WHWordTypeMetric, PronounRatioMetric, QuantifierRateMetric,
    PolysemyScoreMetric, NegationAmbiguityScoreMetric

Well-Formedness (nltk always available; spacy for agreement/consistency;
                 transformers for LM perplexity):
    QuestionMarkMetric, NegationRateMetric, HasVerbMetric,
    SubjectVerbAgreementMetric, QuestionTypeConsistencyMetric,
    LMPerplexityMetric

Diversity (no extra deps; self_bleu uses context.metadata["question_batch"]):
    DistinctNMetric, VocabularyNoveltyMetric, SelfBLEUMetric

Cognitive Demand — no extra deps:
    BloomLevelMetric, DOKLevelMetric, HOTSKeywordsMetric

Response Burden — no extra deps / nltk:
    OpenEndednessMetric, SQPScoreMetric, MultiPartQuestionMetric,
    StemComplexityMetric

Privacy Risk — spacy required:
    PIIRiskMetric, KAnonymityRiskMetric, DPEpsilonRiskMetric
"""

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

# Readability — textstat formula metrics
from src.metrics.flesch_ease import FleschReadingEaseMetric
from src.metrics.flesch_kincaid import FleschKincaidMetric
from src.metrics.dale_chall import DaleChallMetric
from src.metrics.gunning_fog import GunningFogMetric
from src.metrics.coleman_liau import ColemanLiauMetric
from src.metrics.ari import ARIMetric
from src.metrics.smog import SMOGMetric
from src.metrics.linsear_write import LinsearWriteMetric
from src.metrics.spache import SpacheScoreMetric
from src.metrics.flesch_ease import _TEXTSTAT_AVAILABLE  # shared flag

# Readability — count metrics (no extra deps)
from src.metrics.text_counts import (
    WordCountMetric,
    SentenceCountMetric,
    AvgWordLengthMetric,
    AvgSyllablesMetric,
    LongWordRatioMetric,
    TypeTokenRatioMetric,
)

# Readability — nltk-based (always available)
from src.metrics.lexical_density import LexicalDensityMetric

# Linguistic Complexity metrics
from src.metrics.conjunction_rate import ConjunctionRateMetric
from src.metrics.zipf_frequency import ZipfWordFrequencyMetric, _WORDFREQ_AVAILABLE
from src.metrics.parse_tree_depth import ParseTreeDepthMetric, _SPACY_AVAILABLE
from src.metrics.dependency_distance import DependencyDistanceMetric
from src.metrics.constituent_count import ConstituentCountMetric
from src.metrics.passive_voice_rate import PassiveVoiceRateMetric

# Ambiguity metrics
from src.metrics.wh_word_type import WHWordTypeMetric
from src.metrics.pronoun_ratio import PronounRatioMetric
from src.metrics.quantifier_rate import QuantifierRateMetric
from src.metrics.polysemy import PolysemyScoreMetric, _NLTK_WN_AVAILABLE
from src.metrics.negation_ambiguity import NegationAmbiguityScoreMetric

# Well-Formedness metrics
from src.metrics.question_mark import QuestionMarkMetric
from src.metrics.negation_rate import NegationRateMetric
from src.metrics.has_verb import HasVerbMetric
from src.metrics.subject_verb_agreement import SubjectVerbAgreementMetric
from src.metrics.question_type_consistency import QuestionTypeConsistencyMetric
from src.metrics.lm_perplexity import LMPerplexityMetric, _LM_AVAILABLE

# Diversity metrics
from src.metrics.distinct_n import DistinctNMetric
from src.metrics.vocabulary_novelty import VocabularyNoveltyMetric
from src.metrics.self_bleu import SelfBLEUMetric

# Cognitive Demand metrics (always available — no extra deps)
from src.metrics.bloom_level import BloomLevelMetric
from src.metrics.dok_level import DOKLevelMetric
from src.metrics.hots_keywords import HOTSKeywordsMetric

# Response Burden metrics (no extra deps / nltk)
from src.metrics.openendedness import OpenEndednessMetric
from src.metrics.sqp_score import SQPScoreMetric
from src.metrics.multipart_question import MultiPartQuestionMetric
from src.metrics.stem_complexity import StemComplexityMetric

# Privacy Risk metrics (spaCy required)
from src.metrics.pii_risk import PIIRiskMetric
from src.metrics.k_anonymity_risk import KAnonymityRiskMetric
from src.metrics.dp_epsilon_risk import DPEpsilonRiskMetric

__all__ = [
    # Core reference-based — always available
    "BLEUMetric",
    "ROUGELMetric",
    "METEORMetric",
    "chrFMetric",
    "TFIDFMetric",
    "BM25Metric",
    # Optional reference-based — guarded by _AVAILABLE flags
    "BERTScoreMetric",
    "BLEURTMetric",
    "BARTScoreMetric",
    "COMETMetric",
    "SentenceBERTMetric",
    "NLIEntailmentMetric",
    # Readability — textstat formula metrics (optional)
    "FleschReadingEaseMetric",
    "FleschKincaidMetric",
    "DaleChallMetric",
    "GunningFogMetric",
    "ColemanLiauMetric",
    "ARIMetric",
    "SMOGMetric",
    "LinsearWriteMetric",
    "SpacheScoreMetric",
    # Readability — count metrics (always available)
    "WordCountMetric",
    "SentenceCountMetric",
    "AvgWordLengthMetric",
    "AvgSyllablesMetric",
    "LongWordRatioMetric",
    "TypeTokenRatioMetric",
    # Readability — nltk-based (always available)
    "LexicalDensityMetric",
    # Linguistic Complexity metrics
    "ConjunctionRateMetric",
    "ZipfWordFrequencyMetric",
    "ParseTreeDepthMetric",
    "DependencyDistanceMetric",
    "ConstituentCountMetric",
    "PassiveVoiceRateMetric",
    # Ambiguity metrics
    "WHWordTypeMetric",
    "PronounRatioMetric",
    "QuantifierRateMetric",
    "PolysemyScoreMetric",
    "NegationAmbiguityScoreMetric",
    # Well-Formedness metrics
    "QuestionMarkMetric",
    "NegationRateMetric",
    "HasVerbMetric",
    "SubjectVerbAgreementMetric",
    "QuestionTypeConsistencyMetric",
    "LMPerplexityMetric",
    # Diversity metrics
    "DistinctNMetric",
    "VocabularyNoveltyMetric",
    "SelfBLEUMetric",
    # Cognitive Demand metrics (always available)
    "BloomLevelMetric",
    "DOKLevelMetric",
    "HOTSKeywordsMetric",
    # Response Burden metrics (always available)
    "OpenEndednessMetric",
    "SQPScoreMetric",
    "MultiPartQuestionMetric",
    "StemComplexityMetric",
    # Privacy Risk metrics (spaCy required)
    "PIIRiskMetric",
    "KAnonymityRiskMetric",
    "DPEpsilonRiskMetric",
    # Availability flags
    "_CHRF_AVAILABLE",
    "_TFIDF_AVAILABLE",
    "_BM25_AVAILABLE",
    "_BERT_SCORE_AVAILABLE",
    "_BLEURT_AVAILABLE",
    "_BART_SCORE_AVAILABLE",
    "_COMET_AVAILABLE",
    "_SBERT_AVAILABLE",
    "_NLI_AVAILABLE",
    "_TEXTSTAT_AVAILABLE",
    "_WORDFREQ_AVAILABLE",
    "_SPACY_AVAILABLE",
    "_NLTK_WN_AVAILABLE",
    "_LM_AVAILABLE",
]
