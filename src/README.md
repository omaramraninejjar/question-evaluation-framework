# Source Code (`src/`)

All framework source code lives here. The design follows a strict three-layer hierarchy: **Aspect → Dimension → Metric**. Each layer is independently testable and communicates only through the shared data models in `models.py`.

## Module map

```
src/
├── models.py                     # Shared data models (single source of truth)
├── evaluator.py                  # Top-level orchestrator
├── pipeline.py                   # Batch evaluation runner with caching
├── scorer.py                     # Post-evaluation aggregation strategies
├── reporter.py                   # Output formatting (JSON, CSV, …)
│
├── aspects/                      # Aspect layer — one class per aspect
│   ├── base.py                   # BaseAspect abstract class
│   ├── pedagogical.py
│   ├── psychometric.py
│   ├── linguistic_structural.py
│   └── fairness_ethics.py
│
├── dimensions/                   # Dimension layer — one module per dimension
│   ├── base.py                   # BaseDimension abstract class
│   ├── pedagogical/
│   │   ├── curriculum_alignment.py
│   │   ├── cognitive_demand.py
│   │   ├── concept_coverage.py
│   │   └── response_burden.py
│   ├── psychometric/
│   │   ├── difficulty.py
│   │   ├── discrimination.py
│   │   ├── guessing_careless.py
│   │   ├── distractor_functioning.py
│   │   ├── item_fit.py
│   │   ├── dimensionality.py
│   │   └── reliability.py
│   ├── linguistic_structural/
│   │   ├── readability.py
│   │   ├── linguistic_complexity.py
│   │   ├── ambiguity.py
│   │   ├── well_formedness.py
│   │   └── diversity.py
│   └── fairness_ethics/
│       ├── group_bias.py
│       ├── measurement_invariance.py
│       ├── content_sensitivity.py
│       ├── harmful_content_risk.py
│       └── privacy_risk.py
│
└── metrics/                      # Metric layer — one class per atomic metric
    ├── base.py                   # BaseMetric abstract class
    ├── base_readability.py       # Shared base for formula-based readability
    │
    │   ── Readability ──────────────────────────────────────────────────────
    ├── flesch_ease.py            # Flesch Reading Ease
    ├── flesch_kincaid.py         # Flesch-Kincaid Grade Level
    ├── gunning_fog.py            # Gunning Fog Index
    ├── smog.py                   # SMOG Grade
    ├── coleman_liau.py           # Coleman-Liau Index
    ├── ari.py                    # Automated Readability Index
    ├── linsear_write.py          # Linsear Write Formula
    ├── dale_chall.py             # Dale-Chall Score
    ├── spache.py                 # Spache Readability Formula
    │
    │   ── Linguistic Complexity ────────────────────────────────────────────
    ├── parse_tree_depth.py       # Constituency parse tree depth
    ├── dependency_distance.py    # Mean dependency distance
    ├── lexical_density.py        # Content word ratio
    ├── zipf_frequency.py         # Mean Zipf word frequency
    ├── constituent_count.py      # Number of syntactic constituents
    ├── conjunction_rate.py       # Conjunction density
    ├── passive_voice_rate.py     # Passive voice ratio
    ├── quantifier_rate.py        # Quantifier density
    ├── stem_complexity.py        # MCQ stem complexity
    │
    │   ── Ambiguity ──────────────────────────────────────────────────────
    ├── polysemy.py               # Mean polysemy score (WordNet senses)
    ├── pronoun_ratio.py          # Unresolved pronoun ratio
    ├── negation_ambiguity.py     # Scope-ambiguous negation detection
    │
    │   ── Well-Formedness ────────────────────────────────────────────────
    ├── has_verb.py               # Question contains a verb
    ├── question_mark.py          # Question ends with a question mark
    ├── negation_rate.py          # Negation word frequency
    ├── wh_word_type.py           # WH-word type classification
    ├── lm_perplexity.py          # GPT-2 perplexity (grammaticality proxy)
    ├── subject_verb_agreement.py # Subject-verb agreement check
    ├── multipart_question.py     # Detects multi-part / compound questions
    ├── question_type_consistency.py  # MCQ stem–option consistency
    │
    │   ── Diversity ────────────────────────────────────────────────────────
    ├── self_bleu.py              # Self-BLEU (set-level n-gram overlap)
    ├── distinct_n.py             # Distinct-N n-gram diversity
    ├── vocabulary_novelty.py     # Vocabulary novelty vs. batch
    ├── openendedness.py          # Open-ended vs. closed question ratio
    │
    │   ── Pedagogical ─────────────────────────────────────────────────────
    ├── bloom_level.py            # Bloom's Taxonomy level (keyword matching)
    ├── dok_level.py              # Webb's Depth of Knowledge level
    ├── hots_keywords.py          # HOTS keyword presence score
    ├── sqp_score.py              # Survey Quality Predictor (Saris & Gallhofer)
    │
    │   ── Fairness & Ethics ───────────────────────────────────────────────
    ├── pii_risk.py               # PII entity detection (spaCy NER)
    ├── k_anonymity_risk.py       # k-anonymity risk proxy
    ├── dp_epsilon_risk.py        # Differential Privacy ε risk proxy
    │
    │   ── Reference / Retrieval ──────────────────────────────────────────
    ├── bm25.py                   # BM25 retrieval relevance
    ├── tfidf.py                  # TF-IDF cosine similarity
    ├── sentencebert.py           # Sentence-BERT semantic similarity
    ├── bertscore.py              # BERTScore (P, R, F1)
    ├── bleu.py                   # BLEU score
    ├── rouge.py                  # ROUGE (1, 2, L)
    ├── chrf.py                   # chrF / chrF++
    ├── meteor.py                 # METEOR
    ├── comet.py                  # COMET (neural MT metric)
    ├── bartscore.py              # BARTScore
    ├── bleurt.py                 # BLEURT
    └── nli.py                    # NLI entailment score
```

## Data models (`models.py`)

All inputs and outputs share a common type vocabulary:

| Type | Role |
|---|---|
| `Question` | A single assessment item (text, options, metadata) |
| `EvaluationContext` | Curriculum context — learning objectives, course content, rubric |
| `MetricResult` | Output of one atomic metric: `score` (0–1), `rationale`, `flagged` |
| `DimensionResult` | Maps metric name → `MetricResult` for one dimension |
| `AspectResult` | Maps dimension name → `DimensionResult` for one aspect |
| `EvaluationResult` | Maps aspect name → `AspectResult`; top-level output per question |

`EvaluationResult.flat_scores()` returns a fully-nested dict `{aspect: {dimension: {metric: score}}}` for quick inspection.

## Evaluation flow

```
Question + EvaluationContext
        │
        ▼
    Evaluator.evaluate()
        │
        ├── PedagogicalAspect.evaluate()
        │       └── CurriculumAlignmentDimension.evaluate()
        │               └── BM25Metric.compute()  → MetricResult
        │               └── SentenceBERTMetric.compute() → MetricResult
        │               → DimensionResult
        │       → AspectResult
        │
        ├── PsychometricAspect.evaluate()  …
        ├── LinguisticStructuralAspect.evaluate()  …
        └── FairnessEthicsAspect.evaluate()  …
        │
        ▼
    EvaluationResult
        │
        ▼  (optional post-processing)
    Scorer.aggregate_evaluation()  →  float
```

## Scorer strategies

`Scorer` never runs automatically. Call it explicitly when you need a rolled-up number:

| Strategy | When to use |
|---|---|
| `weighted_average` | Default — uniform or custom-weighted mean |
| `min_gate` | Safety/risk dimensions where the worst metric dominates |
| `percentile_rank` | Distributional analysis across large question sets |

## Pipeline (`pipeline.py`)

`Pipeline.run(questions, context)` evaluates a list of questions with optional:
- **Parallelism** (`max_workers`) — thread-pool over questions
- **Caching** — SHA-256 keyed cache in `data/results/.cache/`; skips re-evaluation of unchanged questions
- **Circuit breaker** — aborts after `max_consecutive_failures` to surface systemic errors early

## Adding a new metric

1. Create `src/metrics/my_metric.py` subclassing `BaseMetric`; implement `compute(question, context) → MetricResult`.
2. Register it in the relevant dimension (e.g., `src/dimensions/linguistic_structural/readability.py`).
3. Add a test in `tests/metrics/test_my_metric.py`.
