# Question Quality Evaluation Framework

A modular, multi-aspect framework for automatically evaluating the quality of assessment questions. Designed for AI-generated and human-authored questions in educational settings.

## What it does

Given a question and its curriculum context, the framework computes scores across four quality aspects:

| Aspect | What it measures |
|---|---|
| **Pedagogical** | Educational alignment — does the question target the right learning objectives and cognitive level? |
| **Psychometric** | Measurement quality — is the question well-calibrated for difficulty, discrimination, and distractor functioning? |
| **Linguistic & Structural** | Clarity and form — is the question readable, unambiguous, and well-formed? |
| **Fairness & Ethics** | Equity and safety — does the question contain bias, harmful content, or privacy risk? |

Each aspect breaks into **dimensions**, each measured by one or more atomic **metrics**. The framework returns raw scores at every level with no forced aggregation — you decide how to roll them up. All evaluation is **response-free** — no student response data is needed.

### Dimensions

**Pedagogical** — educational alignment and cognitive appropriateness
`curriculum_alignment` · `concept_coverage` · `cognitive_demand` · `response_burden`

**Psychometric** — item quality from a measurement theory perspective
`difficulty` · `discrimination` · `guessing_careless` · `distractor_functioning` · `item_fit` · `dimensionality` · `reliability`

**Linguistic & Structural** — clarity, form, and variety
`readability` · `linguistic_complexity` · `ambiguity` · `well_formedness` · `diversity`

**Fairness & Ethics** — bias, sensitivity, and privacy
`group_bias` · `measurement_invariance` · `content_sensitivity` · `harmful_content_risk` · `privacy_risk`

## Architecture

```
Evaluator
└── Aspect  (pedagogical | psychometric | linguistic_structural | fairness_ethics)
    └── Dimension  (e.g. readability, cognitive_demand, group_bias …)
        └── Metric  (e.g. FleschKincaid, BloomLevel, PIIRisk …)
```

All layers communicate through shared data models defined in [src/models.py](src/models.py). Aggregation is handled separately by `Scorer`, which supports weighted average, min-gate, and percentile-rank strategies.

## Quick start

```python
from src.models import Question, EvaluationContext
from src.evaluator import Evaluator

question = Question(
    id="q1",
    text="What is the difference between supervised and unsupervised learning?",
    subject="Machine Learning",
    grade_level="undergraduate",
)

context = EvaluationContext(
    learning_objectives=["Distinguish between supervised and unsupervised learning paradigms"],
    course_content="Supervised learning uses labelled data to train a model ...",
)

result = Evaluator().evaluate(question, context)
print(result.flat_scores())
```

For batch evaluation:

```python
from src.pipeline import Pipeline

report = Pipeline().run(questions, context, batch_name="pilot_study")
print(report)  # PipelineReport(total=50, succeeded=50, failed=0, cached=0, success_rate=100.0%)
```

## Project structure

```
framework/
├── config/             # Aspect/dimension taxonomy and settings
├── data/               # Datasets, raw sources, and results (see data/README.md)
├── notebooks/          # Exploratory analysis notebooks
├── scripts/            # Standalone utility scripts
├── src/                # Framework source code (see src/README.md)
├── tests/              # Unit and integration tests
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> Some metrics download large models on first use (GPT-2 ~500 MB, BERTScore ~400 MB, COMET ~1.9 GB). These are cached in `~/.cache/huggingface`.

## Dataset

Questions and course content are sourced from **MIT OpenCourseWare** (OCW). Each course provides:
- **`content/`** — textbook chapters or detailed prose lecture notes (primary `course_content` source)
- **`assignments/`** — problem sets (`source_type: assignment`)
- **`exams/`** — exam papers (`source_type: exam`)

Questions have no pre-existing mapping to course passages. The dataset pipeline creates this mapping via BM25 retrieval: each question is matched to its most semantically relevant content chunk, which becomes `EvaluationContext.course_content`. Each course is also tagged with `field` (e.g., `mathematics`) and `level` (`undergraduate` / `graduate`) for cross-domain analysis.

> **Raw PDFs are not included in this repository.** Download course materials from
> MIT OpenCourseWare (`https://ocw.mit.edu`) using the `course_id` listed in each
> course's `data/raw/{course_id}/course_meta.json`, then place them in the
> corresponding `data/raw/{course_id}/` directory before running the pipeline.

```bash
python -m data.pipeline.extract --course-id 6.042j   # extract + chunk
python -m data.pipeline.align   --course-id 6.042j   # BM25 alignment
python -m data.pipeline.flatten                       # merge → dataset.jsonl
```

See [data/README.md](data/README.md) for the full pipeline description and record schema.

## Configuration

`config/aspects.yaml` defines the full taxonomy — aspects, dimensions, and `critical` flags. No weights or aggregation policy are stored there; those are caller-supplied at analysis time via `Scorer`.

## Testing

```bash
pytest tests/
pytest tests/dimensions/test_readability.py -v
```
