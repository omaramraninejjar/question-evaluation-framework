# Data Directory

This directory holds all data artifacts produced and consumed by the framework, from raw source PDFs through to the final evaluation-ready dataset.

## Directory layout

```
data/
├── raw/                          # Source PDFs, unmodified
│   └── {course_id}/
│       ├── content/              # Textbook chapters or lecture notes (course_content source)
│       ├── assignments/          # Problem set / assignment PDFs
│       ├── exams/                # Exam PDFs
│       ├── syllabus.pdf
│       └── course_meta.json      # Course-level metadata (field, level, title, objectives)
│
├── processed/                    # Per-course extracted artefacts
│   └── {course_id}/
│       ├── chunks.jsonl          # Content passages (retrieval candidates)
│       ├── questions_raw.jsonl   # Questions extracted from assignments + exams
│       └── objectives.txt        # Learning objectives parsed from syllabus
│
├── aligned/                      # Per-course question–context pairs
│   └── {course_id}/
│       └── records.jsonl         # Each question matched to its best chunk via BM25
│
├── results/                      # Evaluation outputs written by Reporter
│
├── course_manifest.yaml          # Registry of all curated courses
└── dataset.jsonl                 # Final flat dataset (merge of all aligned/)
```

> `.gitkeep` files preserve empty directories in version control. Source PDFs and generated files are excluded from git.

---

## Course metadata (`course_meta.json`)

Each course directory contains a `course_meta.json` file that is the authoritative source of course-level metadata. The extraction pipeline reads it and propagates `subject` and `grade_level` into every extracted question.

```json
{
  "course_id": "6.042j",
  "title": "Mathematics for Computer Science",
  "subject": "Mathematics for Computer Science",
  "field": "mathematics",
  "level": "undergraduate",
  "institution": "MIT",
  "source": "MIT OpenCourseWare",
  "learning_objectives": [
    "Understand formal mathematical reasoning and proofs",
    "Apply discrete mathematics concepts to computer science problems",
    "Analyze algorithms using graph theory and probability"
  ]
}
```

**`field`** — broad academic domain: `mathematics`, `physics`, `computer_science`, `engineering`, `biology`, etc.
**`level`** — `undergraduate` or `graduate`.

---

## Course manifest (`course_manifest.yaml`)

A central registry of all curated courses. Used to drive batch pipeline runs.

```yaml
courses:
  - course_id: 6.042j
    title: Mathematics for Computer Science
    field: mathematics
    level: undergraduate
    content_type: textbook       # textbook | lecture_notes | mixed
    notes: "Lehman, Leighton & Meyer textbook PDF in content/"
```

---

## Data sources

Questions and course content are sourced from **MIT OpenCourseWare (OCW)**.
Preferred courses have:
1. A **detailed content source** — a linked open textbook or prose lecture notes (not slides).
2. At least one **problem set** (assignments/) and ideally one **exam** (exams/).
3. A **syllabus** with explicit learning objectives.

When the course uses a textbook, the textbook PDF (or its chapters) goes into `content/`.
Brief lecture slides are not suitable as `course_content` — they produce low-signal BM25 retrieval.

---

## Pipeline stages

### Stage 1 — Raw download

> **Raw PDFs are not included in this repository.** They must be downloaded manually
> from MIT OpenCourseWare before running the pipeline. Each course directory contains
> a `course_meta.json` file with the `course_id` (e.g. `6.042J`) that you can use to
> locate the course at `https://ocw.mit.edu/courses/<course_id>/`. Download the
> content, assignment, and exam PDFs and place them in the directory structure shown
> above.

PDFs are downloaded from OCW and placed in `raw/{course_id}/` with the structure above.
`course_meta.json` is filled in manually or with the help of the OCW course page.

### Stage 2 — Extraction (`raw/ → processed/`)

Run: `python -m data.pipeline.extract --course-id 6.042j`

- **`content/`** PDFs → chunked into passages → `chunks.jsonl`
- **`assignments/`** PDFs → question extraction, `source_type: assignment` → `questions_raw.jsonl`
- **`exams/`** PDFs → question extraction, `source_type: exam` → appended to `questions_raw.jsonl`
- **`syllabus.pdf`** → learning objectives → `objectives.txt`

### Stage 3 — BM25 alignment (`processed/ → aligned/`)

Run: `python -m data.pipeline.align --course-id 6.042j`

Questions have no pre-existing mapping to course content. This step creates the mapping:

1. All chunks for the course are loaded and a BM25 index is built.
2. Each question is issued as a query; the top-1 chunk is selected.
3. A record is written pairing the question with its matched chunk.

Records with BM25 score below 5.0 are flagged (`low_confidence: true`) and retained for manual review.

### Stage 4 — Flatten (`aligned/ → dataset.jsonl`)

Run: `python -m data.pipeline.flatten`

All `aligned/*/records.jsonl` files are merged into `dataset.jsonl`. A `question_batch` field is added to each record's `context.metadata`: all questions from the same source file are grouped together, which is required by the `SelfBLEUMetric` (diversity dimension).

---

## Record schema

Each row in `dataset.jsonl` and `aligned/*/records.jsonl`:

```json
{
  "question": {
    "id": "6.042j-A02-q4",
    "text": "Prove that every tree with n vertices has n−1 edges.",
    "options": null,
    "correct_answer": null,
    "metadata": {
      "course_id": "6.042j",
      "source_file": "pset2.pdf",
      "source_type": "assignment",
      "source_label": "Problem Set 2"
    }
  },
  "context": {
    "learning_objectives": [
      "Understand properties of trees and graphs"
    ],
    "course_content": "A tree is a connected acyclic graph. Any tree with n vertices has exactly n−1 edges ...",
    "rubric": null,
    "metadata": {
      "chunk_id": "6.042j-content-03-p2",
      "source_file": "chapter03.pdf",
      "topic": "Graph Theory",
      "bm25_score": 14.72,
      "retrieval_rank": 1,
      "low_confidence": false,
      "question_batch": [
        "Prove that every tree with n vertices has n−1 edges.",
        "How many edges does a complete graph K_n have?"
      ]
    }
  }
}
```

**Key design notes:**
- `course_content` is populated by the BM25 alignment step — it does not exist in `questions_raw.jsonl`.
- `source_type` distinguishes `"assignment"` from `"exam"` questions — exam questions typically have different cognitive demand and structural properties.
- `subject`, `field`, and `level` are **course-level properties**, not stored per question. They are looked up from `course_meta.json` via `course_id` at evaluation time and injected into `Question.subject` / `Question.grade_level` by the dataset loader.
- `question_batch` lists all questions from the same source file, enabling set-level diversity metrics.

---

## Chunk schema (`processed/{course_id}/chunks.jsonl`)

```json
{
  "chunk_id": "6.042j-content-03-p2",
  "course_id": "6.042j",
  "source_file": "chapter03.pdf",
  "topic": "Graph Theory",
  "text": "A tree is a connected acyclic graph ...",
  "token_count": 312
}
```

## Question schema (`processed/{course_id}/questions_raw.jsonl`)

```json
{
  "q_id": "6.042j-A02-q4",
  "course_id": "6.042j",
  "text": "Prove that every tree with n vertices has n−1 edges.",
  "options": null,
  "source_file": "pset2.pdf",
  "source_type": "assignment",
  "source_label": "Problem Set 2"
}
```

---

## Running the pipeline

```bash
# Single course end-to-end
python -m data.pipeline.extract --course-id 6.042j
python -m data.pipeline.align   --course-id 6.042j

# All courses → final dataset
python -m data.pipeline.flatten
```
