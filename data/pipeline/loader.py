"""
data/pipeline/loader.py
=======================
Loads dataset.jsonl into (Question, EvaluationContext) pairs ready for the evaluator.

Enriches each Question with subject/grade_level from the course's course_meta.json,
since those are course-level properties not stored per record.

Usage
-----
    from data.pipeline.loader import load_dataset

    pairs = load_dataset("data/dataset.jsonl")
    for question, context in pairs:
        result = evaluator.evaluate(question, context)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.models import EvaluationContext, Question

logger = logging.getLogger(__name__)

_meta_cache: dict[str, dict] = {}


def _load_course_meta(course_id: str, data_dir: Path) -> dict:
    if course_id not in _meta_cache:
        meta_path = data_dir / "raw" / course_id / "course_meta.json"
        if meta_path.exists():
            with meta_path.open(encoding="utf-8") as f:
                _meta_cache[course_id] = json.load(f)
        else:
            logger.warning("course_meta.json not found for %s", course_id)
            _meta_cache[course_id] = {}
    return _meta_cache[course_id]


def load_dataset(
    dataset_path: str | Path = "data/dataset.jsonl",
    data_dir: str | Path = "data",
    course_filter: list[str] | None = None,
    source_type_filter: list[str] | None = None,
    limit: int | None = None,
) -> list[tuple[Question, EvaluationContext]]:
    """
    Load (Question, EvaluationContext) pairs from dataset.jsonl.

    Args:
        dataset_path:        Path to dataset.jsonl.
        data_dir:            Root data directory (used to locate course_meta.json).
        course_filter:       If set, only load records from these course IDs.
        source_type_filter:  If set, only load "assignment" or "exam" records.
        limit:               Cap the number of records returned.

    Returns:
        List of (Question, EvaluationContext) tuples.
    """
    dataset_path = Path(dataset_path)
    data_dir = Path(data_dir)
    pairs: list[tuple[Question, EvaluationContext]] = []

    with dataset_path.open(encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)

            # Skip the dataset header record
            if record.get("_type") == "dataset_header":
                continue

            q_data = record["question"]
            ctx_data = record["context"]
            meta = q_data.get("metadata", {})
            course_id = meta.get("course_id", "")

            # Optional filters
            if course_filter and course_id not in course_filter:
                continue
            if source_type_filter and meta.get("source_type") not in source_type_filter:
                continue

            # Enrich with course-level metadata
            course_meta = _load_course_meta(course_id, data_dir)

            question = Question(
                id=q_data["id"],
                text=q_data["text"],
                options=q_data.get("options"),
                correct_answer=q_data.get("correct_answer"),
                subject=course_meta.get("subject"),
                grade_level=course_meta.get("level"),
                metadata={
                    **meta,
                    "field": course_meta.get("field"),
                },
            )

            context = EvaluationContext(
                learning_objectives=ctx_data.get("learning_objectives", []),
                course_content=ctx_data.get("course_content"),
                rubric=ctx_data.get("rubric"),
                metadata=ctx_data.get("metadata", {}),
            )

            pairs.append((question, context))

            if limit and len(pairs) >= limit:
                break

    logger.info("Loaded %d records from %s", len(pairs), dataset_path)
    return pairs
