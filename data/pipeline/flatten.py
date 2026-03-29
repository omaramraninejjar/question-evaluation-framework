"""
data/pipeline/flatten.py
========================
Stage 4 of the dataset construction pipeline: flatten.

Concatenates all aligned/{course_id}/records.jsonl files into a single
data/dataset.jsonl file, and prepends a metadata header record.

Usage
-----
    python -m data.pipeline.flatten
    python -m data.pipeline.flatten --aligned-dir data/aligned --out data/dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

ALIGNED_DIR   = Path("data/aligned")
DATASET_OUT   = Path("data/dataset.jsonl")


def flatten(
    aligned_dir: str | Path = ALIGNED_DIR,
    out_path: str | Path = DATASET_OUT,
) -> int:
    """
    Merge all records.jsonl files under aligned_dir into out_path.
    Returns total number of records written.
    """
    aligned_dir = Path(aligned_dir)
    out_path = Path(out_path)

    record_files = sorted(aligned_dir.glob("*/records.jsonl"))
    if not record_files:
        logger.error("No records.jsonl files found under %s", aligned_dir)
        return 0

    records: list[dict] = []
    course_counts: dict[str, int] = {}

    for records_path in record_files:
        course_id = records_path.parent.name
        course_records = _read_jsonl(records_path)
        course_counts[course_id] = len(course_records)
        records.extend(course_records)
        logger.info("  %s: %d records", course_id, len(course_records))

    # Header record (not an evaluation record — used for dataset-level metadata)
    header = {
        "_type": "dataset_header",
        "date": str(date.today()),
        "total_records": len(records),
        "courses": course_counts,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("dataset.jsonl written to %s (%d records across %d courses)",
                out_path, len(records), len(course_counts))
    return len(records)


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed line in %s: %s", path, exc)
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Flatten all aligned course records into a single dataset.jsonl.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--aligned-dir", default=str(ALIGNED_DIR))
    parser.add_argument("--out", default=str(DATASET_OUT), dest="out_path")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    flatten(aligned_dir=args.aligned_dir, out_path=args.out_path)
