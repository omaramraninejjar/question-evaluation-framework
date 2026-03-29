"""
data/pipeline/align.py
======================
Stage 3 of the dataset construction pipeline: BM25 alignment.

For each question in processed/{course_id}/questions_raw.jsonl, retrieves the
most relevant chunk from processed/{course_id}/chunks.jsonl and writes a
(question, context) record to aligned/{course_id}/records.jsonl.

The mapping does not exist before this stage — this is where it is created.

Usage
-----
    python -m data.pipeline.align --course-id 6.042j

    # Or for all courses at once:
    python -m data.pipeline.align --all

Dependencies
------------
    pip install rank-bm25
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _BM25_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BM25_AVAILABLE = False
    logger.warning("rank-bm25 not installed. Alignment will fail.")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
ALIGNED_DIR   = Path("data/aligned")
LOW_SCORE_THRESHOLD: float = 5.0


# ---------------------------------------------------------------------------
# BM25 aligner
# ---------------------------------------------------------------------------

class CourseAligner:
    """
    Aligns questions to lecture chunks for one course using BM25Okapi.

    For each question:
      - Tokenises the question text.
      - Scores all chunks and selects the top-1.
      - Writes a record containing the full question dict and the matched
        chunk's text as context.course_content.
    """

    def __init__(
        self,
        course_id: str,
        processed_dir: str | Path = PROCESSED_DIR,
        aligned_dir: str | Path = ALIGNED_DIR,
        low_score_threshold: float = LOW_SCORE_THRESHOLD,
    ):
        self.course_id = course_id
        self.processed_dir = Path(processed_dir) / course_id
        self.aligned_dir = Path(aligned_dir) / course_id
        self.low_score_threshold = low_score_threshold

    def run(self) -> int:
        """Run alignment. Returns the number of records written."""
        if not _BM25_AVAILABLE:
            raise RuntimeError("rank-bm25 is required. Run: pip install rank-bm25")

        chunks = self._load_chunks()
        questions = self._load_questions()
        objectives = self._load_objectives()

        if not chunks:
            logger.error("No chunks found for course %s — skipping alignment.", self.course_id)
            return 0
        if not questions:
            logger.warning("No questions found for course %s.", self.course_id)
            return 0

        # Build BM25 index over chunk texts
        tokenised_corpus = [chunk["text"].lower().split() for chunk in chunks]
        bm25 = BM25Okapi(tokenised_corpus)

        self.aligned_dir.mkdir(parents=True, exist_ok=True)

        # Group questions by source file so we can populate question_batch
        batch_map: dict[str, list[str]] = {}
        for q in questions:
            batch_map.setdefault(q["source_file"], []).append(q["text"])

        records: list[dict] = []
        for q in questions:
            record = self._align_one(q, chunks, bm25, objectives, batch_map)
            records.append(record)

        out_path = self.aligned_dir / "records.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        low_conf = sum(1 for r in records if r["context"]["metadata"].get("low_confidence"))
        logger.info(
            "Aligned %d questions for %s → %s  (low-confidence: %d)",
            len(records), self.course_id, out_path, low_conf,
        )
        return len(records)

    # ── Core alignment ────────────────────────────────────────────────────

    def _align_one(
        self,
        question: dict,
        chunks: list[dict],
        bm25: "BM25Okapi",
        objectives: list[str],
        batch_map: dict[str, list[str]],
    ) -> dict:
        query_tokens = question["text"].lower().split()
        scores = bm25.get_scores(query_tokens)
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])
        best_chunk = chunks[best_idx]

        return {
            "question": {
                "id": question["q_id"],
                "text": question["text"],
                "options": question.get("options"),
                "correct_answer": None,
                "metadata": {
                    "course_id": self.course_id,
                    "source_file": question["source_file"],
                    "source_type": question.get("source_type"),
                    "source_label": question.get("source_label"),
                },
            },
            "context": {
                "learning_objectives": objectives,
                "course_content": best_chunk["text"],
                "rubric": None,
                "metadata": {
                    "chunk_id": best_chunk["chunk_id"],
                    "source_file": best_chunk.get("source_file"),
                    "topic": best_chunk["topic"],
                    "bm25_score": round(best_score, 4),
                    "retrieval_rank": 1,
                    "low_confidence": best_score < self.low_score_threshold,
                    "question_batch": batch_map.get(question["source_file"], []),
                },
            },
        }

    # ── Loaders ───────────────────────────────────────────────────────────

    def _load_chunks(self) -> list[dict]:
        path = self.processed_dir / "chunks.jsonl"
        if not path.exists():
            logger.error("chunks.jsonl not found: %s", path)
            return []
        return self._read_jsonl(path)

    def _load_questions(self) -> list[dict]:
        path = self.processed_dir / "questions_raw.jsonl"
        if not path.exists():
            logger.error("questions_raw.jsonl not found: %s", path)
            return []
        return self._read_jsonl(path)

    def _load_objectives(self) -> list[str]:
        path = self.processed_dir / "objectives.txt"
        if not path.exists():
            return []
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        records = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping malformed JSONL line in %s: %s", path, exc)
        return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BM25 alignment: link each question to its best lecture chunk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--course-id", help="Single course to align (e.g. 6.042j)")
    group.add_argument("--all", action="store_true", help="Align all courses in data/processed/")
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--aligned-dir", default=str(ALIGNED_DIR))
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

    if args.all:
        processed_root = Path(args.processed_dir)
        course_ids = [p.name for p in processed_root.iterdir() if p.is_dir()]
    else:
        course_ids = [args.course_id]

    total = 0
    for cid in sorted(course_ids):
        aligner = CourseAligner(
            course_id=cid,
            processed_dir=args.processed_dir,
            aligned_dir=args.aligned_dir,
        )
        total += aligner.run()

    logger.info("Alignment complete. Total records written: %d", total)
