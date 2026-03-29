"""
scripts/run_all.py
==================
End-to-end pipeline orchestrator.

For every course found in data/raw/ that has a course_meta.json:

  Step 1  EXTRACT   — if data/processed/{course_id}/chunks.jsonl is missing
  Step 2  ALIGN     — if data/aligned/{course_id}/records.jsonl  is missing
  Step 3  FLATTEN   — always re-run (fast; keeps dataset.jsonl in sync)
  Step 4  EVALUATE  — if data/results/{course_id}_results.jsonl  is missing
                      saves results in rich format (score + flagged + rationale
                      + course_content + topic)

Usage
-----
    # From the repo root:
    python scripts/run_all.py

    # Dry-run: show what would be done, skip nothing
    python scripts/run_all.py --dry-run

    # Force re-evaluation for a specific course even if results exist
    python scripts/run_all.py --force-eval 6.042J

    # Skip evaluation entirely (pipeline only)
    python scripts/run_all.py --no-eval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ── Repo root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ALIGNED_DIR   = ROOT / "data" / "aligned"
DATASET_PATH  = ROOT / "data" / "dataset.jsonl"
RESULTS_DIR   = ROOT / "data" / "results"


# ══════════════════════════════════════════════════════════════════════════
# Discovery
# ══════════════════════════════════════════════════════════════════════════

def discover_courses() -> list[str]:
    """Return sorted list of course IDs that have a course_meta.json."""
    return sorted(
        p.parent.name
        for p in RAW_DIR.glob("*/course_meta.json")
    )


# ══════════════════════════════════════════════════════════════════════════
# Pipeline steps (wrappers around existing modules)
# ══════════════════════════════════════════════════════════════════════════

def run_extract(course_id: str) -> bool:
    """Run extraction for one course. Returns True on success."""
    from data.pipeline.extract import CourseExtractor
    course_dir = RAW_DIR / course_id
    out_dir    = PROCESSED_DIR / course_id
    try:
        extractor = CourseExtractor(course_dir=course_dir, out_dir=out_dir)
        extractor.run()
        return True
    except Exception as exc:
        logger.error("[%s] Extract failed: %s", course_id, exc)
        return False


def run_align(course_id: str) -> bool:
    """Run BM25 alignment for one course. Returns True on success."""
    from data.pipeline.align import CourseAligner
    try:
        aligner = CourseAligner(
            course_id=course_id,
            processed_dir=PROCESSED_DIR,
            aligned_dir=ALIGNED_DIR,
        )
        aligner.run()
        return True
    except Exception as exc:
        logger.error("[%s] Align failed: %s", course_id, exc)
        return False


def run_flatten() -> bool:
    """Merge all aligned records into dataset.jsonl. Returns True on success."""
    from data.pipeline.flatten import flatten
    try:
        flatten(aligned_dir=ALIGNED_DIR, out_path=DATASET_PATH)
        return True
    except Exception as exc:
        logger.error("Flatten failed: %s", exc)
        return False


# ══════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════

def run_evaluate(course_id: str) -> bool:
    """
    Evaluate all questions for one course and save to data/results/{course_id}_results.jsonl.
    Returns True on success.
    """
    from data.pipeline.loader import load_dataset
    from src.evaluator import Evaluator

    results_path = RESULTS_DIR / f"{course_id}_results.jsonl"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("[%s] Loading dataset …", course_id)
    try:
        pairs = load_dataset(
            dataset_path=DATASET_PATH,
            data_dir=ROOT / "data",
            course_filter=[course_id],
        )
    except Exception as exc:
        logger.error("[%s] Failed to load dataset: %s", course_id, exc)
        return False

    if not pairs:
        logger.warning("[%s] No pairs found in dataset — skipping evaluation.", course_id)
        return False

    logger.info("[%s] Evaluating %d questions …", course_id, len(pairs))
    evaluator = Evaluator()
    rows: list[dict] = []
    errors = 0

    t0 = time.time()
    for i, (q, ctx) in enumerate(pairs, 1):
        try:
            result = evaluator.evaluate(q, ctx)
        except Exception as exc:
            logger.warning("[%s] Q %s failed: %s", course_id, q.id, exc)
            errors += 1
            continue

        # Rich format: score + flagged + rationale per metric
        full_scores = {
            asp: {
                dim: {
                    m: {
                        "score": mr.score,
                        "flagged": mr.flagged,
                        "rationale": mr.rationale,
                    }
                    for m, mr in dim_res.scores.items()
                }
                for dim, dim_res in asp_res.scores.items()
            }
            for asp, asp_res in result.scores.items()
        }

        rows.append({
            "question_id":  q.id,
            "source_type":  q.metadata.get("source_type"),
            "source_label": q.metadata.get("source_label"),
            "course_id":    q.metadata.get("course_id"),
            "bm25_score":   ctx.metadata.get("bm25_score"),
            "topic":        ctx.metadata.get("topic"),
            "course_content": ctx.course_content,
            "question_text": q.text,
            "scores": full_scores,
        })

        # Progress log every 10 questions
        if i % 10 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            remaining = (len(pairs) - i) / rate if rate > 0 else 0
            logger.info(
                "[%s] %d/%d done  (%.1f q/min, ~%.0f min remaining)",
                course_id, i, len(pairs), rate * 60, remaining / 60,
            )

    elapsed_min = (time.time() - t0) / 60
    logger.info(
        "[%s] Evaluation complete: %d saved, %d errors  (%.1f min total)",
        course_id, len(rows), errors, elapsed_min,
    )

    if not rows:
        logger.error("[%s] No results to save.", course_id)
        return False

    with results_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("[%s] Results saved → %s", course_id, results_path)
    return True


# ══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════

def run_all(
    dry_run: bool = False,
    no_eval: bool = False,
    force_eval: list[str] | None = None,
) -> None:
    force_eval = set(force_eval or [])
    courses = discover_courses()

    if not courses:
        logger.error("No courses found in %s", RAW_DIR)
        return

    logger.info("Discovered %d course(s): %s", len(courses), ", ".join(courses))

    # ── Steps 1 & 2: extract + align per course ───────────────────────────
    needs_flatten = False

    for course_id in courses:
        chunks_path  = PROCESSED_DIR / course_id / "chunks.jsonl"
        records_path = ALIGNED_DIR   / course_id / "records.jsonl"

        needs_extract = not chunks_path.exists()
        needs_align   = not records_path.exists()

        if not needs_extract and not needs_align:
            logger.info("[%s] Extract + align already done — skipping.", course_id)
            continue

        if dry_run:
            if needs_extract:
                logger.info("[DRY-RUN] [%s] Would run: extract", course_id)
            if needs_align:
                logger.info("[DRY-RUN] [%s] Would run: align", course_id)
            continue

        if needs_extract:
            logger.info("[%s] Running extract …", course_id)
            ok = run_extract(course_id)
            if not ok:
                logger.error("[%s] Extract failed — skipping align.", course_id)
                continue

        if needs_align or not records_path.exists():
            logger.info("[%s] Running align …", course_id)
            ok = run_align(course_id)
            if not ok:
                continue

        needs_flatten = True

    # ── Step 3: flatten (always, to keep dataset.jsonl in sync) ───────────
    if dry_run:
        logger.info("[DRY-RUN] Would run: flatten")
    else:
        logger.info("Running flatten …")
        run_flatten()

    # ── Step 4: evaluate per course ───────────────────────────────────────
    if no_eval:
        logger.info("--no-eval set — skipping evaluation.")
        return

    for course_id in courses:
        results_path = RESULTS_DIR / f"{course_id}_results.jsonl"

        if results_path.exists() and course_id not in force_eval:
            logger.info(
                "[%s] Results already exist (%d bytes) — skipping evaluation.",
                course_id, results_path.stat().st_size,
            )
            continue

        if dry_run:
            action = "re-evaluate (--force-eval)" if course_id in force_eval else "evaluate"
            logger.info("[DRY-RUN] [%s] Would run: %s", course_id, action)
            continue

        if course_id in force_eval:
            logger.info("[%s] --force-eval set — re-running evaluation.", course_id)

        run_evaluate(course_id)

    logger.info("All done.")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the full QG pipeline for all courses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without executing anything.",
    )
    p.add_argument(
        "--no-eval", action="store_true",
        help="Run pipeline steps only; skip metric evaluation.",
    )
    p.add_argument(
        "--force-eval", nargs="+", metavar="COURSE_ID", default=[],
        help="Re-run evaluation for these course IDs even if results already exist.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_all(
        dry_run=args.dry_run,
        no_eval=args.no_eval,
        force_eval=args.force_eval,
    )
