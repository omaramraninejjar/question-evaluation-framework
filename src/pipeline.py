"""
Pipeline — batch evaluation runner.
"""

from __future__ import annotations
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from src.models import Question, EvaluationContext, EvaluationResult
from src.evaluator import Evaluator
from src.reporter import Reporter

logger = logging.getLogger(__name__)


@dataclass
class PipelineReport:
    total: int
    succeeded: int
    failed: int
    skipped_cached: int
    results: list[EvaluationResult] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.total if self.total else 0.0

    def __repr__(self) -> str:
        return (
            f"PipelineReport(total={self.total}, succeeded={self.succeeded}, "
            f"failed={self.failed}, cached={self.skipped_cached}, "
            f"success_rate={self.success_rate:.1%})"
        )


class EvalCache:
    VERSION = "1"

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, question: Question) -> str:
        content = json.dumps(
            {"id": question.id, "text": question.text, "v": self.VERSION},
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, question: Question) -> EvaluationResult | None:
        path = self.cache_dir / f"{self._key(question)}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return EvaluationResult(
                    question_id=data["question_id"],
                    metadata=data.get("metadata", {}),
                )
            except Exception as exc:
                logger.warning("Cache read failed for %s: %s", question.id, exc)
        return None

    def set(self, question: Question, result: EvaluationResult) -> None:
        path = self.cache_dir / f"{self._key(question)}.json"
        try:
            path.write_text(
                json.dumps({"question_id": result.question_id, "metadata": result.metadata}, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Cache write failed for %s: %s", question.id, exc)


class Pipeline:
    """
    Orchestrates batch evaluation of a list of Questions.

    Usage:
        pipeline = Pipeline()
        report   = pipeline.run(questions, context, batch_name="pilot")
    """

    def __init__(
        self,
        evaluator: Evaluator | None = None,
        reporter: Reporter | None = None,
        max_workers: int = 4,
        cache_enabled: bool = True,
        cache_dir: str | Path = "data/results/.cache",
        max_consecutive_failures: int = 10,
    ):
        self.evaluator = evaluator or Evaluator()
        self.reporter  = reporter or Reporter()
        self.max_workers = max_workers
        self.cache_enabled = cache_enabled
        self.cache = EvalCache(Path(cache_dir)) if cache_enabled else None
        self.max_consecutive_failures = max_consecutive_failures

    def run(
        self,
        questions: list[Question],
        context: EvaluationContext,
        batch_name: str = "batch",
        write_report: bool = True,
    ) -> PipelineReport:
        results: list[EvaluationResult] = []
        errors: dict[str, str] = {}
        skipped = 0
        consecutive_failures = 0
        futures_map: dict = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for q in questions:
                if self.cache:
                    cached = self.cache.get(q)
                    if cached:
                        results.append(cached)
                        skipped += 1
                        continue
                futures_map[pool.submit(self._evaluate_one, q, context)] = q

            for future in as_completed(futures_map):
                q = futures_map[future]
                try:
                    result = future.result()
                    results.append(result)
                    if self.cache:
                        self.cache.set(q, result)
                    consecutive_failures = 0
                except Exception as exc:
                    errors[q.id] = str(exc)
                    consecutive_failures += 1
                    logger.error("Failed: %s — %s", q.id, exc)
                    if consecutive_failures >= self.max_consecutive_failures:
                        pool.shutdown(wait=False, cancel_futures=True)
                        break

        report = PipelineReport(
            total=len(questions),
            succeeded=len(results),
            failed=len(errors),
            skipped_cached=skipped,
            results=results,
            errors=errors,
        )

        if write_report and results:
            written = self.reporter.write(results, filename_stem=batch_name)
            for fmt, path in written.items():
                logger.info("Report written [%s]: %s", fmt, path)

        return report

    def _evaluate_one(self, question: Question, context: EvaluationContext) -> EvaluationResult:
        return self.evaluator.evaluate(question, context)