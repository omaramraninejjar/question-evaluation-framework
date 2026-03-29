"""
Reporter — renders EvaluationResult objects into JSON and Markdown.

Both renderers walk the nested dict structure:
    EvaluationResult.scores
        -> {aspect_name: AspectResult}
              -> .scores {dimension_name: DimensionResult}
                    -> .scores {metric_name: MetricResult}
"""

from __future__ import annotations
import json
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime, timezone

from src.models import EvaluationResult, AspectResult, DimensionResult, MetricResult


# ---------------------------------------------------------------------------
# Base renderer
# ---------------------------------------------------------------------------

class ReportRenderer(ABC):

    @abstractmethod
    def render(
        self,
        results: list[EvaluationResult],
        include_rationale: bool = True,
        include_metadata: bool = False,
        score_precision: int = 4,
    ) -> str: ...


# ---------------------------------------------------------------------------
# JSON renderer
# ---------------------------------------------------------------------------

class JsonRenderer(ReportRenderer):

    def render(
        self,
        results: list[EvaluationResult],
        include_rationale: bool = True,
        include_metadata: bool = False,
        score_precision: int = 4,
    ) -> str:

        def _metric(name: str, r: MetricResult) -> dict:
            d: dict = {
                "metric": name,
                "score": round(r.score, score_precision),
                "flagged": r.flagged,
            }
            if include_rationale:
                d["rationale"] = r.rationale
            if include_metadata:
                d["metadata"] = r.metadata
            return d

        def _dimension(name: str, r: DimensionResult) -> dict:
            return {
                "dimension": name,
                "notes": r.notes,
                "metrics": [
                    _metric(m_name, m_result)
                    for m_name, m_result in r.scores.items()
                ],
            }

        def _aspect(name: str, r: AspectResult) -> dict:
            return {
                "aspect": name,
                "dimensions": [
                    _dimension(d_name, d_result)
                    for d_name, d_result in r.scores.items()
                ],
            }

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_questions": len(results),
            "evaluations": [
                {
                    "question_id": ev.question_id,
                    "aspects": [
                        _aspect(a_name, a_result)
                        for a_name, a_result in ev.scores.items()
                    ],
                }
                for ev in results
            ],
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

class MarkdownRenderer(ReportRenderer):

    def render(
        self,
        results: list[EvaluationResult],
        include_rationale: bool = True,
        include_metadata: bool = False,
        score_precision: int = 4,
    ) -> str:
        lines: list[str] = []
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines += [
            "# Evaluation Report",
            f"_Generated: {ts} — {len(results)} question(s)_",
            "",
        ]

        for ev in results:
            lines += [f"## Question `{ev.question_id}`", ""]

            for asp_name, asp_result in ev.scores.items():
                lines += [
                    f"### {asp_name.replace('_', ' ').title()}",
                    "",
                ]

                for dim_name, dim_result in asp_result.scores.items():
                    flagged = dim_result.flagged_metrics()
                    flag_str = " ⚠️" if flagged else ""
                    lines += [
                        f"#### {dim_name.replace('_', ' ').title()}{flag_str}",
                    ]
                    if dim_result.notes:
                        lines.append(f"> {dim_result.notes}")

                    if dim_result.scores:
                        header = "| Metric | Score | Flagged |"
                        sep    = "|--------|-------|---------|"
                        if include_rationale:
                            header += " Rationale |"
                            sep    += "-----------|"
                        lines += ["", header, sep]

                        for m_name, m_result in dim_result.scores.items():
                            row = (
                                f"| {m_name} "
                                f"| {m_result.score:.{score_precision}f} "
                                f"| {'yes' if m_result.flagged else 'no'} |"
                            )
                            if include_rationale:
                                row += f" {m_result.rationale} |"
                            lines.append(row)

                    lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reporter facade
# ---------------------------------------------------------------------------

RENDERERS: dict[str, ReportRenderer] = {
    "json": JsonRenderer(),
    "markdown": MarkdownRenderer(),
}


class Reporter:
    """
    Writes evaluation results to disk in one or more formats.

    Usage:
        reporter = Reporter(output_dir="data/results", formats=["json", "markdown"])
        reporter.write(results, filename_stem="batch_01")
    """

    def __init__(
        self,
        output_dir: str | Path = "data/results",
        formats: list[str] | None = None,
        include_rationale: bool = True,
        include_metadata: bool = False,
        score_precision: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.formats = formats or ["json", "markdown"]
        self.include_rationale = include_rationale
        self.include_metadata = include_metadata
        self.score_precision = score_precision

        unknown = set(self.formats) - set(RENDERERS)
        if unknown:
            raise ValueError(
                f"Unknown report format(s): {unknown}. Available: {list(RENDERERS)}"
            )

    def write(
        self,
        results: list[EvaluationResult],
        filename_stem: str = "report",
    ) -> dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        ext_map = {"json": ".json", "markdown": ".md"}

        for fmt in self.formats:
            content = RENDERERS[fmt].render(
                results,
                include_rationale=self.include_rationale,
                include_metadata=self.include_metadata,
                score_precision=self.score_precision,
            )
            path = self.output_dir / f"{filename_stem}{ext_map.get(fmt, f'.{fmt}')}"
            path.write_text(content, encoding="utf-8")
            written[fmt] = path

        return written

    def render_string(
        self,
        results: list[EvaluationResult],
        fmt: str = "json",
    ) -> str:
        return RENDERERS[fmt].render(
            results,
            include_rationale=self.include_rationale,
            include_metadata=self.include_metadata,
            score_precision=self.score_precision,
        )