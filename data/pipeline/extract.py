"""
data/pipeline/extract.py
========================
Stage 2 of the dataset construction pipeline: extract text from raw PDFs and
produce per-course artefacts in data/processed/{course_id}/.

Outputs
-------
chunks.jsonl        -- lecture passages (retrieval candidates for BM25 alignment)
questions_raw.jsonl -- questions extracted from assignment PDFs
objectives.txt      -- one learning objective per line (from syllabus)

Usage
-----
    # As a module (programmatic)
    from data.pipeline.extract import CourseExtractor
    extractor = CourseExtractor(course_dir="data/raw/6.042j", out_dir="data/processed/6.042j")
    extractor.run()

    # As a script
    python -m data.pipeline.extract --course-dir data/raw/6.042j --out-dir data/processed/6.042j

Dependencies
------------
    pip install pdfplumber
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency — pdfplumber
# ---------------------------------------------------------------------------

try:
    import pdfplumber  # type: ignore
    _PDFPLUMBER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not installed. PDF extraction will be skipped.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_CHUNK_TOKENS: int = 500      # split sections longer than this
CHUNK_OVERLAP_TOKENS: int = 50   # overlap between adjacent chunks when re-splitting
MIN_CHUNK_TOKENS: int = 20       # discard chunks shorter than this (page headers, etc.)
MIN_QUESTION_CHARS: int = 30     # ignore extracted "questions" shorter than this
BM25_LOW_SCORE_THRESHOLD: float = 5.0  # used downstream; exported for reference

# Regex patterns
_HEADING_RE = re.compile(r"\n(?=[A-Z][A-Za-z0-9 \-:]{3,60}\n)")
_PARAGRAPH_RE = re.compile(r"\n{2,}")
# MCQ option line: capital letter followed by . or ) — avoids matching subpart labels like (a)
_MCQ_OPTION_RE = re.compile(
    r"^\s*(?:[A-E][.\)]\s)",
    re.MULTILINE,
)
# Question stem patterns — tried in order, first match wins per block.
# Pattern 1: "Problem N." / "Exercise N." / "Question N." (OCW style)
_PROBLEM_LABEL_RE = re.compile(
    r"(?:^|\n)((?:Problem|Exercise|Question|Part)\s+\d+[a-z]?\.?\s*(?:\[\d+\s*(?:pts?|points?)\])?"
    r".+?(?=\n(?:Problem|Exercise|Question|Part)\s+\d|\Z))",
    re.IGNORECASE | re.DOTALL,
)
# Pattern 2: Numbered stem ending with ? or action verb (fallback)
_NUMBERED_STEM_RE = re.compile(
    r"(?:^|\n)\s*(\d+[.\)]\s{1,3}(?:[A-Z].{15,}?)(?:\?|(?:explain|describe|prove|show|find|"
    r"calculate|derive|state|define|give|list|compare|discuss|evaluate)[^.\n]*\.))",
    re.IGNORECASE | re.DOTALL,
)

# ---------------------------------------------------------------------------
# Data classes (mirror the JSONL schemas documented in data/README.md)
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    course_id: str
    source_file: str       # which PDF this chunk came from
    topic: str
    text: str
    token_count: int = 0

    def __post_init__(self) -> None:
        if not self.token_count:
            self.token_count = len(self.text.split())


@dataclass
class RawQuestion:
    q_id: str
    course_id: str
    text: str
    options: list[str] | None
    source_file: str
    source_type: str       # assignment | exam
    source_label: str      # human-readable label (e.g. "Problem Set 2")


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _approx_tokens(text: str) -> int:
    """Approximate token count by whitespace splitting."""
    return len(text.split())


def _chunk_by_heading(text: str) -> list[str]:
    """
    Split text on section headings (title-cased lines).
    Returns a list of section strings.
    """
    parts = _HEADING_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_long_section(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Split a text that exceeds max_tokens at paragraph boundaries.
    Adjacent chunks share overlap_tokens of trailing/leading words.
    """
    paragraphs = [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]
    chunks: list[str] = []
    current_words: list[str] = []

    for para in paragraphs:
        para_words = para.split()
        if current_words and _approx_tokens(" ".join(current_words)) + len(para_words) > max_tokens:
            chunks.append(" ".join(current_words))
            current_words = current_words[-overlap_tokens:] + para_words
        else:
            current_words.extend(para_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


_CID_RE = re.compile(r"\(cid:\d+\)")          # e.g. (cid:127)
_LIGATURE_MAP = str.maketrans({              # common PDF ligature artifacts
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl",
})


def _clean_pdf_text(text: str) -> str:
    """Remove CID glyph placeholders and normalise ligature artifacts."""
    text = _CID_RE.sub("", text)
    text = text.translate(_LIGATURE_MAP)
    # Collapse runs of whitespace introduced by removals
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def _extract_text_from_pdf(path: Path) -> str:
    """Extract all text from a PDF using pdfplumber, with artifact cleanup."""
    if not _PDFPLUMBER_AVAILABLE:
        raise RuntimeError("pdfplumber is required. Run: pip install pdfplumber")
    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(_clean_pdf_text(text))
    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# Content chunker
# ---------------------------------------------------------------------------

class ContentChunker:
    """
    Turns a content PDF (textbook chapter or lecture notes) into Chunk objects.

    Strategy:
        1. Split on section headings (title-cased lines).
        2. If a section exceeds MAX_CHUNK_TOKENS, re-split at paragraph
           boundaries with CHUNK_OVERLAP_TOKENS overlap.
        3. Assign a stable chunk_id: {course_id}-content-{n:04d}.

    chunk_offset lets the caller accumulate a global index across multiple PDFs.
    """

    def __init__(
        self,
        max_tokens: int = MAX_CHUNK_TOKENS,
        overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, pdf_path: Path, course_id: str, chunk_offset: int = 0) -> list[Chunk]:
        text = _extract_text_from_pdf(pdf_path)
        raw_sections = _chunk_by_heading(text)

        chunks: list[Chunk] = []
        part_index = chunk_offset

        for section in raw_sections:
            if _approx_tokens(section) > self.max_tokens:
                sub_chunks = _split_long_section(section, self.max_tokens, self.overlap_tokens)
            else:
                sub_chunks = [section]

            for sub in sub_chunks:
                if not sub.strip():
                    continue
                if _approx_tokens(sub) < MIN_CHUNK_TOKENS:
                    continue  # skip page headers, author lines, etc.
                first_line = sub.split("\n")[0].strip()[:80]
                chunks.append(Chunk(
                    chunk_id=f"{course_id}-content-{part_index:04d}",
                    course_id=course_id,
                    source_file=pdf_path.name,
                    topic=first_line,
                    text=sub,
                ))
                part_index += 1

        logger.info("Chunked %s → %d chunks", pdf_path.name, len(chunks))
        return chunks


# ---------------------------------------------------------------------------
# Question extractor (assignments and exams)
# ---------------------------------------------------------------------------

class QuestionExtractor:
    """
    Extracts questions from an assignment or exam PDF.

    Handles two formats:
        - Numbered open-ended questions  (1. Prove that …)
        - MCQ items with lettered options (A. / B. / …)

    Produces a list of RawQuestion objects.
    """

    def __init__(self, min_chars: int = MIN_QUESTION_CHARS):
        self.min_chars = min_chars

    def extract(
        self,
        pdf_path: Path,
        course_id: str,
        source_type: str,
        q_offset: int = 0,
    ) -> list[RawQuestion]:
        """
        Args:
            pdf_path:    Path to the PDF.
            course_id:   Course identifier.
            source_type: "assignment" or "exam".
            q_offset:    Global question index offset for stable q_ids.
        """
        text = _extract_text_from_pdf(pdf_path)
        # Use only the last underscore-separated token as the label (strips hash prefix)
        stem_parts = pdf_path.stem.split("_")
        source_label = " ".join(stem_parts[-3:]).replace("-", " ").title() if len(stem_parts) > 1 else pdf_path.stem
        questions: list[RawQuestion] = []

        # Try "Problem N." style first (OCW / structured assignments)
        stems = _PROBLEM_LABEL_RE.findall(text)
        # Fall back to numbered-sentence style if nothing found
        if not stems:
            stems = _NUMBERED_STEM_RE.findall(text)

        for idx, stem_text in enumerate(stems):
            stem_text = re.sub(r"\s{2,}", " ", stem_text.strip())
            if len(stem_text) < self.min_chars:
                continue

            prefix = source_type[0].upper()  # "A" or "E"
            q_id = f"{course_id}-{prefix}{q_offset + idx + 1:04d}"
            options = self._extract_options(text, stem_text)

            questions.append(RawQuestion(
                q_id=q_id,
                course_id=course_id,
                text=stem_text,
                options=options if options else None,
                source_file=pdf_path.name,
                source_type=source_type,
                source_label=source_label,
            ))

        if not questions:
            logger.warning("No questions found in %s", pdf_path.name)
        else:
            logger.info(
                "Extracted %d %s questions from %s",
                len(questions), source_type, pdf_path.name,
            )

        return questions

    @staticmethod
    def _extract_options(full_text: str, stem: str) -> list[str]:
        """
        Find MCQ option lines that follow the stem in the full document text.
        Returns a list of option strings, or [] if none found.
        """
        stem_pos = full_text.find(stem)
        if stem_pos == -1:
            return []
        segment = full_text[stem_pos + len(stem): stem_pos + len(stem) + 800]
        option_matches = _MCQ_OPTION_RE.findall(segment)
        if not option_matches:
            return []
        # Extract the full option lines
        lines = segment.split("\n")
        options: list[str] = []
        for line in lines:
            if _MCQ_OPTION_RE.match(line):
                options.append(line.strip())
            elif options and line.strip() and not _MCQ_OPTION_RE.match(line):
                # Non-option line after options started — we've passed the option block
                break
        return options


# ---------------------------------------------------------------------------
# Objective extractor
# ---------------------------------------------------------------------------

class ObjectiveExtractor:
    """
    Parses learning objectives from a syllabus PDF.

    Looks for bullet-point or numbered lists under headings that contain
    "objective", "outcome", or "goal".
    """

    _SECTION_RE = re.compile(
        r"(learning\s+objectives?|course\s+objectives?|learning\s+outcomes?|goals?)",
        re.IGNORECASE,
    )
    _BULLET_RE = re.compile(r"^\s*[-•*]\s+(.+)", re.MULTILINE)
    _NUMBERED_RE = re.compile(r"^\s*\d+[.\)]\s+(.+)", re.MULTILINE)

    def extract(self, syllabus_path: Path) -> list[str]:
        text = _extract_text_from_pdf(syllabus_path)
        objectives: list[str] = []

        # Find the section that talks about objectives
        match = self._SECTION_RE.search(text)
        if match:
            section_text = text[match.start(): match.start() + 2000]
        else:
            # Fall back to the whole document
            section_text = text

        for pattern in (self._BULLET_RE, self._NUMBERED_RE):
            for m in pattern.finditer(section_text):
                obj = m.group(1).strip()
                if len(obj) > 10:
                    objectives.append(obj)

        logger.info(
            "Extracted %d objectives from %s",
            len(objectives),
            syllabus_path.name,
        )
        return objectives


# ---------------------------------------------------------------------------
# CourseExtractor — orchestrates all three extractors for one course
# ---------------------------------------------------------------------------

class CourseExtractor:
    """
    Runs the full Stage 2 extraction for a single OCW course directory.

    Expected input layout:
        course_dir/
            content/     *.pdf   (textbook chapters or lecture notes)
            assignments/ *.pdf
            exams/       *.pdf   (optional)
            syllabus.pdf         (optional)
            course_meta.json     (recommended — provides subject/field/level)

    Outputs written to out_dir/:
        chunks.jsonl
        questions_raw.jsonl
        objectives.txt
    """

    def __init__(
        self,
        course_dir: str | Path,
        out_dir: str | Path,
        course_id: str | None = None,
    ):
        self.course_dir = Path(course_dir)
        self.out_dir = Path(out_dir)
        self.course_id = course_id or self.course_dir.name
        self.chunker = ContentChunker()
        self.question_extractor = QuestionExtractor()
        self.objective_extractor = ObjectiveExtractor()
        self._meta: dict = self._load_meta()

    def _load_meta(self) -> dict:
        meta_path = self.course_dir / "course_meta.json"
        if not meta_path.exists():
            logger.warning(
                "course_meta.json not found in %s — subject/field/level will be empty.",
                self.course_dir,
            )
            return {}
        with meta_path.open(encoding="utf-8") as f:
            return json.load(f)

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._extract_content()
        self._extract_questions()
        self._extract_objectives()

        logger.info("Extraction complete for course %s → %s", self.course_id, self.out_dir)

    # ── Content chunks ────────────────────────────────────────────────────

    def _extract_content(self) -> None:
        content_dir = self.course_dir / "content"
        if not content_dir.exists():
            logger.warning("No content/ directory found in %s", self.course_dir)
            return

        all_chunks: list[Chunk] = []
        chunk_offset = 0
        for pdf in sorted(content_dir.glob("*.pdf")):
            try:
                chunks = self.chunker.chunk(pdf, self.course_id, chunk_offset=chunk_offset)
                chunk_offset += len(chunks)
                all_chunks.extend(chunks)
            except Exception as exc:
                logger.error("Failed to chunk %s: %s", pdf.name, exc)

        self._write_jsonl(
            self.out_dir / "chunks.jsonl",
            [asdict(c) for c in all_chunks],
        )
        logger.info("Wrote %d chunks to chunks.jsonl", len(all_chunks))

    # ── Questions (assignments + exams) ───────────────────────────────────

    def _extract_questions(self) -> None:
        all_questions: list[RawQuestion] = []
        q_offset = 0

        for source_type, subdir_name in [("assignment", "assignments"), ("exam", "exams")]:
            subdir = self.course_dir / subdir_name
            if not subdir.is_dir():
                continue
            for pdf in sorted(subdir.glob("*.pdf")):
                try:
                    questions = self.question_extractor.extract(
                        pdf,
                        self.course_id,
                        source_type=source_type,
                        q_offset=q_offset,
                    )
                    q_offset += len(questions)
                    all_questions.extend(questions)
                except Exception as exc:
                    logger.error("Failed to extract questions from %s: %s", pdf.name, exc)

        self._write_jsonl(
            self.out_dir / "questions_raw.jsonl",
            [asdict(q) for q in all_questions],
        )
        logger.info("Wrote %d questions to questions_raw.jsonl", len(all_questions))

    # ── Syllabus / objectives ──────────────────────────────────────────────

    def _extract_objectives(self) -> None:
        # Prefer objectives already in course_meta.json
        if self._meta.get("learning_objectives"):
            objectives = self._meta["learning_objectives"]
        else:
            syllabus = self.course_dir / "syllabus.pdf"
            if not syllabus.exists():
                logger.warning("No syllabus.pdf and no learning_objectives in course_meta.json")
                return
            try:
                objectives = self.objective_extractor.extract(syllabus)
            except Exception as exc:
                logger.error("Failed to extract objectives from syllabus: %s", exc)
                return

        out_path = self.out_dir / "objectives.txt"
        out_path.write_text("\n".join(objectives), encoding="utf-8")
        logger.info("Wrote %d objectives to objectives.txt", len(objectives))

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract text artefacts from a raw OCW course directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--course-id",
        required=True,
        help="Course identifier matching a subdirectory under data/raw/ (e.g. 6.042j)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    data_root = Path(args.data_dir)
    CourseExtractor(
        course_dir=data_root / "raw" / args.course_id,
        out_dir=data_root / "processed" / args.course_id,
        course_id=args.course_id,
    ).run()
