from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .extractors.base import EvidenceLocation, ProjectProvenance, StructuredRecord, TemporalScope

MAX_SECTION_CHARS = 3_000
NAVIGATION_FILES = frozenset({"INDEX.md", "QUESTIONS.md"})

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_LAST_UPDATED = re.compile(r"^\*\*Last updated:\*\*\s*(.+?)\s*$", re.MULTILINE)
_STATUS = re.compile(r"^\*\*Status:\*\*\s*(.+?)\s*$", re.MULTILINE)
_SOURCES = re.compile(r"^\*\*Sources:\*\*\s*(.+?)\s*$", re.MULTILINE)
_WIKILINK = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_BACKTICKED = re.compile(r"`([^`\n]+)`")
_NON_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(value: str) -> str:
    return _NON_SLUG.sub("-", value.lower()).strip("-") or "section"


def markdown_source_id(knowledge_base_id: str, wiki_dir: Path, article: Path) -> str:
    relative = article.relative_to(wiki_dir).with_suffix("")
    relative_slug = "--".join(_slug(part) for part in relative.parts)
    return f"{_slug(knowledge_base_id)}--{relative_slug}"


def markdown_articles(wiki_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in wiki_dir.rglob("*.md")
        if path.is_file() and path.name not in NAVIGATION_FILES
    )


@dataclass(frozen=True)
class MarkdownSection:
    heading_path: tuple[str, ...]
    text: str


def _sections(markdown: str) -> tuple[str, list[MarkdownSection]]:
    title = "Untitled"
    heading_stack: list[tuple[int, str]] = []
    current_path: tuple[str, ...] | None = None
    current_lines: list[str] = []
    sections: list[MarkdownSection] = []

    def flush() -> None:
        if current_path is None:
            return
        text = "\n".join(current_lines).strip()
        if text:
            sections.append(MarkdownSection(current_path, text))

    for line in markdown.splitlines():
        match = _HEADING.match(line)
        if not match:
            if current_path is not None:
                current_lines.append(line)
            continue

        level = len(match.group(1))
        heading = match.group(2).strip()
        if level == 1:
            title = heading
            continue

        flush()
        current_lines = []
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, heading))
        current_path = tuple(item[1] for item in heading_stack)

    flush()
    return title, sections


def _split_long_block(block: str, limit: int) -> list[str]:
    if len(block) <= limit:
        return [block]
    words = block.split()
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for word in words:
        added = len(word) + (1 if current else 0)
        if current and size + added > limit:
            pieces.append(" ".join(current))
            current = []
            size = 0
        current.append(word)
        size += len(word) + (1 if len(current) > 1 else 0)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _bounded_parts(text: str, limit: int = MAX_SECTION_CHARS) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    normalized = [piece for block in blocks for piece in _split_long_block(block, limit)]
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for block in normalized:
        added = len(block) + (2 if current else 0)
        if current and size + added > limit:
            parts.append("\n\n".join(current))
            current = []
            size = 0
        current.append(block)
        size += len(block) + (2 if len(current) > 1 else 0)
    if current:
        parts.append("\n\n".join(current))
    return parts


def _timestamp(markdown: str) -> datetime | None:
    match = _LAST_UPDATED.search(markdown)
    if not match:
        return None
    try:
        parsed = datetime.fromisoformat(match.group(1).strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _document_metadata(markdown: str) -> tuple[str | None, list[str]]:
    status_match = _STATUS.search(markdown)
    sources_match = _SOURCES.search(markdown)
    status = status_match.group(1).strip() if status_match else None
    references = _WIKILINK.findall(sources_match.group(1)) if sources_match else []
    return status, list(dict.fromkeys(references))[:100]


def _configured(value: str | None, env_name: str) -> str:
    resolved = value or os.getenv(env_name, "")
    if not resolved:
        raise ValueError(f"{env_name} must be configured")
    return resolved


class MarkdownKnowledgeBaseExtractor:
    """Deterministically maps curated Markdown sections to Episode Records.

    This is a source adapter, not an LLM extractor: article prose is already
    curated knowledge, so it is preserved rather than summarized again.
    """

    name = "markdown"
    model = "deterministic"
    prompt_version = 1

    def __init__(
        self,
        *,
        knowledge_base_id: str,
        project_id: str,
        project_root: str | None = None,
        operator_id: str | None = None,
        temporal_scope: TemporalScope = "time_sensitive",
    ) -> None:
        self.knowledge_base_id = knowledge_base_id
        self.project_id = project_id
        self.operator_id = _configured(operator_id, "SESSION_RAG_OPERATOR_ID")
        self.project = ProjectProvenance(project_id=project_id, project_root=project_root)
        self.temporal_scope = temporal_scope

    def extract(self, article: Path) -> list[StructuredRecord]:
        markdown = article.read_text(errors="replace")
        title, sections = _sections(markdown)
        timestamp = _timestamp(markdown)
        document_status, source_references = _document_metadata(markdown)
        source_id = article.stem
        records: list[StructuredRecord] = []
        identifiers: dict[str, int] = {}

        for section in sections:
            parts = _bounded_parts(section.text)
            base_identifier = "/".join(_slug(heading) for heading in section.heading_path)
            occurrence = identifiers.get(base_identifier, 0) + 1
            identifiers[base_identifier] = occurrence
            if occurrence > 1:
                base_identifier = f"{base_identifier}/occurrence-{occurrence}"

            for part_number, text in enumerate(parts, start=1):
                identifier = base_identifier if len(parts) == 1 else f"{base_identifier}/part-{part_number}"
                question = f"{title}: {' > '.join(section.heading_path)}"
                if len(parts) > 1:
                    question += f" (part {part_number})"
                records.append(
                    StructuredRecord(
                        question=question,
                        summary=text,
                        systems=[self.knowledge_base_id],
                        code_references=list(dict.fromkeys(_BACKTICKED.findall(text)))[:100],
                        temporal_scope=self.temporal_scope,
                        timestamp=timestamp,
                        evidence_location=EvidenceLocation(identifier=identifier, preserved_text=text),
                        source=str(article.resolve()),
                        source_session_id=source_id,
                        source_type="markdown_knowledge_base",
                        operator_id=self.operator_id,
                        project=self.project,
                        prompt_version=self.prompt_version,
                        document_status=document_status,
                        source_references=source_references,
                    )
                )
        return records
