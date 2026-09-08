from .base import KnowledgeExtractor, ProjectProvenance, StructuredRecord
from .cursor import CursorExtractor


def create_extractor(
    name: str,
    *,
    cursor_mode: str | None = None,
    cursor_model: str | None = None,
    max_sanitized_chars: int | None = None,
    operator_id: str | None = None,
    project: ProjectProvenance | None = None,
) -> KnowledgeExtractor:
    if name == "cursor":
        return CursorExtractor(
            mode=cursor_mode,
            model=cursor_model,
            max_sanitized_chars=max_sanitized_chars,
            operator_id=operator_id,
            project=project,
        )
    raise ValueError(f"Unknown extractor: {name}")


__all__ = ["CursorExtractor", "KnowledgeExtractor", "StructuredRecord", "create_extractor"]
