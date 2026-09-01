from .base import KnowledgeExtractor, StructuredRecord
from .cursor import CursorExtractor


def create_extractor(
    name: str,
    *,
    cursor_mode: str | None = None,
    cursor_model: str | None = None,
) -> KnowledgeExtractor:
    if name == "cursor":
        return CursorExtractor(mode=cursor_mode, model=cursor_model)
    raise ValueError(f"Unknown extractor: {name}")


__all__ = ["CursorExtractor", "KnowledgeExtractor", "StructuredRecord", "create_extractor"]
