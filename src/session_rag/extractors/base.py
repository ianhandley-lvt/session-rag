from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000)]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


class ExtractedKnowledge(BaseModel):
    """Model-generated fields. Provenance is deliberately absent."""

    model_config = ConfigDict(extra="forbid")

    question: NonEmptyText
    summary: NonEmptyText
    resolution: NonEmptyText | None = None
    systems: list[ShortText] = Field(default_factory=list, max_length=100)
    code_references: list[ShortText] = Field(default_factory=list, max_length=100)
    author: ShortText | None = None
    timestamp: datetime | None = None


class StructuredRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: NonEmptyText
    summary: NonEmptyText
    resolution: NonEmptyText | None = None
    systems: list[ShortText] = Field(default_factory=list, max_length=100)
    code_references: list[ShortText] = Field(default_factory=list, max_length=100)
    author: ShortText | None = None
    source: NonEmptyText
    source_session_id: ShortText
    timestamp: datetime | None = None
    authority: Literal["working_session", "verified_decision", "unknown"] = "working_session"


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[ExtractedKnowledge] = Field(max_length=50)


class ExtractionError(RuntimeError):
    """A provider failed to return valid structured knowledge."""


class KnowledgeExtractor(Protocol):
    def extract(self, transcript: Path) -> list[StructuredRecord]: ...
