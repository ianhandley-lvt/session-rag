from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000)]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


TemporalScope = Literal["durable", "time_sensitive"]
SourceType = Literal["claude_session"]


class Attribution(BaseModel):
    """An untrusted, content-level claim that a transcript credits a decision to a
    specific person — never promoted to trusted provenance (see Operator ID)."""

    model_config = ConfigDict(extra="forbid")

    person: ShortText
    citation: ShortText


class ExtractedKnowledge(BaseModel):
    """Model-generated fields. Trusted provenance is deliberately absent."""

    model_config = ConfigDict(extra="forbid")

    question: NonEmptyText
    summary: NonEmptyText
    resolution: NonEmptyText | None = None
    systems: list[ShortText] = Field(default_factory=list, max_length=100)
    code_references: list[ShortText] = Field(default_factory=list, max_length=100)
    attribution: Attribution | None = None
    temporal_scope: TemporalScope | None = None
    timestamp: datetime | None = None
    # 0-indexed line number in the sanitized transcript that supports this
    # record. Model-proposed, application-validated (see CursorExtractor):
    # rejected (set to null) if it falls outside the sanitized content's
    # actual line range — never trusted as-is.
    evidence_location: int | None = None


class ProjectProvenance(BaseModel):
    """Trusted context describing which codebase a session concerns.

    Optional for sessions outside a Git repository. Never sent to an
    extraction provider — project_root in particular stays local-only.
    """

    model_config = ConfigDict(extra="forbid")

    project_id: ShortText
    project_root: NonEmptyText | None = None
    repository_revision: ShortText | None = None
    working_tree_dirty: bool | None = None


class StructuredRecord(ExtractedKnowledge):
    """Everything ExtractedKnowledge has, plus trusted provenance attached by
    application code — never accepted from model output."""

    model_config = ConfigDict(extra="forbid")

    source: NonEmptyText
    source_session_id: ShortText

    source_type: SourceType
    operator_id: ShortText
    project: ProjectProvenance | None = None
    prompt_version: int


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[ExtractedKnowledge] = Field(max_length=50)


class ExtractionError(RuntimeError):
    """A provider failed to return valid structured knowledge."""


class ExtractionBlocked(ExtractionError):
    """Input was rejected before an extraction attempt was made (e.g. oversized after sanitization)."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class ExtractionPendingRetry(ExtractionError):
    """Cursor was unavailable, timed out, or reported non-success (e.g. quota
    exhaustion) — an infra-level condition, retryable later without penalty.
    Distinct from a plain ExtractionError, which means the model's own output
    was invalid."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class KnowledgeExtractor(Protocol):
    name: str
    model: str
    prompt_version: int

    def extract(self, transcript: Path) -> list[StructuredRecord]: ...
