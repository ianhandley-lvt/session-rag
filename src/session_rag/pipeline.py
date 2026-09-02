from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .artifacts import (
    JobStatus,
    artifact_path,
    clear_job_status,
    read_active_hash,
    read_artifact,
    set_active_hash,
    source_hash,
    write_artifact,
    write_job_status,
)
from .extractors.base import (
    ExtractionBlocked,
    ExtractionError,
    ExtractionPendingRetry,
    KnowledgeExtractor,
    SourceType,
    StructuredRecord,
)

OutcomeStatus = Literal["activated", "no_op", "pending_retry", "failed", "blocked"]


@dataclass
class ExtractionOutcome:
    status: OutcomeStatus
    hash_value: str
    artifact_path: Path | None = None
    reason: str | None = None
    orphaned_questions: list[str] = field(default_factory=list)


def _orphaned_questions(old_records: list[dict], new_questions: set[str]) -> list[str]:
    """Records from the prior active revision with no obvious counterpart in
    the new one — a simple, informational nudge (question-text match), not
    record-level reconciliation. Never changes any record's state."""

    return [record["question"] for record in old_records if record["question"] not in new_questions]


def _activate(
    root: Path,
    *,
    source_type: SourceType,
    source_id: str,
    current_active: str | None,
    hash_value: str,
    new_records: list[dict],
) -> list[str]:
    orphaned: list[str] = []
    if current_active and current_active != hash_value:
        old_path = artifact_path(root, source_type=source_type, source_id=source_id, hash_value=current_active)
        if old_path.exists():
            new_questions = {record["question"] for record in new_records}
            orphaned = _orphaned_questions(read_artifact(old_path)["episode_records"], new_questions)

    set_active_hash(root, source_type=source_type, source_id=source_id, hash_value=hash_value)
    clear_job_status(root, source_type=source_type, source_id=source_id)
    return orphaned


def run_extraction(
    extractor: KnowledgeExtractor,
    transcript: Path,
    artifacts_root: Path,
    *,
    source_type: SourceType = "claude_session",
) -> ExtractionOutcome:
    """Extract, persist, and activate one source revision — or record why it
    didn't happen. Never writes a partial artifact and never moves the
    Active Revision pointer except after a fully validated extraction."""

    source_id = transcript.stem
    hash_value = source_hash(transcript)
    current_active = read_active_hash(artifacts_root, source_type=source_type, source_id=source_id)
    existing_path = artifact_path(artifacts_root, source_type=source_type, source_id=source_id, hash_value=hash_value)

    def activate(new_records: list[dict]) -> list[str]:
        return _activate(
            artifacts_root,
            source_type=source_type,
            source_id=source_id,
            current_active=current_active,
            hash_value=hash_value,
            new_records=new_records,
        )

    def record_failure(status: JobStatus, reason: str) -> ExtractionOutcome:
        write_job_status(
            artifacts_root,
            source_type=source_type,
            source_id=source_id,
            status=status,
            reason=reason,
            attempted_hash=hash_value,
            # Not part of the KnowledgeExtractor Protocol — introspected
            # defensively so extractors that don't expose it (e.g. simple
            # test doubles) still work, just without project attribution.
            project_id=getattr(extractor, "project_id", None),
        )
        return ExtractionOutcome(status=status, hash_value=hash_value, reason=reason)

    if existing_path.exists():
        if current_active == hash_value:
            return ExtractionOutcome(status="no_op", hash_value=hash_value, artifact_path=existing_path)
        orphaned = activate(read_artifact(existing_path)["episode_records"])
        return ExtractionOutcome(
            status="activated", hash_value=hash_value, artifact_path=existing_path, orphaned_questions=orphaned
        )

    try:
        records: list[StructuredRecord] = extractor.extract(transcript)
    except ExtractionBlocked as error:
        return record_failure("blocked", error.reason)
    except ExtractionPendingRetry as error:
        return record_failure("pending_retry", error.reason)
    except ExtractionError as error:
        return record_failure("failed", str(error))

    written_path = write_artifact(
        artifacts_root,
        source_type=source_type,
        source_id=source_id,
        source_uri=str(transcript.resolve()),
        hash_value=hash_value,
        extractor=extractor.name,
        extractor_model=extractor.model,
        prompt_version=extractor.prompt_version,
        episode_records=records,
    )
    orphaned = activate([record.model_dump(mode="json") for record in records])
    return ExtractionOutcome(
        status="activated", hash_value=hash_value, artifact_path=written_path, orphaned_questions=orphaned
    )
