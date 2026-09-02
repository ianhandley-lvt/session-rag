from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .extractors.base import SourceType, StructuredRecord
from .jsonio import atomic_write_json, read_json

SCHEMA_VERSION = 1

JobStatus = Literal["pending_retry", "failed", "blocked"]


def source_hash(transcript: Path) -> str:
    """Content fingerprint identifying a source revision. Changes iff the
    transcript's bytes change — this is what an Active Revision keys on."""

    digest = hashlib.sha256(transcript.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def artifact_path(root: Path, *, source_type: SourceType, source_id: str, hash_value: str) -> Path:
    # ':' is invalid in filenames on some platforms; the hash is still unique without it.
    safe_hash = hash_value.replace(":", "-")
    return root / source_type / source_id / f"{safe_hash}.json"


def _record_id(hash_value: str, index: int) -> str:
    """Deterministic, artifact-scoped — never derived from a LanceDB row."""

    return f"{hash_value}:{index}"


def _source_dir(root: Path, *, source_type: SourceType, source_id: str) -> Path:
    return root / source_type / source_id


def write_artifact(
    root: Path,
    *,
    source_type: SourceType,
    source_id: str,
    source_uri: str,
    hash_value: str,
    extractor: str,
    extractor_model: str,
    prompt_version: int,
    episode_records: list[StructuredRecord],
) -> Path:
    """Persist one immutable Extraction Artifact, atomically (temp file + rename).

    Keyed by source_id + hash_value. Artifacts are immutable: if one already
    exists for this exact (source_id, hash_value), this is a no-op that
    returns the existing path unchanged — re-extracting a non-deterministic
    LLM output for the same source content must never silently replace
    content or reassign an existing Episode Record's id. A changed hash
    writes to a different path, leaving the prior revision's file untouched.
    """

    path = artifact_path(root, source_type=source_type, source_id=source_id, hash_value=hash_value)
    if path.exists():
        return path

    envelope = {
        "schema_version": SCHEMA_VERSION,
        "source_type": source_type,
        "source_id": source_id,
        "source_uri": source_uri,
        "source_hash": hash_value,
        "extractor": extractor,
        "extractor_model": extractor_model,
        "prompt_version": prompt_version,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "episode_records": [
            {"id": _record_id(hash_value, index), **record.model_dump(mode="json")}
            for index, record in enumerate(episode_records)
        ],
    }
    atomic_write_json(path, envelope)
    return path


def read_artifact(path: Path) -> dict:
    return read_json(path)


def active_revision_path(root: Path, *, source_type: SourceType, source_id: str) -> Path:
    return _source_dir(root, source_type=source_type, source_id=source_id) / "active.json"


def read_active_hash(root: Path, *, source_type: SourceType, source_id: str) -> str | None:
    path = active_revision_path(root, source_type=source_type, source_id=source_id)
    if not path.exists():
        return None
    return read_artifact(path)["active_hash"]


def set_active_hash(root: Path, *, source_type: SourceType, source_id: str, hash_value: str) -> None:
    """Atomically mark hash_value as the retrieval-eligible revision for this
    source. Only ever called after the corresponding artifact is fully
    written and validated — never on a failed or partial extraction."""

    path = active_revision_path(root, source_type=source_type, source_id=source_id)
    atomic_write_json(path, {"active_hash": hash_value, "activated_at": datetime.now(timezone.utc).isoformat()})


def job_status_path(root: Path, *, source_type: SourceType, source_id: str) -> Path:
    return _source_dir(root, source_type=source_type, source_id=source_id) / "job_status.json"


def write_job_status(
    root: Path, *, source_type: SourceType, source_id: str, status: JobStatus, reason: str, attempted_hash: str
) -> None:
    """Non-sensitive metadata identifying which source revision needs a
    retry — no transcript content, no secrets. Retained until the next
    successful extraction of this source clears it."""

    path = job_status_path(root, source_type=source_type, source_id=source_id)
    atomic_write_json(
        path,
        {
            "status": status,
            "reason": reason,
            "attempted_hash": attempted_hash,
            "attempted_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def clear_job_status(root: Path, *, source_type: SourceType, source_id: str) -> None:
    job_status_path(root, source_type=source_type, source_id=source_id).unlink(missing_ok=True)


def _iter_source_dirs(root: Path) -> Iterator[tuple[str, Path]]:
    """Yields (source_type, source_dir) for every source directory under root."""

    if not root.exists():
        return
    for source_type_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for source_dir in sorted(p for p in source_type_dir.iterdir() if p.is_dir()):
            yield source_type_dir.name, source_dir


def _denormalize(record: dict, envelope: dict) -> dict:
    """A record as read from an artifact carries only its own fields —
    resolving a citation back to the exact artifact needs the envelope's
    source_type/source_id/source_hash too."""

    return {
        **record,
        "source_type": envelope["source_type"],
        "source_id": envelope["source_id"],
        "source_hash": envelope["source_hash"],
    }


def load_active_episode_records(root: Path) -> list[dict]:
    """Every Episode Record from every source's currently Active Revision —
    the sole input to building the LanceDB index. Never reads raw session
    transcripts; sources with no active revision (never extracted, or only
    ever failed/blocked) contribute nothing."""

    records: list[dict] = []
    for source_type, source_dir in _iter_source_dirs(root):
        active_hash = read_active_hash(root, source_type=source_type, source_id=source_dir.name)
        if active_hash is None:
            continue
        path = artifact_path(root, source_type=source_type, source_id=source_dir.name, hash_value=active_hash)
        if not path.exists():
            continue
        envelope = read_artifact(path)
        records.extend(_denormalize(record, envelope) for record in envelope["episode_records"])
    return records


def _iter_source_envelopes(source_dir: Path) -> Iterator[dict]:
    """Every artifact envelope (any revision, active or not) in one source
    directory — the shared scan `find_record` and `forget_source` both need."""

    for artifact_file in sorted(source_dir.glob("sha256-*.json")):
        yield read_artifact(artifact_file)


def find_record(root: Path, record_id: str) -> dict | None:
    """Locate one Episode Record by its stable id, across every artifact —
    active or not. Verification/history commands take only a record id, so
    this is a full scan; the local corpus this targets makes that fine."""

    for _, source_dir in _iter_source_dirs(root):
        for envelope in _iter_source_envelopes(source_dir):
            for record in envelope["episode_records"]:
                if record["id"] == record_id:
                    return _denormalize(record, envelope)
    return None


def forget_source(root: Path, source_id: str) -> list[str]:
    """Hard-delete every artifact revision (all hashes, active pointer, job
    status) for one source, across whichever source_type it lives under.
    Returns the record ids that existed, so callers can also purge Record
    State Overlay entries — forget must leave no trace anywhere."""

    record_ids: list[str] = []
    for _, source_dir in _iter_source_dirs(root):
        if source_dir.name != source_id:
            continue
        for envelope in _iter_source_envelopes(source_dir):
            record_ids.extend(record["id"] for record in envelope["episode_records"])
        shutil.rmtree(source_dir)
    return record_ids
