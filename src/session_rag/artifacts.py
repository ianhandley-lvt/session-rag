from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .extractors.base import SourceType, StructuredRecord

SCHEMA_VERSION = 1


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
    path.parent.mkdir(parents=True, exist_ok=True)

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

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-artifact-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(envelope, handle, indent=2)
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path
