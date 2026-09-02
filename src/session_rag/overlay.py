from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .artifacts import find_record
from .jsonio import atomic_write_json, read_json

VerificationStatus = Literal["unreviewed", "verified", "rejected", "superseded"]

EXCLUDED_FROM_SEARCH: frozenset[VerificationStatus] = frozenset({"rejected", "superseded"})

_ALLOWED_TRANSITIONS: dict[VerificationStatus, frozenset[VerificationStatus]] = {
    "unreviewed": frozenset({"verified", "rejected", "superseded"}),
    "verified": frozenset({"rejected", "superseded"}),
}

DEFAULT_STATE: dict = {"verification_status": "unreviewed", "superseded_by": None}


class InvalidTransition(ValueError):
    """The requested verification_status transition isn't allowed from the
    record's current state."""


class SupersedeRequiresReplacement(ValueError):
    """supersede was called without a replacement record id."""


class UnknownReplacementRecord(ValueError):
    """The replacement record id doesn't refer to any known Episode Record —
    a superseded_by link must never dangle."""


def overlay_path(root: Path) -> Path:
    return root / "overlay.json"


def _read_overlay(root: Path) -> dict:
    path = overlay_path(root)
    if not path.exists():
        return {}
    return read_json(path)


def read_state(root: Path, record_id: str) -> dict:
    """verification_status + supersession link for one record. Durable,
    independent of both the Extraction Artifact (immutable) and LanceDB (a
    disposable derived index) — surviving a full LanceDB rebuild is exactly
    the property this storage exists to guarantee."""

    return _read_overlay(root).get(record_id, dict(DEFAULT_STATE))


def _write_state(root: Path, record_id: str, status: VerificationStatus, superseded_by: str | None) -> None:
    overlay = _read_overlay(root)
    overlay[record_id] = {
        "verification_status": status,
        "superseded_by": superseded_by,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(overlay_path(root), overlay)


def _transition(
    root: Path, record_id: str, to_status: VerificationStatus, *, superseded_by: str | None = None
) -> None:
    current = read_state(root, record_id)["verification_status"]
    if to_status not in _ALLOWED_TRANSITIONS.get(current, frozenset()):
        raise InvalidTransition(f"record {record_id} cannot move from {current!r} to {to_status!r}")
    _write_state(root, record_id, to_status, superseded_by)


def verify(root: Path, record_id: str) -> None:
    _transition(root, record_id, "verified")


def reject(root: Path, record_id: str) -> None:
    _transition(root, record_id, "rejected")


def supersede(root: Path, record_id: str, replacement_id: str | None) -> None:
    if not replacement_id:
        raise SupersedeRequiresReplacement("supersede requires a replacement record id")
    if find_record(root, replacement_id) is None:
        raise UnknownReplacementRecord(f"replacement record {replacement_id} does not exist")
    _transition(root, record_id, "superseded", superseded_by=replacement_id)


def filter_retrievable(root: Path, records: list[dict]) -> list[dict]:
    """Drop rejected/superseded records before they ever reach the index —
    excluded records still exist in their (immutable) artifact and remain
    reachable via history lookup, just not through normal search."""

    return [record for record in records if read_state(root, record["id"])["verification_status"] not in EXCLUDED_FROM_SEARCH]
