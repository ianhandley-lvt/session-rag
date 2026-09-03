from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypedDict

from .artifacts import find_record
from .jsonio import atomic_write_json, read_json

VerificationStatus = Literal["unreviewed", "verified", "rejected", "superseded"]
DuplicateReviewStatus = Literal["exact", "possible"]


class ReinforcementLink(TypedDict):
    record_id: str
    similarity: float


class DuplicateStateUpdate(TypedDict):
    duplicate_of: str | None
    reinforces: list[ReinforcementLink]
    duplicate_review_status: DuplicateReviewStatus | None

EXCLUDED_FROM_SEARCH: frozenset[VerificationStatus] = frozenset({"rejected", "superseded"})

_ALLOWED_TRANSITIONS: dict[VerificationStatus, frozenset[VerificationStatus]] = {
    "unreviewed": frozenset({"verified", "rejected", "superseded"}),
    "verified": frozenset({"rejected", "superseded"}),
}

DEFAULT_STATE: dict = {
    "verification_status": "unreviewed",
    "superseded_by": None,
    "duplicate_of": None,
    "reinforces": [],
    "duplicate_review_status": None,
}


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

    return {**DEFAULT_STATE, **_read_overlay(root).get(record_id, {})}


def read_states(root: Path, record_ids: list[str]) -> dict[str, dict]:
    """Read lifecycle and duplicate state for a corpus with one file parse."""

    overlay = _read_overlay(root)
    return {record_id: {**DEFAULT_STATE, **overlay.get(record_id, {})} for record_id in record_ids}


def _write_state(root: Path, record_id: str, status: VerificationStatus, superseded_by: str | None) -> None:
    overlay = _read_overlay(root)
    overlay[record_id] = {
        **overlay.get(record_id, {}),
        "verification_status": status,
        "superseded_by": superseded_by,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(overlay_path(root), overlay)


def write_duplicate_states(root: Path, updates: dict[str, DuplicateStateUpdate]) -> None:
    """Atomically replace derived duplicate metadata for a corpus rebuild."""
    overlay = _read_overlay(root)
    checked_at = datetime.now(timezone.utc).isoformat()
    for record_id, update in updates.items():
        overlay[record_id] = {**overlay.get(record_id, {}), **update, "duplicate_checked_at": checked_at}
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


def forget_records(root: Path, record_ids: list[str]) -> None:
    """Purge these ids from the overlay entirely — part of forget's erasure
    guarantee. Not an overlay entry itself; leaves nothing behind."""

    if not record_ids:
        return
    overlay = _read_overlay(root)
    changed = False
    forgotten = set(record_ids)
    for record_id in record_ids:
        if overlay.pop(record_id, None) is not None:
            changed = True
    for state in overlay.values():
        if state.get("duplicate_of") in forgotten:
            state["duplicate_of"] = None
            state["duplicate_review_status"] = None
            changed = True
        original = state.get("reinforces", [])
        remaining = [link for link in original if link.get("record_id") not in forgotten]
        if remaining != original:
            state["reinforces"] = remaining
            if not remaining and state.get("duplicate_review_status") == "possible":
                state["duplicate_review_status"] = None
            changed = True
    if changed:
        atomic_write_json(overlay_path(root), overlay)


def filter_retrievable(root: Path, records: list[dict]) -> list[dict]:
    """Drop rejected/superseded records before they ever reach the index —
    excluded records still exist in their (immutable) artifact and remain
    reachable via history lookup, just not through normal search."""

    overlay = _read_overlay(root)
    retrievable = []
    for record in records:
        state = {**DEFAULT_STATE, **overlay.get(record["id"], {})}
        if state["verification_status"] not in EXCLUDED_FROM_SEARCH and state["duplicate_of"] is None:
            retrievable.append(record)
    return retrievable
