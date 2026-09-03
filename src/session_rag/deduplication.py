from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .overlay import DuplicateStateUpdate, EXCLUDED_FROM_SEARCH, read_states, write_duplicate_states
from .store import Embedder, retrieval_text

DEFAULT_SEMANTIC_DUPLICATE_THRESHOLD = 0.92


@dataclass(frozen=True)
class DeduplicationResult:
    exact_duplicates: int
    possible_duplicates: int


def _normalized_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip().casefold()


def _fingerprint(record: dict) -> str:
    content = {
        "question": _normalized_text(record.get("question")),
        "summary": _normalized_text(record.get("summary")),
        "resolution": _normalized_text(record.get("resolution")),
        "systems": sorted(_normalized_text(value) for value in record.get("systems", [])),
        "code_references": sorted(_normalized_text(value) for value in record.get("code_references", [])),
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _project_id(record: dict) -> str | None:
    return (record.get("project") or {}).get("project_id")


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


def reconcile_duplicates(
    artifacts_root: Path,
    records: list[dict],
    embedder: Embedder,
    *,
    semantic_threshold: float = DEFAULT_SEMANTIC_DUPLICATE_THRESHOLD,
) -> DeduplicationResult:
    """Rebuild derived duplicate links within each trusted project scope.

    Exact duplicates retain their artifacts and provenance but only the oldest
    record remains retrievable. Semantic matches remain retrievable and are
    flagged for human review; no fuzzy match is silently merged or removed.
    """

    states = read_states(artifacts_root, [record["id"] for record in records])
    eligible = [record for record in records if states[record["id"]]["verification_status"] not in EXCLUDED_FROM_SEARCH]
    exact_count = 0
    possible_count = 0
    canonical_records: list[dict] = []
    updates: dict[str, DuplicateStateUpdate] = {
        record["id"]: {"duplicate_of": None, "reinforces": [], "duplicate_review_status": None}
        for record in eligible
    }
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in eligible:
        project_id = _project_id(record)
        if project_id is not None:
            groups.setdefault((project_id, _fingerprint(record)), []).append(record)

    for group in groups.values():
        ordered = sorted(group, key=lambda record: (record.get("extracted_at", ""), record["id"]))
        canonical_records.append(ordered[0])
        for duplicate in ordered[1:]:
            updates[duplicate["id"]] = {
                "duplicate_of": ordered[0]["id"],
                "reinforces": [],
                "duplicate_review_status": "exact",
            }
            exact_count += 1

    ordered = sorted(canonical_records, key=lambda record: (record.get("extracted_at", ""), record["id"]))
    vectors = embedder.embed([retrieval_text(record) for record in ordered]) if ordered else []
    for index, record in enumerate(ordered):
        links = []
        for earlier_index in range(index):
            earlier = ordered[earlier_index]
            if _project_id(record) != _project_id(earlier):
                continue
            similarity = _cosine(vectors[index], vectors[earlier_index])
            if similarity >= semantic_threshold:
                links.append({"record_id": earlier["id"], "similarity": round(similarity, 6)})
        updates[record["id"]] = {
            "duplicate_of": None,
            "reinforces": links,
            "duplicate_review_status": "possible" if links else None,
        }
        if links:
            possible_count += 1
    write_duplicate_states(artifacts_root, updates)
    return DeduplicationResult(exact_count, possible_count)
