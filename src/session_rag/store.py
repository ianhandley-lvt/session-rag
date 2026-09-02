from __future__ import annotations

from pathlib import Path
from typing import Protocol

import lancedb

TABLE_NAME = "episode_records"


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _retrieval_text(record: dict) -> str:
    """Assembled from question/summary/resolution/systems/code_references —
    the fields a search over durable knowledge should match against, not the
    raw conversational transcript."""

    parts = [record["question"], record["summary"]]
    if record.get("resolution"):
        parts.append(record["resolution"])
    if record.get("systems"):
        parts.append(", ".join(record["systems"]))
    if record.get("code_references"):
        parts.append(", ".join(record["code_references"]))
    return "\n".join(parts)


def index_episode_records(database: Path, records: list[dict], embedder: Embedder) -> int:
    """Build the LanceDB index purely by replaying Episode Records (from
    Extraction Artifacts) — never raw transcripts. LanceDB is a disposable
    derived index (ADR-0001): deleting it and re-running this against the
    same artifacts reproduces it exactly, with no re-extraction."""

    if not records:
        return 0
    database.mkdir(parents=True, exist_ok=True)
    texts = [_retrieval_text(record) for record in records]
    vectors = embedder.embed(texts)
    rows = [
        {
            "id": record["id"],
            "question": record["question"],
            "summary": record["summary"],
            "text": text,
            "source": record["source"],
            "source_type": record["source_type"],
            "source_id": record["source_id"],
            "source_session_id": record["source_session_id"],
            "source_hash": record["source_hash"],
            "temporal_scope": record.get("temporal_scope") or "",
            "timestamp": record.get("timestamp") or "",
            "embedding_model": embedder.model_name,
            "vector": vector,
        }
        for record, text, vector in zip(records, texts, vectors, strict=True)
    ]
    connection = lancedb.connect(database)
    table = connection.create_table(TABLE_NAME, data=rows, mode="overwrite")
    table.create_fts_index("text", replace=True)
    return len(rows)
