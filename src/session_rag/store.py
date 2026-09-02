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


def _rrf(result_lists: list[list[dict]], limit: int) -> list[dict]:
    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}
    for result_list in result_lists:
        for rank, row in enumerate(result_list, start=1):
            row_id = row["id"]
            scores[row_id] = scores.get(row_id, 0.0) + 1.0 / (60 + rank)
            rows[row_id] = row
    return [rows[row_id] for row_id in sorted(scores, key=scores.get, reverse=True)[:limit]]


def search_episode_records(database: Path, query: str, embedder: Embedder, limit: int = 4) -> list[dict]:
    if not database.exists():
        return []
    connection = lancedb.connect(database)
    if TABLE_NAME not in connection.table_names():
        return []
    table = connection.open_table(TABLE_NAME)
    query_vector = embedder.embed([query])[0]
    vector_rows = table.search(query_vector, vector_column_name="vector").limit(limit * 3).to_list()
    try:
        text_rows = (
            table.search(query, query_type="fts", fts_columns="text")
            .limit(limit * 3)
            .to_list()
        )
    except Exception:
        text_rows = []
    return _rrf([vector_rows, text_rows], limit)
