from __future__ import annotations

from pathlib import Path
from typing import Protocol

import lancedb

from .transcripts import SessionMemory


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


TABLE_NAME = "session_memories"


def index_memories(database: Path, memories: list[SessionMemory], embedder: Embedder) -> int:
    if not memories:
        return 0
    database.mkdir(parents=True, exist_ok=True)
    vectors = embedder.embed([memory.text for memory in memories])
    rows = [
        {
            "id": memory.id,
            "session_id": memory.session_id,
            "text": memory.text,
            "source": memory.source,
            "timestamp": memory.timestamp,
            "embedding_model": embedder.model_name,
            "vector": vector,
        }
        for memory, vector in zip(memories, vectors, strict=True)
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


def search_memories(database: Path, query: str, embedder: Embedder, limit: int = 4) -> list[dict]:
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

