from __future__ import annotations

import json
import re
import sqlite3
import tempfile
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class SessionSource:
    source_type: str
    source_id: str
    path: Path
    project_id: str | None = None
    project_root: Path | None = None
    source_uri: str | None = None
    updated_at: float | None = None


def claude_sessions(projects: dict, claude_home: Path, project_id: str | None = None) -> list[SessionSource]:
    selected = {project_id: projects[project_id]} if project_id else projects
    found: list[SessionSource] = []
    for identifier, project in selected.items():
        encoded = re.sub(r"[^A-Za-z0-9_-]", "-", str(project.root.expanduser().resolve()))
        for path in (claude_home / "projects" / encoded).glob("*.jsonl"):
            found.append(SessionSource("claude_session", path.stem, path, identifier, project.root, updated_at=path.stat().st_mtime))
    return sorted(found, key=lambda item: item.path.stat().st_mtime_ns)


def cursor_sessions(database: Path, output_dir: Path) -> list[SessionSource]:
    """Read Cursor's local conversation search index through a SQLite snapshot.

    Only local rows are imported; cloud-cache rows can duplicate them. Cursor's
    root fingerprint is intentionally not interpreted as project provenance.
    """
    if not database.exists():
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        snapshot = Path(temporary) / "cursor.db"
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as source, sqlite3.connect(snapshot) as target:
            source.backup(target)
        with sqlite3.connect(snapshot) as connection:
            rows = connection.execute(
                "SELECT c.id, c.title, f.body, c.updated_at FROM conversations c "
                "JOIN conversation_fts f ON f.rowid = c.fts_rowid WHERE c.source = 'local' ORDER BY c.updated_at"
            ).fetchall()
    result = []
    for identifier, title, body, updated_at in rows:
        safe_identifier = hashlib.sha256(identifier.encode()).hexdigest()
        path = output_dir / f"{safe_identifier}.jsonl"
        payload = {"type": "user", "uuid": identifier, "message": {"content": f"{title}\n\n{body}"}}
        path.write_text(json.dumps(payload) + "\n")
        timestamp = float(updated_at)
        if timestamp > 100_000_000_000:  # Cursor stores Unix milliseconds.
            timestamp /= 1000
        result.append(SessionSource("cursor_session", identifier, path, source_uri=f"cursor://conversation/{identifier}", updated_at=timestamp))
    return result


def parse_since(value: str | None) -> float | None:
    if value is None:
        return None
    return datetime.fromisoformat(value).timestamp()
