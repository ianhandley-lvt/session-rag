from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def append_json_line(path: Path, data: dict) -> None:
    """Append one JSON object as a line — for durable, append-only logs
    (e.g. Retrieval Traces) where atomic whole-file replacement isn't the
    right shape."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(data) + "\n")


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2)
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise
