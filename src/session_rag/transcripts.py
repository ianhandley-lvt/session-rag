from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SessionMemory:
    id: str
    session_id: str
    text: str
    source: str
    timestamp: str


def _message_text(record: dict) -> str:
    content = record.get("message", {}).get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "").strip()
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
        ).strip()
    return ""


def parse_session(path: Path) -> list[SessionMemory]:
    turns: list[SessionMemory] = []
    pending_user: tuple[str, str, str] | None = None
    for line in path.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        record_type = record.get("type")
        text = _message_text(record)
        if not text:
            continue
        session_id = record.get("sessionId", path.stem)
        timestamp = record.get("timestamp", "")
        if record_type == "user":
            pending_user = (text, session_id, timestamp)
        elif record_type == "assistant" and pending_user:
            user_text, session_id, user_timestamp = pending_user
            memory_id = f"{session_id}:{len(turns)}"
            turns.append(
                SessionMemory(
                    id=memory_id,
                    session_id=session_id,
                    text=f"User: {user_text}\nAssistant: {text}",
                    source=str(path),
                    timestamp=timestamp or user_timestamp,
                )
            )
            pending_user = None
    return turns


def load_sessions(directory: Path) -> list[SessionMemory]:
    memories: list[SessionMemory] = []
    for path in sorted(directory.glob("*.jsonl")):
        memories.extend(parse_session(path))
    return memories

