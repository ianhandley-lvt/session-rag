from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

Runner = Callable[..., subprocess.CompletedProcess[str]]


def run_cursor_json(
    prompt: str,
    *,
    executable: str,
    runner: Runner,
    workspace: Path,
    mode: str,
    model: str,
    timeout: int,
) -> dict:
    """One shared Cursor CLI boundary returning its JSON result payload."""

    completed = runner(
        [executable, "--print", "--output-format", "json", "--mode", mode, "--model", model,
         "--sandbox", "enabled", "--workspace", str(workspace), "--trust"],
        input=prompt, text=True, capture_output=True, check=True, timeout=timeout,
    )
    envelope = json.loads(completed.stdout)
    if not isinstance(envelope, dict):
        raise ValueError("Cursor envelope must be a JSON object")
    if envelope.get("type") != "result" or envelope.get("subtype") != "success":
        raise ValueError(f"Cursor did not return a successful result: {envelope.get('subtype')!r}")
    result_text = envelope.get("result")
    if not isinstance(result_text, str):
        raise ValueError("Cursor result must be text")
    cleaned = result_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Cursor result must be a JSON object")
    return value
