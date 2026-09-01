from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_MAX_SANITIZED_CHARS = 20_000
MAX_TOOL_OUTPUT_CHARS = 500

REDACTED = "[REDACTED]"

# Best-effort, pattern-based secret detection. This is NOT a guarantee that no
# secret can escape sanitization — it catches common credential shapes only.
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # OpenAI-style keys
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),  # GitHub tokens
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key ids
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),  # Google API keys
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),  # Slack tokens
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),  # JWTs
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
    re.compile(r"(?i)(?:api[_-]?key|secret|token|password)['\"]?\s*[:=]\s*['\"]([^'\"\s]{8,})['\"]"),
]


class SanitizationBudgetExceeded(RuntimeError):
    """A sanitized session exceeded the configured size limit."""

    def __init__(self, size: int, limit: int) -> None:
        self.size = size
        self.limit = limit
        super().__init__(
            f"Sanitized session is {size} characters, exceeding the {limit}-character limit. "
            "Extraction was not attempted rather than silently truncated; reduce the session size "
            "or raise the configured maximum."
        )


def redact_secrets(text: str, *, sensitive_paths: tuple[str, ...] = ()) -> str:
    """Best-effort redaction. Does not guarantee no secret can escape.

    `sensitive_paths` are matched by exact substring — a path referenced with
    different formatting (relative vs. absolute, trailing slash, a symlink)
    will not match. Configure the exact string(s) that actually appear in
    your transcripts.
    """

    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(REDACTED, redacted)
    for path in sensitive_paths:
        if path:
            redacted = redacted.replace(path, REDACTED)
    return redacted


def _speaker(record: dict) -> str:
    if record.get("isSidechain"):
        return "Subagent"
    record_type = record.get("type")
    if record_type == "user":
        return "User"
    if record_type == "assistant":
        return "Assistant"
    return record_type or "Unknown"


def _content_blocks(content: object) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return []


_DIFF_KEYS = ("old_string", "new_string", "diff", "patch")


def _tool_use_info(block: dict) -> dict:
    name = block.get("name", "tool")
    input_value = block.get("input")
    input_value = input_value if isinstance(input_value, dict) else {}
    detail = (
        input_value.get("command")
        or input_value.get("file_path")
        or input_value.get("path")
        or input_value.get("pattern")
    )
    diff_size = sum(len(input_value[key]) for key in _DIFF_KEYS if isinstance(input_value.get(key), str))
    return {"name": name, "detail": detail, "diff_size": diff_size or None}


def _tool_use_label(info: dict) -> str:
    label = f"Tool[{info['name']}] $ {info['detail']}" if info["detail"] else f"Tool[{info['name']}]"
    if info["diff_size"]:
        label += f" (diff omitted, {info['diff_size']} chars)"
    return label


def _tool_result_text(block: dict) -> str:
    raw = block.get("content", "")
    if isinstance(raw, list):
        raw = "\n".join(
            item.get("text", "") for item in raw if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(raw)


def _tool_result_marker(block: dict, *, limit: int, tool_uses: dict[str, dict]) -> str:
    text = _tool_result_text(block)
    status = "error" if block.get("is_error") else "ok"
    use = tool_uses.get(block.get("tool_use_id"))
    label = _tool_use_label(use) if use else "Tool result"
    if len(text) <= limit:
        return f"{label} ({status}): {text}"
    return f"{label} [output omitted: {status}, {len(text)} chars]"


def _render_block(block: dict, *, tool_output_limit: int, tool_uses: dict[str, dict]) -> str | None:
    block_type = block.get("type")
    if block_type == "text":
        text = block.get("text", "").strip()
        return text or None
    if block_type == "tool_use":
        info = _tool_use_info(block)
        tool_use_id = block.get("id")
        if tool_use_id:
            tool_uses[tool_use_id] = info
        return _tool_use_label(info)
    if block_type == "tool_result":
        return _tool_result_marker(block, limit=tool_output_limit, tool_uses=tool_uses)
    return None


def sanitize_session(
    path: Path,
    *,
    sensitive_paths: tuple[str, ...] = (),
    max_chars: int = DEFAULT_MAX_SANITIZED_CHARS,
    tool_output_limit: int = MAX_TOOL_OUTPUT_CHARS,
) -> str:
    """Render a session transcript into sanitized, provider-safe text.

    Preserves user, assistant, tool, and subagent evidence (tool/subagent
    content is never dropped, unlike the raw turn parser), replaces oversized
    tool output/diffs with bounded metadata-only markers, and redacts
    secret-shaped values everywhere. Raises SanitizationBudgetExceeded rather
    than silently truncating when the result is still too large.
    """

    lines: list[str] = []
    tool_uses: dict[str, dict] = {}
    for raw_line in path.read_text(errors="replace").splitlines():
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        rendered = [
            _render_block(block, tool_output_limit=tool_output_limit, tool_uses=tool_uses)
            for block in _content_blocks(message.get("content"))
        ]
        rendered_text = "\n".join(part for part in rendered if part)
        if not rendered_text:
            continue
        lines.append(f"{_speaker(record)}: {rendered_text}")

    sanitized = redact_secrets("\n".join(lines), sensitive_paths=sensitive_paths)
    if len(sanitized) > max_chars:
        raise SanitizationBudgetExceeded(len(sanitized), max_chars)
    return sanitized
