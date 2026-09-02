from __future__ import annotations

import json
import re
from dataclasses import dataclass
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


@dataclass(frozen=True)
class SanitizedEntry:
    """One turn of a sanitized session, addressable by a stable identifier
    that is independent of its position in the rendered text — a turn's own
    rendered text can itself span multiple lines (a multi-block message), and
    turns with no renderable content are skipped without shifting any other
    entry's identifier (see sanitize_session)."""

    identifier: str
    text: str


@dataclass(frozen=True)
class SanitizedSession:
    """Sanitized, provider-safe rendering of one source revision, plus the
    deterministic entry-identifier mapping Evidence Location needs (see
    extractors.base.EvidenceLocation) — sanitize_session is the only place
    that ever has both the raw transcript's true per-turn identity and the
    provider-safe text, so it's the only place that mapping can be built."""

    entries: list[SanitizedEntry]

    @property
    def prompt_text(self) -> str:
        """What's sent to the extraction provider — each entry prefixed with
        its stable identifier in brackets, so the model can copy one rather
        than counting or guessing a position."""

        return "\n".join(f"[{entry.identifier}] {entry.text}" for entry in self.entries)

    def text_for(self, identifier: str) -> str | None:
        """None if identifier doesn't name a real entry in this exact
        revision's sanitized rendering — the only membership test Evidence
        Location validation needs, and the only way a preserved snippet is
        ever looked up."""

        for entry in self.entries:
            if entry.identifier == identifier:
                return entry.text
        return None


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


def _entry_identifier(record: dict, raw_index: int) -> str:
    """Prefer the transcript's own stable per-turn identifier (its `uuid`,
    when the source provides one) over a fallback derived from raw file
    position — a source-provided uuid survives edits to the file itself,
    unlike a position. `raw_index` is the turn's 0-indexed position among
    *all* raw JSONL lines (computed before any skip-filtering), so a
    fallback identifier never shifts when an earlier record turns out to be
    unparsable or renders no content."""

    uuid_value = record.get("uuid")
    if isinstance(uuid_value, str) and uuid_value:
        return uuid_value
    return f"line-{raw_index}"


def sanitize_session(
    path: Path,
    *,
    sensitive_paths: tuple[str, ...] = (),
    max_chars: int = DEFAULT_MAX_SANITIZED_CHARS,
    tool_output_limit: int = MAX_TOOL_OUTPUT_CHARS,
) -> SanitizedSession:
    """Render a session transcript into a SanitizedSession: sanitized,
    provider-safe text plus the stable identifier -> text mapping Evidence
    Location resolution needs.

    Preserves user, assistant, tool, and subagent evidence (tool/subagent
    content is never dropped, unlike the raw turn parser), replaces oversized
    tool output/diffs with bounded metadata-only markers, and redacts
    secret-shaped values everywhere. Raises SanitizationBudgetExceeded rather
    than silently truncating when the result is still too large.
    """

    entries: list[SanitizedEntry] = []
    tool_uses: dict[str, dict] = {}
    for raw_index, raw_line in enumerate(path.read_text(errors="replace").splitlines()):
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
        text = redact_secrets(f"{_speaker(record)}: {rendered_text}", sensitive_paths=sensitive_paths)
        entries.append(SanitizedEntry(identifier=_entry_identifier(record, raw_index), text=text))

    sanitized = SanitizedSession(entries=entries)
    if len(sanitized.prompt_text) > max_chars:
        raise SanitizationBudgetExceeded(len(sanitized.prompt_text), max_chars)
    return sanitized
