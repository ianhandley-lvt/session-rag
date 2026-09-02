import json
from pathlib import Path

import pytest

from session_rag.sanitize import SanitizationBudgetExceeded, redact_secrets, sanitize_session


@pytest.mark.parametrize(
    "secret",
    [
        "sk-abcdefghijklmnopqrstuvwx",
        "ghp_abcdefghijklmnopqrstuvwxyz1234",
        "AKIAABCDEFGHIJKLMNOP",
        "AIzaSyAbcdefghijklmnopqrstuvwxyz01234",
        "xoxb-1234567890-abcdefghij",
    ],
)
def test_redact_secrets_removes_known_credential_shapes(secret):
    text = f"the token is {secret} — keep this"
    redacted = redact_secrets(text)

    assert secret not in redacted
    assert "[REDACTED]" in redacted
    assert "keep this" in redacted


def test_redact_secrets_removes_key_value_assignments():
    redacted = redact_secrets('config: api_key: "abcdefgh12345678"')

    assert "abcdefgh12345678" not in redacted
    assert "[REDACTED]" in redacted


def test_redact_secrets_removes_configured_sensitive_paths():
    redacted = redact_secrets(
        "reading /Users/ian/secret-project/creds.txt now",
        sensitive_paths=("/Users/ian/secret-project/creds.txt",),
    )

    assert "/Users/ian/secret-project/creds.txt" not in redacted
    assert "[REDACTED]" in redacted


def write_lines(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_sanitize_session_preserves_user_and_assistant_text(tmp_path):
    transcript = tmp_path / "session.jsonl"
    write_lines(
        transcript,
        [
            {"type": "user", "message": {"content": "Why did it break?"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Heartbeat timeout."}]}},
        ],
    )

    sanitized = sanitize_session(transcript)

    assert "User: Why did it break?" in sanitized.prompt_text
    assert "Assistant: Heartbeat timeout." in sanitized.prompt_text


def test_sanitize_session_preserves_subagent_messages(tmp_path):
    transcript = tmp_path / "session.jsonl"
    write_lines(
        transcript,
        [{"type": "assistant", "isSidechain": True, "message": {"content": "Delegated investigation result."}}],
    )

    sanitized = sanitize_session(transcript)

    assert "Subagent: Delegated investigation result." in sanitized.prompt_text


def test_sanitize_session_preserves_small_tool_output(tmp_path):
    transcript = tmp_path / "session.jsonl"
    write_lines(
        transcript,
        [
            {
                "type": "user",
                "message": {
                    "content": [{"type": "tool_result", "content": "exit 0: ok", "is_error": False}]
                },
            }
        ],
    )

    sanitized = sanitize_session(transcript)

    assert "Tool result (ok): exit 0: ok" in sanitized.prompt_text


def test_sanitize_session_replaces_oversized_tool_output_with_marker(tmp_path):
    transcript = tmp_path / "session.jsonl"
    huge_output = "x" * 5_000
    write_lines(
        transcript,
        [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "id": "call-1", "name": "Bash", "input": {"command": "cat big.log"}}
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call-1", "content": huge_output, "is_error": False}
                    ]
                },
            },
        ],
    )

    sanitized = sanitize_session(transcript, tool_output_limit=500)

    assert huge_output not in sanitized.prompt_text
    # The correlated result marker carries tool name, command, and status together.
    assert "Tool[Bash] $ cat big.log [output omitted: ok, 5000 chars]" in sanitized.prompt_text


def test_sanitize_session_replaces_oversized_diff_with_bounded_marker(tmp_path):
    transcript = tmp_path / "session.jsonl"
    old_body = "a" * 2_000
    new_body = "b" * 2_000
    write_lines(
        transcript,
        [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call-2",
                            "name": "Edit",
                            "input": {"file_path": "/repo/src/app.py", "old_string": old_body, "new_string": new_body},
                        }
                    ]
                },
            }
        ],
    )

    sanitized = sanitize_session(transcript)

    assert old_body not in sanitized.prompt_text
    assert new_body not in sanitized.prompt_text
    assert "Tool[Edit] $ /repo/src/app.py" in sanitized.prompt_text
    assert "(diff omitted, 4000 chars)" in sanitized.prompt_text


def test_sanitize_session_redacts_secrets_in_commands_and_tool_output(tmp_path):
    transcript = tmp_path / "session.jsonl"
    write_lines(
        transcript,
        [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Bash", "input": {"command": "curl -H sk-abcdefghijklmnopqrstuvwx"}}
                    ]
                },
            }
        ],
    )

    sanitized = sanitize_session(transcript)

    assert "sk-abcdefghijklmnopqrstuvwx" not in sanitized.prompt_text
    assert "[REDACTED]" in sanitized.prompt_text


def test_sanitize_session_raises_when_over_budget(tmp_path):
    transcript = tmp_path / "session.jsonl"
    write_lines(transcript, [{"type": "user", "message": {"content": "a" * 1000}}])

    with pytest.raises(SanitizationBudgetExceeded) as excinfo:
        sanitize_session(transcript, max_chars=50)

    assert "50" in str(excinfo.value)


def test_sanitize_session_assigns_a_fallback_identifier_per_raw_record(tmp_path):
    transcript = tmp_path / "session.jsonl"
    write_lines(
        transcript,
        [
            {"type": "user", "message": {"content": "first"}},
            {"type": "assistant", "message": {"content": "second"}},
        ],
    )

    sanitized = sanitize_session(transcript)

    assert [entry.identifier for entry in sanitized.entries] == ["line-0", "line-1"]
    assert sanitized.text_for("line-0") == "User: first"
    assert sanitized.text_for("line-1") == "Assistant: second"


def test_sanitize_session_uses_the_records_own_uuid_when_present(tmp_path):
    # A source-provided stable identifier is preferred over a position-based
    # fallback — it survives edits to the file itself, unlike a raw index.
    transcript = tmp_path / "session.jsonl"
    write_lines(
        transcript,
        [{"type": "user", "uuid": "turn-abc-123", "message": {"content": "hello"}}],
    )

    sanitized = sanitize_session(transcript)

    assert sanitized.entries[0].identifier == "turn-abc-123"
    assert sanitized.text_for("turn-abc-123") == "User: hello"
    assert sanitized.text_for("line-0") is None


def test_sanitize_session_multiline_entry_keeps_a_single_identifier(tmp_path):
    # One raw record with multiple content blocks renders as text containing
    # embedded newlines — it must still be exactly one entry/identifier, not
    # one identifier per rendered line (that would be the old, broken
    # sanitized-line-number scheme this replaces).
    transcript = tmp_path / "session.jsonl"
    write_lines(
        transcript,
        [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "first line"},
                        {"type": "text", "text": "second line"},
                        {"type": "text", "text": "third line"},
                    ]
                },
            }
        ],
    )

    sanitized = sanitize_session(transcript)

    assert len(sanitized.entries) == 1
    entry = sanitized.entries[0]
    assert "first line" in entry.text
    assert "second line" in entry.text
    assert "third line" in entry.text
    assert "\n" in entry.text  # genuinely multiline, and still one identifier


def test_sanitize_session_skipped_records_do_not_shift_later_identifiers(tmp_path):
    # A blank/no-content record and a malformed (non-JSON) record are both
    # skipped — the identifiers of records after them must still reflect
    # their true raw position, not a renumbered position in the filtered
    # stream.
    transcript = tmp_path / "session.jsonl"
    lines = [
        json.dumps({"type": "user", "message": {"content": "first"}}),  # raw index 0
        "not valid json",  # raw index 1 — skipped (malformed)
        json.dumps({"type": "assistant", "message": {"content": []}}),  # raw index 2 — skipped (no content)
        json.dumps({"type": "user", "message": {"content": "fourth"}}),  # raw index 3
    ]
    transcript.write_text("\n".join(lines) + "\n")

    sanitized = sanitize_session(transcript)

    assert [entry.identifier for entry in sanitized.entries] == ["line-0", "line-3"]
    assert sanitized.text_for("line-3") == "User: fourth"
