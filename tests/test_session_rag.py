import json
from pathlib import Path

from session_rag.cli import run
from session_rag.hook import handle_user_prompt


class KeywordEmbedder:
    dimensions = 2
    model_name = "keyword-test"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [float("rabbitmq" in text.lower()), float("postgres" in text.lower())]
            for text in texts
        ]


def write_session(path: Path) -> None:
    records = [
        {
            "type": "user",
            "sessionId": "session-123",
            "timestamp": "2026-08-26T10:00:00Z",
            "message": {"role": "user", "content": "Why did RabbitMQ reconnect?"},
        },
        {
            "type": "assistant",
            "sessionId": "session-123",
            "timestamp": "2026-08-26T10:01:00Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "The heartbeat timeout caused the reconnect."},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "secret"}},
                ],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_cli_ingests_sessions_and_returns_cited_search_results(tmp_path, capsys):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    write_session(sessions / "session-123.jsonl")
    database = tmp_path / "memory.lance"
    embedder = KeywordEmbedder()

    assert run(["ingest-sessions", str(sessions), "--database", str(database)], embedder) == 0
    capsys.readouterr()
    assert run(["search", "rabbitmq timeout", "--database", str(database)], embedder) == 0

    output = capsys.readouterr().out
    assert "heartbeat timeout caused the reconnect" in output
    assert "session-123.jsonl" in output
    assert "secret" not in output


def test_user_prompt_hook_returns_additional_context(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    write_session(sessions / "session-123.jsonl")
    database = tmp_path / "memory.lance"
    embedder = KeywordEmbedder()
    run(["ingest-sessions", str(sessions), "--database", str(database)], embedder)

    response = handle_user_prompt(
        {"hook_event_name": "UserPromptSubmit", "prompt": "What caused RabbitMQ to reconnect?"},
        database,
        embedder,
    )

    assert response["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    context = response["hookSpecificOutput"]["additionalContext"]
    assert "Retrieved local session memory" in context
    assert "heartbeat timeout" in context
    assert "session-123.jsonl" in context


def test_cli_extract_session_reports_blocked_for_oversized_session(tmp_path, capsys, monkeypatch):
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(json.dumps({"type": "user", "message": {"content": "a" * 1000}}) + "\n")
    monkeypatch.setenv("SESSION_RAG_MAX_SANITIZED_CHARS", "50")

    exit_code = run(["extract-session", str(transcript)])

    assert exit_code == 2
    assert "blocked:" in capsys.readouterr().err


def test_hook_fails_open_when_database_does_not_exist(tmp_path):
    response = handle_user_prompt(
        {"hook_event_name": "UserPromptSubmit", "prompt": "Anything"},
        tmp_path / "missing.lance",
        KeywordEmbedder(),
    )

    assert response == {}
