import json
from pathlib import Path

from session_rag.cli import run
from session_rag.hook import handle_user_prompt

from conftest import make_record


class FakeExtractor:
    name = "fake"
    model = "fake-model"
    prompt_version = 1

    def __init__(self, records):
        self._records = records
        self.calls = 0

    def extract(self, transcript):
        self.calls += 1
        return self._records


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
    monkeypatch.setenv("SESSION_RAG_OPERATOR_ID", "test-operator")

    exit_code = run(["extract-session", str(transcript), "--artifacts", str(tmp_path / "artifacts")])

    assert exit_code == 2
    assert "blocked:" in capsys.readouterr().err


def test_cli_extract_session_reports_configuration_error_without_operator_id(tmp_path, capsys, monkeypatch):
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(json.dumps({"type": "user", "message": {"content": "hi"}}) + "\n")
    monkeypatch.delenv("SESSION_RAG_OPERATOR_ID", raising=False)

    exit_code = run(["extract-session", str(transcript), "--artifacts", str(tmp_path / "artifacts")])

    assert exit_code == 3
    assert "configuration error" in capsys.readouterr().err


def test_cli_extract_session_writes_extraction_artifact(tmp_path, capsys):
    transcript = tmp_path / "session-123.jsonl"
    transcript.write_text(json.dumps({"type": "user", "message": {"content": "Why did it break?"}}) + "\n")
    artifacts_dir = tmp_path / "artifacts"
    record = make_record(
        question="Why did it break?",
        summary="Heartbeat expired.",
        source=str(transcript.resolve()),
        source_session_id="session-123",
    )

    exit_code = run(
        ["extract-session", str(transcript), "--artifacts", str(artifacts_dir)],
        extractor=FakeExtractor([record]),
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    artifact_path = Path(output["artifact_path"])
    assert artifact_path.exists()
    assert artifact_path.is_relative_to(artifacts_dir)
    envelope = json.loads(artifact_path.read_text())
    assert envelope["source_id"] == "session-123"
    assert envelope["extractor"] == "fake"
    assert envelope["extractor_model"] == "fake-model"
    assert envelope["episode_records"][0]["question"] == "Why did it break?"


def test_cli_extract_session_skips_extraction_when_artifact_already_exists(tmp_path):
    transcript = tmp_path / "session-123.jsonl"
    transcript.write_text(json.dumps({"type": "user", "message": {"content": "unchanged"}}) + "\n")
    artifacts_dir = tmp_path / "artifacts"
    record = make_record(source=str(transcript.resolve()), source_session_id="session-123")
    first_extractor = FakeExtractor([record])
    second_extractor = FakeExtractor([record])

    run(["extract-session", str(transcript), "--artifacts", str(artifacts_dir)], extractor=first_extractor)
    exit_code = run(
        ["extract-session", str(transcript), "--artifacts", str(artifacts_dir)], extractor=second_extractor
    )

    assert exit_code == 0
    assert first_extractor.calls == 1
    # Same source hash already has an artifact — no reason to pay for extraction again.
    assert second_extractor.calls == 0
    written = list((artifacts_dir / "claude_session" / "session-123").glob("*.json"))
    assert len(written) == 1


def test_hook_fails_open_when_database_does_not_exist(tmp_path):
    response = handle_user_prompt(
        {"hook_event_name": "UserPromptSubmit", "prompt": "Anything"},
        tmp_path / "missing.lance",
        KeywordEmbedder(),
    )

    assert response == {}
