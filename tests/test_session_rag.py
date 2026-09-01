import json
from pathlib import Path

from session_rag.artifacts import job_status_path, read_active_hash
from session_rag.cli import run
from session_rag.extractors.base import ExtractionBlocked, ExtractionError, ExtractionPendingRetry
from session_rag.hook import handle_user_prompt

from conftest import make_record


class FakeExtractor:
    name = "fake"
    model = "fake-model"
    prompt_version = 1

    def __init__(self, records=None, error=None):
        self._records = records
        self._error = error
        self.calls = 0

    def extract(self, transcript):
        self.calls += 1
        if self._error:
            raise self._error
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
    written = [
        path
        for path in (artifacts_dir / "claude_session" / "session-123").glob("*.json")
        if path.name not in {"active.json", "job_status.json"}
    ]
    assert len(written) == 1


def test_cli_extract_session_activates_new_revision_on_changed_content(tmp_path):
    transcript = tmp_path / "session-123.jsonl"
    transcript.write_text(json.dumps({"type": "user", "message": {"content": "v1"}}) + "\n")
    artifacts_dir = tmp_path / "artifacts"
    record_v1 = make_record(question="Q1", source=str(transcript.resolve()), source_session_id="session-123")
    run(["extract-session", str(transcript), "--artifacts", str(artifacts_dir)], extractor=FakeExtractor([record_v1]))
    first_active = read_active_hash(artifacts_dir, source_type="claude_session", source_id="session-123")
    first_artifacts = list((artifacts_dir / "claude_session" / "session-123").glob("sha256-*.json"))
    assert len(first_artifacts) == 1

    transcript.write_text(json.dumps({"type": "user", "message": {"content": "v2"}}) + "\n")
    record_v2 = make_record(question="Q2", source=str(transcript.resolve()), source_session_id="session-123")
    exit_code = run(
        ["extract-session", str(transcript), "--artifacts", str(artifacts_dir)], extractor=FakeExtractor([record_v2])
    )

    second_active = read_active_hash(artifacts_dir, source_type="claude_session", source_id="session-123")
    second_artifacts = list((artifacts_dir / "claude_session" / "session-123").glob("sha256-*.json"))

    assert exit_code == 0
    assert second_active != first_active
    # Prior revision's artifact file is untouched, not replaced.
    assert len(second_artifacts) == 2
    assert first_artifacts[0].exists()
    assert first_artifacts[0].read_text()  # still readable, still there


def test_cli_extract_session_pending_retry_leaves_no_partial_artifact_or_active_change(tmp_path, capsys):
    transcript = tmp_path / "session-123.jsonl"
    transcript.write_text(json.dumps({"type": "user", "message": {"content": "hi"}}) + "\n")
    artifacts_dir = tmp_path / "artifacts"

    exit_code = run(
        ["extract-session", str(transcript), "--artifacts", str(artifacts_dir)],
        extractor=FakeExtractor(error=ExtractionPendingRetry("Cursor timed out")),
    )

    assert exit_code == 4
    assert "pending_retry: Cursor timed out" in capsys.readouterr().err
    assert read_active_hash(artifacts_dir, source_type="claude_session", source_id="session-123") is None
    session_dir = artifacts_dir / "claude_session" / "session-123"
    assert list(session_dir.glob("sha256-*.json")) == []
    status = json.loads(job_status_path(artifacts_dir, source_type="claude_session", source_id="session-123").read_text())
    assert status["status"] == "pending_retry"
    assert status["reason"] == "Cursor timed out"
    assert status["attempted_hash"]


def test_cli_extract_session_blocked_writes_job_status_and_leaves_active_unchanged(tmp_path, capsys):
    transcript = tmp_path / "session-123.jsonl"
    transcript.write_text(json.dumps({"type": "user", "message": {"content": "hi"}}) + "\n")
    artifacts_dir = tmp_path / "artifacts"

    exit_code = run(
        ["extract-session", str(transcript), "--artifacts", str(artifacts_dir)],
        extractor=FakeExtractor(error=ExtractionBlocked("session too large")),
    )

    assert exit_code == 2
    assert "blocked: session too large" in capsys.readouterr().err
    status = json.loads(job_status_path(artifacts_dir, source_type="claude_session", source_id="session-123").read_text())
    assert status["status"] == "blocked"
    assert read_active_hash(artifacts_dir, source_type="claude_session", source_id="session-123") is None


def test_cli_extract_session_failed_writes_job_status_and_leaves_active_unchanged(tmp_path, capsys):
    transcript = tmp_path / "session-123.jsonl"
    transcript.write_text(json.dumps({"type": "user", "message": {"content": "hi"}}) + "\n")
    artifacts_dir = tmp_path / "artifacts"

    exit_code = run(
        ["extract-session", str(transcript), "--artifacts", str(artifacts_dir)],
        extractor=FakeExtractor(error=ExtractionError("invalid model output")),
    )

    assert exit_code == 1
    assert "failed: invalid model output" in capsys.readouterr().err
    status = json.loads(job_status_path(artifacts_dir, source_type="claude_session", source_id="session-123").read_text())
    assert status["status"] == "failed"
    assert status["reason"] == "invalid model output"
    assert read_active_hash(artifacts_dir, source_type="claude_session", source_id="session-123") is None
    session_dir = artifacts_dir / "claude_session" / "session-123"
    assert list(session_dir.glob("sha256-*.json")) == []


def test_cli_extract_session_prints_orphaned_diff_on_activation(tmp_path, capsys):
    transcript = tmp_path / "session-123.jsonl"
    transcript.write_text(json.dumps({"type": "user", "message": {"content": "v1"}}) + "\n")
    artifacts_dir = tmp_path / "artifacts"
    record_q1 = make_record(question="Q1", source=str(transcript.resolve()), source_session_id="session-123")
    record_q2 = make_record(question="Q2", source=str(transcript.resolve()), source_session_id="session-123")
    run(
        ["extract-session", str(transcript), "--artifacts", str(artifacts_dir)],
        extractor=FakeExtractor([record_q1, record_q2]),
    )

    transcript.write_text(json.dumps({"type": "user", "message": {"content": "v2"}}) + "\n")
    capsys.readouterr()
    run(
        ["extract-session", str(transcript), "--artifacts", str(artifacts_dir)],
        extractor=FakeExtractor([record_q1]),
    )

    stderr = capsys.readouterr().err
    assert "1 record(s) from the previous revision" in stderr
    assert "Q2" in stderr
    assert "Q1" not in stderr


def test_hook_fails_open_when_database_does_not_exist(tmp_path):
    response = handle_user_prompt(
        {"hook_event_name": "UserPromptSubmit", "prompt": "Anything"},
        tmp_path / "missing.lance",
        KeywordEmbedder(),
    )

    assert response == {}
