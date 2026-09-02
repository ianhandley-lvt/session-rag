import json
import subprocess
from pathlib import Path

import pytest
from session_rag.extractors.cursor import CursorExtractor
from session_rag.extractors.base import ExtractionBlocked, ExtractionError, ExtractionPendingRetry, ProjectProvenance


@pytest.fixture(autouse=True)
def operator_id_env(monkeypatch):
    monkeypatch.setenv("SESSION_RAG_OPERATOR_ID", "test-operator")
    # Deterministic regardless of the host shell's own environment.
    for var in ("SESSION_RAG_PROJECT_ID", "SESSION_RAG_PROJECT_ROOT", "SESSION_RAG_REPOSITORY_REVISION", "SESSION_RAG_WORKING_TREE_DIRTY"):
        monkeypatch.delenv(var, raising=False)


def cursor_response(payload: dict) -> subprocess.CompletedProcess[str]:
    envelope = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": json.dumps(payload),
        "session_id": "cursor-session",
    }
    return subprocess.CompletedProcess([], 0, stdout=json.dumps(envelope), stderr="")


def write_transcript(path: Path) -> None:
    path.write_text('{"type":"user","message":{"content":"Why did RabbitMQ reconnect?"}}\n')


def test_cursor_extractor_returns_validated_structured_records(tmp_path):
    transcript = tmp_path / "claude-session-123.jsonl"
    write_transcript(transcript)
    observed: dict = {}

    def runner(command, **kwargs):
        observed["command"] = command
        observed["input"] = kwargs["input"]
        return cursor_response(
            {
                "records": [
                    {
                        "question": "Why did RabbitMQ reconnect?",
                        "summary": "The heartbeat expired.",
                        "resolution": "Increase the heartbeat interval.",
                        "systems": ["RabbitMQ"],
                        "code_references": ["RabbitConnectionManager"],
                        "attribution": {"person": "Ian", "citation": "message-1"},
                        "temporal_scope": "durable",
                        "timestamp": "2026-08-27T10:00:00Z",
                    }
                ]
            }
        )

    records = CursorExtractor(runner=runner).extract(transcript)

    assert records[0].resolution == "Increase the heartbeat interval."
    assert records[0].systems == ["RabbitMQ"]
    assert records[0].source == str(transcript.resolve())
    assert records[0].source_session_id == transcript.stem
    assert records[0].attribution.person == "Ian"
    assert records[0].temporal_scope == "durable"
    assert records[0].source_type == "claude_session"
    assert records[0].operator_id == "test-operator"
    assert records[0].prompt_version == 1
    assert ["--mode", "ask"] == observed["command"][4:6]
    assert "--sandbox" in observed["command"]
    assert "--workspace" in observed["command"]
    assert "--trust" in observed["command"]
    assert "Why did RabbitMQ reconnect?" in observed["input"]
    assert str(transcript) not in observed["input"]


def test_cursor_extractor_raises_pending_retry_on_timeout(tmp_path):
    transcript = tmp_path / "session.jsonl"
    write_transcript(transcript)

    def runner(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="cursor-agent", timeout=120)

    with pytest.raises(ExtractionPendingRetry):
        CursorExtractor(runner=runner).extract(transcript)


def test_cursor_extractor_raises_pending_retry_on_nonzero_exit(tmp_path):
    transcript = tmp_path / "session.jsonl"
    write_transcript(transcript)

    def runner(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd="cursor-agent")

    with pytest.raises(ExtractionPendingRetry):
        CursorExtractor(runner=runner).extract(transcript)


def test_cursor_extractor_raises_pending_retry_when_cursor_unavailable(tmp_path):
    transcript = tmp_path / "session.jsonl"
    write_transcript(transcript)

    def runner(*args, **kwargs):
        raise FileNotFoundError("cursor-agent not found")

    with pytest.raises(ExtractionPendingRetry):
        CursorExtractor(runner=runner).extract(transcript)


def test_cursor_extractor_raises_failed_on_reported_non_success_envelope(tmp_path):
    # A non-success envelope isn't assumed to be infra-level (no evidence for
    # what subtypes Cursor reports for a bad request vs. real unavailability)
    # — it's retried like malformed output, then lands in `failed`, not
    # `pending_retry`. Only genuine subprocess-level failures are pending_retry.
    transcript = tmp_path / "session.jsonl"
    write_transcript(transcript)

    def runner(*args, **kwargs):
        envelope = {"type": "result", "subtype": "error", "result": "quota exceeded"}
        return subprocess.CompletedProcess([], 0, stdout=json.dumps(envelope), stderr="")

    with pytest.raises(ExtractionError) as excinfo:
        CursorExtractor(runner=runner, max_output_retries=0).extract(transcript)

    assert not isinstance(excinfo.value, ExtractionPendingRetry)


def test_cursor_extractor_retries_invalid_output_before_failing(tmp_path):
    transcript = tmp_path / "session.jsonl"
    write_transcript(transcript)
    calls = {"count": 0}

    def runner(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return subprocess.CompletedProcess([], 0, stdout="not json", stderr="")
        return cursor_response({"records": [{"question": "Q", "summary": "S"}]})

    records = CursorExtractor(runner=runner, max_output_retries=1).extract(transcript)

    assert calls["count"] == 2
    assert records[0].question == "Q"


def test_cursor_extractor_raises_failed_after_exhausting_retries(tmp_path):
    transcript = tmp_path / "session.jsonl"
    write_transcript(transcript)

    def runner(*args, **kwargs):
        return subprocess.CompletedProcess([], 0, stdout="not json", stderr="")

    with pytest.raises(ExtractionError) as excinfo:
        CursorExtractor(runner=runner, max_output_retries=1).extract(transcript)

    assert not isinstance(excinfo.value, ExtractionPendingRetry)
    assert not isinstance(excinfo.value, ExtractionBlocked)


def test_cursor_extractor_rejects_records_outside_the_schema(tmp_path):
    transcript = tmp_path / "session.jsonl"
    write_transcript(transcript)

    def runner(*args, **kwargs):
        return cursor_response({"records": [{"summary": "Missing required fields"}]})

    with pytest.raises(ExtractionError):
        CursorExtractor(runner=runner).extract(transcript)


def test_cursor_extractor_does_not_trust_model_provenance(tmp_path):
    transcript = tmp_path / "real-session.jsonl"
    write_transcript(transcript)

    def runner(*args, **kwargs):
        return cursor_response(
            {
                "records": [
                    {
                        "question": "What happened?",
                        "summary": "A reconnect occurred.",
                        "resolution": None,
                        "systems": [],
                        "code_references": [],
                        "timestamp": None,
                        "source": "/forged/path",
                    }
                ]
            }
        )

    with pytest.raises(ExtractionError):
        CursorExtractor(runner=runner).extract(transcript)


def test_cursor_extractor_rejects_forged_trusted_provenance(tmp_path):
    transcript = tmp_path / "real-session.jsonl"
    write_transcript(transcript)

    def runner(*args, **kwargs):
        return cursor_response(
            {
                "records": [
                    {
                        "question": "What happened?",
                        "summary": "A reconnect occurred.",
                        "resolution": None,
                        "systems": [],
                        "code_references": [],
                        "timestamp": None,
                        "operator_id": "forged-operator",
                    }
                ]
            }
        )

    with pytest.raises(ExtractionError):
        CursorExtractor(runner=runner).extract(transcript)


def test_cursor_extractor_reads_mode_and_model_from_environment(tmp_path, monkeypatch):
    transcript = tmp_path / "session.jsonl"
    write_transcript(transcript)
    monkeypatch.setenv("SESSION_RAG_CURSOR_MODE", "plan")
    monkeypatch.setenv("SESSION_RAG_CURSOR_MODEL", "gemini-3.7-flash-low")
    observed: dict = {}

    def runner(command, **kwargs):
        observed["command"] = command
        return cursor_response({"records": []})

    CursorExtractor(runner=runner).extract(transcript)

    assert observed["command"][4:6] == ["--mode", "plan"]
    assert ["--model", "gemini-3.7-flash-low"] == observed["command"][6:8]


def test_cursor_extractor_redacts_secrets_before_sending(tmp_path):
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "message": {"content": "leaked key sk-abcdefghijklmnopqrstuvwx here"}}) + "\n"
    )
    observed: dict = {}

    def runner(command, **kwargs):
        observed["input"] = kwargs["input"]
        return cursor_response({"records": []})

    CursorExtractor(runner=runner).extract(transcript)

    assert "sk-abcdefghijklmnopqrstuvwx" not in observed["input"]
    assert "[REDACTED]" in observed["input"]


def test_cursor_extractor_replaces_oversized_tool_output_with_marker(tmp_path):
    transcript = tmp_path / "session.jsonl"
    huge_output = "y" * 5_000
    lines = [
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "tool_use", "id": "call-1", "name": "Bash", "input": {"command": "cat big.log"}}]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": huge_output, "is_error": False}]
            },
        },
    ]
    transcript.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    observed: dict = {}

    def runner(command, **kwargs):
        observed["input"] = kwargs["input"]
        return cursor_response({"records": []})

    CursorExtractor(runner=runner).extract(transcript)

    assert huge_output not in observed["input"]
    assert "Tool[Bash] $ cat big.log [output omitted: ok, 5000 chars]" in observed["input"]


def test_cursor_extractor_preserves_subagent_messages(tmp_path):
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        json.dumps({"type": "assistant", "isSidechain": True, "message": {"content": "Subagent found the cause."}})
        + "\n"
    )
    observed: dict = {}

    def runner(command, **kwargs):
        observed["input"] = kwargs["input"]
        return cursor_response({"records": []})

    CursorExtractor(runner=runner).extract(transcript)

    assert "Subagent: Subagent found the cause." in observed["input"]


def test_cursor_extractor_honors_configured_sensitive_paths(tmp_path):
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "message": {"content": "check /Users/ian/secret-project/creds.txt"}}) + "\n"
    )
    observed: dict = {}

    def runner(command, **kwargs):
        observed["input"] = kwargs["input"]
        return cursor_response({"records": []})

    CursorExtractor(runner=runner, sensitive_paths=("/Users/ian/secret-project/creds.txt",)).extract(transcript)

    assert "/Users/ian/secret-project/creds.txt" not in observed["input"]


def test_cursor_extractor_rejects_forged_project_id(tmp_path):
    transcript = tmp_path / "real-session.jsonl"
    write_transcript(transcript)

    def runner(*args, **kwargs):
        return cursor_response(
            {
                "records": [
                    {
                        "question": "What happened?",
                        "summary": "A reconnect occurred.",
                        "resolution": None,
                        "systems": [],
                        "code_references": [],
                        "timestamp": None,
                        "project_id": "forged-project",
                    }
                ]
            }
        )

    with pytest.raises(ExtractionError):
        CursorExtractor(runner=runner).extract(transcript)


def test_cursor_extractor_rejects_attribution_masquerading_as_trusted(tmp_path):
    transcript = tmp_path / "real-session.jsonl"
    write_transcript(transcript)

    def runner(*args, **kwargs):
        return cursor_response(
            {
                "records": [
                    {
                        "question": "What happened?",
                        "summary": "A reconnect occurred.",
                        "resolution": None,
                        "systems": [],
                        "code_references": [],
                        "timestamp": None,
                        "attribution": {"person": "Ian", "citation": "message-1", "trusted": True},
                    }
                ]
            }
        )

    with pytest.raises(ExtractionError):
        CursorExtractor(runner=runner).extract(transcript)


def test_cursor_extractor_never_credits_a_non_person_attribution(tmp_path):
    transcript = tmp_path / "session.jsonl"
    write_transcript(transcript)

    def runner(*args, **kwargs):
        return cursor_response(
            {
                "records": [
                    {
                        "question": "What happened?",
                        "summary": "A reconnect occurred.",
                        "resolution": None,
                        "systems": [],
                        "code_references": [],
                        "attribution": {"person": "Subagent", "citation": "message-2"},
                        "timestamp": None,
                    }
                ]
            }
        )

    records = CursorExtractor(runner=runner).extract(transcript)

    assert records[0].attribution is None


def test_cursor_extractor_preserves_a_real_person_attribution(tmp_path):
    transcript = tmp_path / "session.jsonl"
    write_transcript(transcript)

    def runner(*args, **kwargs):
        return cursor_response(
            {
                "records": [
                    {
                        "question": "What happened?",
                        "summary": "A reconnect occurred.",
                        "resolution": None,
                        "systems": [],
                        "code_references": [],
                        "attribution": {"person": "Ian", "citation": "message-2"},
                        "timestamp": None,
                    }
                ]
            }
        )

    records = CursorExtractor(runner=runner).extract(transcript)

    assert records[0].attribution.person == "Ian"
    assert records[0].attribution.citation == "message-2"


def test_cursor_extractor_requires_operator_id_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("SESSION_RAG_OPERATOR_ID", raising=False)

    with pytest.raises(ValueError, match="operator_id"):
        CursorExtractor(runner=lambda *a, **k: None)


def test_cursor_extractor_attaches_project_provenance_without_sending_project_root(tmp_path):
    transcript = tmp_path / "session.jsonl"
    write_transcript(transcript)
    observed: dict = {}

    def runner(command, **kwargs):
        observed["input"] = kwargs["input"]
        return cursor_response({"records": [{"question": "Q", "summary": "S"}]})

    project = ProjectProvenance(
        project_id="session-rag",
        project_root="/Users/ian/src/session-rag",
        repository_revision="abc123",
        working_tree_dirty=True,
    )
    records = CursorExtractor(runner=runner, project=project).extract(transcript)

    assert records[0].project.project_id == "session-rag"
    assert records[0].project.project_root == "/Users/ian/src/session-rag"
    assert records[0].project.repository_revision == "abc123"
    assert records[0].project.working_tree_dirty is True
    assert "/Users/ian/src/session-rag" not in observed["input"]


def test_cursor_extractor_leaves_project_none_without_configured_project(tmp_path):
    transcript = tmp_path / "session.jsonl"
    write_transcript(transcript)

    def runner(*args, **kwargs):
        return cursor_response({"records": [{"question": "Q", "summary": "S"}]})

    records = CursorExtractor(runner=runner).extract(transcript)

    assert records[0].project is None


def test_cursor_extractor_reads_project_provenance_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_RAG_PROJECT_ID", "session-rag")
    monkeypatch.setenv("SESSION_RAG_REPOSITORY_REVISION", "def456")
    monkeypatch.setenv("SESSION_RAG_WORKING_TREE_DIRTY", "true")
    monkeypatch.delenv("SESSION_RAG_PROJECT_ROOT", raising=False)
    transcript = tmp_path / "session.jsonl"
    write_transcript(transcript)

    def runner(*args, **kwargs):
        return cursor_response({"records": [{"question": "Q", "summary": "S"}]})

    records = CursorExtractor(runner=runner).extract(transcript)

    assert records[0].project.project_id == "session-rag"
    assert records[0].project.repository_revision == "def456"
    assert records[0].project.working_tree_dirty is True
    assert records[0].project.project_root is None


def test_cursor_extractor_blocks_oversized_sanitized_session(tmp_path):
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(json.dumps({"type": "user", "message": {"content": "a" * 1000}}) + "\n")

    def runner(*args, **kwargs):
        raise AssertionError("runner should not be invoked when the session is blocked")

    with pytest.raises(ExtractionBlocked) as excinfo:
        CursorExtractor(runner=runner, max_sanitized_chars=50).extract(transcript)

    assert "50" in excinfo.value.reason


def test_explicit_cursor_configuration_overrides_environment(tmp_path, monkeypatch):
    transcript = tmp_path / "session.jsonl"
    write_transcript(transcript)
    monkeypatch.setenv("SESSION_RAG_CURSOR_MODE", "plan")
    monkeypatch.setenv("SESSION_RAG_CURSOR_MODEL", "auto")
    observed: dict = {}

    def runner(command, **kwargs):
        observed["command"] = command
        return cursor_response({"records": []})

    CursorExtractor(
        runner=runner,
        mode="ask",
        model="gemini-3.7-flash-low",
    ).extract(transcript)

    assert observed["command"][4:6] == ["--mode", "ask"]
    assert ["--model", "gemini-3.7-flash-low"] == observed["command"][6:8]


def test_cursor_extractor_no_longer_sets_authority(tmp_path):
    transcript = tmp_path / "session.jsonl"
    write_transcript(transcript)

    def runner(*args, **kwargs):
        return cursor_response({"records": [{"question": "Q", "summary": "S"}]})

    records = CursorExtractor(runner=runner).extract(transcript)

    assert "authority" not in records[0].model_dump()


def test_cursor_extractor_accepts_evidence_location_matching_a_real_entry(tmp_path):
    transcript = tmp_path / "session.jsonl"
    write_transcript(transcript)  # sanitizes to exactly one entry, identifier "line-0"

    def runner(*args, **kwargs):
        return cursor_response({"records": [{"question": "Q", "summary": "S", "evidence_location": "line-0"}]})

    records = CursorExtractor(runner=runner).extract(transcript)

    assert records[0].evidence_location.identifier == "line-0"
    assert records[0].evidence_location.preserved_text == "User: Why did RabbitMQ reconnect?"


def test_cursor_extractor_rejects_unknown_evidence_location_identifier(tmp_path):
    transcript = tmp_path / "session.jsonl"
    write_transcript(transcript)  # only "line-0" exists

    def runner(*args, **kwargs):
        return cursor_response({"records": [{"question": "Q", "summary": "S", "evidence_location": "line-5"}]})

    records = CursorExtractor(runner=runner).extract(transcript)

    assert records[0].evidence_location is None


def test_cursor_extractor_rejects_an_entirely_invented_identifier(tmp_path):
    # Not even shaped like a real fallback identifier — proves rejection is a
    # genuine membership check against the sanitizer's own entries, not a
    # pattern/range check that a plausible-looking guess could satisfy.
    transcript = tmp_path / "session.jsonl"
    write_transcript(transcript)

    def runner(*args, **kwargs):
        return cursor_response({"records": [{"question": "Q", "summary": "S", "evidence_location": "made-up-turn-99"}]})

    records = CursorExtractor(runner=runner).extract(transcript)

    assert records[0].evidence_location is None


def test_cursor_extractor_multiline_record_resolves_to_the_full_multiline_evidence(tmp_path):
    # One raw record with several content blocks renders as text containing
    # embedded newlines — it must still be exactly one selectable identifier,
    # whose preserved_text carries the whole multiline entry, not just
    # whatever a naive per-rendered-line scheme would have picked up.
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "Checked the logs."},
                        {"type": "text", "text": "Found a heartbeat timeout."},
                    ]
                },
            }
        )
        + "\n"
    )

    def runner(*args, **kwargs):
        return cursor_response({"records": [{"question": "Q", "summary": "S", "evidence_location": "line-0"}]})

    records = CursorExtractor(runner=runner).extract(transcript)

    location = records[0].evidence_location
    assert location.identifier == "line-0"
    assert "Checked the logs." in location.preserved_text
    assert "Found a heartbeat timeout." in location.preserved_text


def test_evidence_location_preserved_text_survives_transcript_mutation(tmp_path):
    # A citation must remain resolvable after the original transcript
    # changes — preserved_text is a snapshot captured at extraction time,
    # never re-derived by re-reading the live file.
    transcript = tmp_path / "session.jsonl"
    write_transcript(transcript)

    def runner(*args, **kwargs):
        return cursor_response({"records": [{"question": "Q", "summary": "S", "evidence_location": "line-0"}]})

    records = CursorExtractor(runner=runner).extract(transcript)
    original_preserved_text = records[0].evidence_location.preserved_text

    transcript.write_text(json.dumps({"type": "user", "message": {"content": "totally different content"}}) + "\n")

    assert records[0].evidence_location.preserved_text == original_preserved_text
    assert "totally different content" not in original_preserved_text
