import json
import re
import shutil
import sqlite3
from pathlib import Path

import pytest

from session_rag.artifacts import artifact_path, find_record, job_status_path, load_active_episode_records, read_active_hash
from session_rag.cli import run
from session_rag.extractors.base import (
    EvidenceLocation,
    ExtractionBlocked,
    ExtractionError,
    ExtractionPendingRetry,
    ProjectProvenance,
)
from session_rag.hook import HookConfig, _INTRO, _estimate_tokens, _format_record, handle_user_prompt
from session_rag.retrieval import RetrievalScope, search
from session_rag.overlay import read_state

from conftest import make_record


class FakeExtractor:
    name = "fake"
    model = "fake-model"
    prompt_version = 1

    def __init__(self, records=None, error=None, project_id=None):
        self._records = records
        self._error = error
        # Mirrors CursorExtractor.project_id — introspected via getattr by
        # run_extraction so a failed attempt can still record which project
        # it was configured for (see write_job_status).
        self.project_id = project_id
        self.calls = 0
        self.last_transcript = None

    def extract(self, transcript):
        self.calls += 1
        self.last_transcript = transcript
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


class FakeHealthChecker:
    def __init__(self, findings):
        self.findings = findings
        self.calls = 0

    def analyze(self, project_id, records):
        self.calls += 1
        return self.findings


def _extract_and_activate(artifacts_dir, transcript, content, question, summary):
    transcript.write_text(json.dumps({"type": "user", "message": {"content": content}}) + "\n")
    record = make_record(question=question, summary=summary, source=str(transcript.resolve()), source_session_id=transcript.stem)
    run(["extract-session", str(transcript), "--artifacts", str(artifacts_dir)], extractor=FakeExtractor([record]))


def test_cli_ingests_from_artifacts_and_returns_cited_search_results(tmp_path, capsys):
    transcript = tmp_path / "session-123.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    _extract_and_activate(
        artifacts_dir,
        transcript,
        "why did rabbitmq reconnect",
        question="Why did RabbitMQ reconnect?",
        summary="The heartbeat timeout caused the reconnect.",
    )
    embedder = KeywordEmbedder()
    capsys.readouterr()

    assert run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder) == 0
    capsys.readouterr()
    assert run(["search", "rabbitmq timeout", "--database", str(database), "--artifacts", str(artifacts_dir), "--global-scope"], embedder) == 0

    output = capsys.readouterr().out
    assert "heartbeat timeout caused the reconnect" in output
    assert "session-123" in output


def test_cli_uses_storage_paths_from_toml_config(tmp_path, capsys):
    transcript = tmp_path / "session-123.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'artifacts = "{artifacts_dir}"\n'
        f'database = "{database}"\n'
        'operator_id = "ian"\n'
    )
    _extract_and_activate(
        artifacts_dir,
        transcript,
        "why did rabbitmq reconnect",
        question="Why did RabbitMQ reconnect?",
        summary="The heartbeat timeout caused the reconnect.",
    )
    embedder = KeywordEmbedder()
    capsys.readouterr()

    assert run(["--config", str(config_path), "ingest"], embedder) == 0
    assert run(["--config", str(config_path), "search", "rabbitmq timeout", "--global-scope"], embedder) == 0

    output = capsys.readouterr().out
    assert "heartbeat timeout caused the reconnect" in output


def test_cli_config_show_reports_effective_project_for_current_directory(tmp_path, capsys, monkeypatch):
    project_root = tmp_path / "lvcore"
    project_root.mkdir()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'artifacts = "{tmp_path / "artifacts"}"\n'
        f'database = "{tmp_path / "database"}"\n'
        'operator_id = "ian"\n'
        '[extractor]\n'
        'mode = "ask"\n'
        'model = "test-model"\n'
        'max_sanitized_chars = 500000\n'
        '[projects.lvcore]\n'
        f'root = "{project_root}"\n'
    )
    monkeypatch.chdir(project_root)

    assert run(["--config", str(config_path), "config", "show"]) == 0

    shown = json.loads(capsys.readouterr().out)
    assert shown["operator_id"] == "ian"
    assert shown["extractor"]["max_sanitized_chars"] == 500000
    assert shown["project"] == {"id": "lvcore", "root": str(project_root), "knowledge_base": None}


def test_cli_capture_latest_extracts_newest_registered_project_session_and_indexes_it(
    tmp_path, capsys, monkeypatch
):
    project_root = tmp_path / "lvcore"
    project_root.mkdir()
    artifacts = tmp_path / "artifacts"
    database = tmp_path / "database"
    claude_home = tmp_path / ".claude"
    transcript_dir = claude_home / "projects" / re.sub(r"[^A-Za-z0-9_-]", "-", str(project_root.resolve()))
    transcript_dir.mkdir(parents=True)
    older = transcript_dir / "older.jsonl"
    latest = transcript_dir / "latest.jsonl"
    older.write_text('{"type":"user","message":{"content":"old"}}\n')
    latest.write_text('{"type":"user","message":{"content":"rabbitmq latest"}}\n')
    older.touch()
    latest.touch()
    older_mtime = latest.stat().st_mtime - 10
    import os

    os.utime(older, (older_mtime, older_mtime))
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'artifacts = "{artifacts}"\n'
        f'database = "{database}"\n'
        'operator_id = "ian"\n'
        '[projects.lvcore]\n'
        f'root = "{project_root}"\n'
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_home))
    monkeypatch.chdir(project_root)
    record = make_record(
        question="What was learned?",
        summary="RabbitMQ latest session evidence.",
        source=str(latest.resolve()),
        source_session_id="latest",
        project=ProjectProvenance(project_id="lvcore", project_root=str(project_root)),
    )
    extractor = FakeExtractor([record], project_id="lvcore")

    assert run(["--config", str(config_path), "capture", "--latest"], KeywordEmbedder(), extractor) == 0

    result = json.loads(capsys.readouterr().out)
    assert extractor.last_transcript == latest
    assert result == {
        "status": "activated",
        "session_id": "latest",
        "records": 1,
        "indexed": 1,
        "exact_duplicates": 0,
        "possible_duplicates": 0,
    }
    assert run(["--config", str(config_path), "search", "rabbitmq", "--project-id", "lvcore"], KeywordEmbedder()) == 0
    assert "RabbitMQ latest session evidence" in capsys.readouterr().out


def test_cli_capture_rejects_project_not_registered_in_config(tmp_path, capsys, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'artifacts = "{tmp_path / "artifacts"}"\n'
        f'database = "{tmp_path / "database"}"\n'
        'operator_id = "ian"\n'
        '[projects.lvcore]\n'
        f'root = "{tmp_path / "lvcore"}"\n'
    )
    monkeypatch.setenv("SESSION_RAG_PROJECT_ID", "session-rag")
    monkeypatch.setenv("SESSION_RAG_PROJECT_ROOT", str(tmp_path / "session-rag"))
    extractor = FakeExtractor([], project_id="session-rag")

    assert run(["--config", str(config_path), "capture", "--latest"], KeywordEmbedder(), extractor) == 3

    assert extractor.calls == 0
    assert "is not registered in the config" in capsys.readouterr().err


def test_cli_capture_latest_is_no_op_for_unchanged_active_session(tmp_path, capsys, monkeypatch):
    project_root = tmp_path / "lvcore"
    project_root.mkdir()
    claude_home = tmp_path / ".claude"
    transcript_dir = claude_home / "projects" / re.sub(r"[^A-Za-z0-9_-]", "-", str(project_root.resolve()))
    transcript_dir.mkdir(parents=True)
    transcript = transcript_dir / "session-1.jsonl"
    transcript.write_text('{"type":"user","message":{"content":"knowledge"}}\n')
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'artifacts = "{tmp_path / "artifacts"}"\n'
        f'database = "{tmp_path / "database"}"\n'
        'operator_id = "ian"\n'
        '[projects.lvcore]\n'
        f'root = "{project_root}"\n'
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_home))
    monkeypatch.chdir(project_root)
    record = make_record(
        source=str(transcript.resolve()),
        source_session_id="session-1",
        project=ProjectProvenance(project_id="lvcore", project_root=str(project_root)),
    )
    extractor = FakeExtractor([record], project_id="lvcore")

    assert run(["--config", str(config_path), "capture", "--latest"], KeywordEmbedder(), extractor) == 0
    capsys.readouterr()
    assert run(["--config", str(config_path), "capture", "--latest"], KeywordEmbedder(), extractor) == 0

    assert extractor.calls == 1
    assert json.loads(capsys.readouterr().out)["status"] == "no_op"


def test_user_prompt_hook_returns_additional_context(tmp_path):
    transcript = tmp_path / "session-123.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    _extract_and_activate(
        artifacts_dir,
        transcript,
        "why did rabbitmq reconnect",
        question="Why did RabbitMQ reconnect?",
        summary="The heartbeat timeout caused the reconnect.",
    )
    embedder = KeywordEmbedder()
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder)

    response = handle_user_prompt(
        {"hook_event_name": "UserPromptSubmit", "prompt": "What caused RabbitMQ to reconnect?"},
        database,
        artifacts_dir,
        embedder,
        scope=RetrievalScope(global_scope=True),
    )

    assert response["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    context = response["hookSpecificOutput"]["additionalContext"]
    assert "Retrieved local session memory" in context
    assert "heartbeat timeout" in context
    assert "session-123" in context


def test_cli_ingest_rebuilds_index_from_artifacts_alone_no_reextraction(tmp_path, capsys):
    transcript = tmp_path / "session-123.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    _extract_and_activate(
        artifacts_dir,
        transcript,
        "why did rabbitmq reconnect",
        question="Why did RabbitMQ reconnect?",
        summary="The heartbeat timeout caused the reconnect.",
    )
    embedder = KeywordEmbedder()
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder)

    shutil.rmtree(database)
    capsys.readouterr()

    # No extractor passed — if ingest ever tried to re-extract, this would crash.
    exit_code = run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder)
    capsys.readouterr()
    run(["search", "rabbitmq timeout", "--database", str(database), "--artifacts", str(artifacts_dir), "--global-scope"], embedder)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "heartbeat timeout caused the reconnect" in output


def test_search_result_citation_resolves_to_the_exact_artifact_file(tmp_path):
    transcript = tmp_path / "session-123.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    _extract_and_activate(
        artifacts_dir, transcript, "why did rabbitmq reconnect", question="Q", summary="RabbitMQ heartbeat"
    )
    embedder = KeywordEmbedder()
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder)

    results, _trace = search(database, artifacts_dir, "rabbitmq", embedder, scope=RetrievalScope(global_scope=True))

    assert len(results) == 1
    result = results[0]
    active_hash = read_active_hash(artifacts_dir, source_type=result["source_type"], source_id=result["source_id"])
    resolved_path = artifact_path(
        artifacts_dir, source_type=result["source_type"], source_id=result["source_id"], hash_value=active_hash
    )
    assert resolved_path.exists()
    assert result["source_hash"] == active_hash


def test_cli_ingest_indexes_only_active_revision(tmp_path, capsys):
    transcript = tmp_path / "session-123.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    _extract_and_activate(artifacts_dir, transcript, "v1", question="Old question", summary="Old summary")
    _extract_and_activate(artifacts_dir, transcript, "v2", question="New question", summary="New summary")
    embedder = KeywordEmbedder()

    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder)
    capsys.readouterr()
    run(["search", "new question", "--database", str(database), "--artifacts", str(artifacts_dir), "--global-scope"], embedder)
    output = capsys.readouterr().out

    assert "New question" in output
    assert "Old question" not in output


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
        tmp_path / "artifacts",
        KeywordEmbedder(),
    )

    assert response == {}


def _extract_and_get_record_id(artifacts_dir, transcript, content, question, summary, capsys):
    transcript.write_text(json.dumps({"type": "user", "message": {"content": content}}) + "\n")
    record = make_record(question=question, summary=summary, source=str(transcript.resolve()), source_session_id=transcript.stem)
    capsys.readouterr()
    run(["extract-session", str(transcript), "--artifacts", str(artifacts_dir)], extractor=FakeExtractor([record]))
    output = json.loads(capsys.readouterr().out)
    return output["records"][0]["id"]


def test_verify_transitions_unreviewed_to_verified(tmp_path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    transcript = tmp_path / "session-123.jsonl"
    record_id = _extract_and_get_record_id(artifacts_dir, transcript, "hi", "Q", "S", capsys)

    exit_code = run(["verify", record_id, "--artifacts", str(artifacts_dir)])

    assert exit_code == 0
    assert f"verified {record_id}" in capsys.readouterr().out
    exit_code = run(["history", record_id, "--artifacts", str(artifacts_dir)])
    history = json.loads(capsys.readouterr().out)
    assert history["verification_status"] == "verified"


def test_reject_then_reject_again_is_an_invalid_transition(tmp_path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    transcript = tmp_path / "session-123.jsonl"
    record_id = _extract_and_get_record_id(artifacts_dir, transcript, "hi", "Q", "S", capsys)

    run(["reject", record_id, "--artifacts", str(artifacts_dir)])
    capsys.readouterr()
    exit_code = run(["reject", record_id, "--artifacts", str(artifacts_dir)])

    assert exit_code == 1
    assert "cannot move from 'rejected'" in capsys.readouterr().err


def test_supersede_requires_replacement_id_argument(tmp_path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    transcript = tmp_path / "session-123.jsonl"
    record_id = _extract_and_get_record_id(artifacts_dir, transcript, "hi", "Q", "S", capsys)

    with pytest.raises(SystemExit):
        run(["supersede", record_id, "--artifacts", str(artifacts_dir)])


def test_supersede_records_replacement_link_and_is_visible_via_history(tmp_path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    old_transcript = tmp_path / "old-session.jsonl"
    new_transcript = tmp_path / "new-session.jsonl"
    old_id = _extract_and_get_record_id(artifacts_dir, old_transcript, "hi old", "Q old", "S old", capsys)
    new_id = _extract_and_get_record_id(artifacts_dir, new_transcript, "hi new", "Q new", "S new", capsys)

    exit_code = run(["supersede", old_id, new_id, "--artifacts", str(artifacts_dir)])
    capsys.readouterr()
    run(["history", old_id, "--artifacts", str(artifacts_dir)])
    history = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert history["verification_status"] == "superseded"
    assert history["superseded_by"] == new_id


def test_supersede_rejects_a_nonexistent_replacement_id(tmp_path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    transcript = tmp_path / "session-123.jsonl"
    record_id = _extract_and_get_record_id(artifacts_dir, transcript, "hi", "Q", "S", capsys)

    exit_code = run(["supersede", record_id, "sha256:doesnotexist:0", "--artifacts", str(artifacts_dir)])

    assert exit_code == 1
    assert "does not exist" in capsys.readouterr().err


def test_verify_unknown_record_id_fails_clearly(tmp_path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    exit_code = run(["verify", "sha256:doesnotexist:0", "--artifacts", str(artifacts_dir)])

    assert exit_code == 1
    assert "no such record" in capsys.readouterr().err


def test_rejected_records_excluded_from_search_but_visible_via_history(tmp_path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    transcript = tmp_path / "session-123.jsonl"
    record_id = _extract_and_get_record_id(
        artifacts_dir, transcript, "why did rabbitmq reconnect", "Q", "RabbitMQ heartbeat issue", capsys
    )
    run(["reject", record_id, "--artifacts", str(artifacts_dir)])

    embedder = KeywordEmbedder()
    capsys.readouterr()
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder)
    capsys.readouterr()
    run(["search", "rabbitmq", "--database", str(database), "--artifacts", str(artifacts_dir)], embedder)
    search_output = capsys.readouterr().out

    assert "RabbitMQ heartbeat issue" not in search_output
    assert "No relevant session memory found" in search_output

    run(["history", record_id, "--artifacts", str(artifacts_dir)])
    history_output = json.loads(capsys.readouterr().out)
    assert history_output["verification_status"] == "rejected"
    assert history_output["question"] == "Q"


def test_verification_state_survives_lancedb_rebuild(tmp_path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    rejected_transcript = tmp_path / "rejected-session.jsonl"
    kept_transcript = tmp_path / "kept-session.jsonl"
    rejected_id = _extract_and_get_record_id(
        artifacts_dir, rejected_transcript, "why did rabbitmq reconnect", "Q1", "RabbitMQ heartbeat issue", capsys
    )
    _extract_and_get_record_id(
        artifacts_dir, kept_transcript, "why did postgres crash", "Q2", "Postgres out of memory", capsys
    )
    run(["reject", rejected_id, "--artifacts", str(artifacts_dir)])
    embedder = KeywordEmbedder()
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder)

    # Deleting the derived index must not lose verification state — it lives
    # in the overlay, outside both the artifact and LanceDB.
    shutil.rmtree(database)
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder)
    capsys.readouterr()
    run(["search", "rabbitmq postgres", "--database", str(database), "--artifacts", str(artifacts_dir), "--global-scope"], embedder)
    output = capsys.readouterr().out

    assert "RabbitMQ heartbeat issue" not in output
    assert "Postgres out of memory" in output


def test_supersession_link_survives_lancedb_rebuild(tmp_path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    old_transcript = tmp_path / "old-session.jsonl"
    new_transcript = tmp_path / "new-session.jsonl"
    old_id = _extract_and_get_record_id(artifacts_dir, old_transcript, "hi old", "Q old", "S old", capsys)
    new_id = _extract_and_get_record_id(artifacts_dir, new_transcript, "hi new", "Q new", "S new", capsys)
    run(["supersede", old_id, new_id, "--artifacts", str(artifacts_dir)])

    embedder = KeywordEmbedder()
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder)
    if database.exists():
        shutil.rmtree(database)
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder)

    capsys.readouterr()
    run(["history", old_id, "--artifacts", str(artifacts_dir)])
    history = json.loads(capsys.readouterr().out)

    assert history["verification_status"] == "superseded"
    assert history["superseded_by"] == new_id


def test_forget_removes_artifacts_overlay_and_index_rows(tmp_path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    transcript = tmp_path / "session-123.jsonl"
    record_id = _extract_and_get_record_id(artifacts_dir, transcript, "hi", "Q", "S", capsys)
    run(["reject", record_id, "--artifacts", str(artifacts_dir)])
    embedder = KeywordEmbedder()
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder)

    exit_code = run(["forget", "session-123", "--artifacts", str(artifacts_dir), "--database", str(database)])

    assert exit_code == 0
    assert not (artifacts_dir / "claude_session" / "session-123").exists()
    overlay_path = artifacts_dir / "overlay.json"
    if overlay_path.exists():
        assert record_id not in overlay_path.read_text()
    assert find_record(artifacts_dir, record_id) is None


def test_forget_prints_summary_only_to_terminal_no_file_retains_source(tmp_path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    transcript = tmp_path / "secret-session.jsonl"
    record_id = _extract_and_get_record_id(artifacts_dir, transcript, "hi", "Q", "S", capsys)
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder=KeywordEmbedder())
    capsys.readouterr()

    run(["forget", "secret-session", "--artifacts", str(artifacts_dir), "--database", str(database)])
    output = capsys.readouterr()

    assert "secret-session" in output.out  # terminal display is fine — ephemeral
    # No file anywhere under artifacts/ mentions the forgotten source id.
    for path in artifacts_dir.rglob("*"):
        if path.is_file():
            assert "secret-session" not in path.read_text()


def test_forget_does_not_block_reingestion_of_same_source(tmp_path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    transcript = tmp_path / "session-123.jsonl"
    _extract_and_get_record_id(artifacts_dir, transcript, "hi", "Q", "S", capsys)
    run(["forget", "session-123", "--artifacts", str(artifacts_dir), "--database", str(database)])

    record = make_record(question="Q2", summary="S2", source=str(transcript.resolve()), source_session_id="session-123")
    exit_code = run(
        ["extract-session", str(transcript), "--artifacts", str(artifacts_dir)], extractor=FakeExtractor([record])
    )

    assert exit_code == 0
    assert (artifacts_dir / "claude_session" / "session-123").exists()


def test_forget_source_removes_job_status_entirely(tmp_path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    transcript = tmp_path / "session-123.jsonl"
    transcript.write_text(json.dumps({"type": "user", "message": {"content": "hi"}}) + "\n")
    run(
        ["extract-session", str(transcript), "--artifacts", str(artifacts_dir)],
        extractor=FakeExtractor(error=ExtractionPendingRetry("Cursor timed out"), project_id="project-a"),
    )
    job_path = job_status_path(artifacts_dir, source_type="claude_session", source_id="session-123")
    assert job_path.exists()

    run(["forget", "session-123", "--artifacts", str(artifacts_dir), "--database", str(database)])

    assert not job_path.exists()


def test_forget_purges_record_ids_from_retrieval_trace_log(tmp_path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    transcript = tmp_path / "session-123.jsonl"
    record_id = _extract_and_get_record_id(artifacts_dir, transcript, "rabbitmq heartbeat", "Q", "S", capsys)
    embedder = KeywordEmbedder()
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder)
    run(["search", "rabbitmq", "--database", str(database), "--artifacts", str(artifacts_dir)], embedder)
    trace_log = artifacts_dir / "retrieval_traces.jsonl"
    assert record_id in trace_log.read_text()

    run(["forget", "session-123", "--artifacts", str(artifacts_dir), "--database", str(database)])

    assert record_id not in trace_log.read_text()


class SlowEmbedder:
    dimensions = 2
    model_name = "slow-test"

    def embed(self, texts):
        import time

        time.sleep(0.3)
        return [[0.0, 0.0] for _ in texts]


class RaisingEmbedder:
    dimensions = 2
    model_name = "raising-test"

    def embed(self, texts):
        raise RuntimeError("embedding backend unavailable")


def test_hook_fails_open_on_slow_retrieval_timeout(tmp_path):
    transcript = tmp_path / "session-123.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    _extract_and_activate(artifacts_dir, transcript, "hi", question="Q", summary="S")
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], KeywordEmbedder())

    response = handle_user_prompt(
        {"hook_event_name": "UserPromptSubmit", "prompt": "hi"},
        database,
        artifacts_dir,
        SlowEmbedder(),
        config=HookConfig(retrieval_timeout_ms=10),
    )

    assert response == {}
    # Metric recording for THIS call has essentially no remaining budget
    # (retrieval already used the full 10ms) and is itself best-effort — see
    # test_hook_records_metric_without_prompt_text_on_success for the
    # no-prompt-text guarantee in the case where recording does complete.


class SlowInitEmbedder:
    """Simulates process/model-init latency happening in the constructor —
    not in embed() — proving retrieval_timeout_ms bounds embedder
    construction too, not just search() (Fix 6)."""

    dimensions = 2
    model_name = "slow-init-test"

    def __init__(self):
        import time

        time.sleep(0.3)

    def embed(self, texts):
        return [[0.0, 0.0] for _ in texts]


def test_hook_timeout_bounds_embedder_construction_not_just_search(tmp_path):
    import time

    transcript = tmp_path / "session-123.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    _extract_and_activate(artifacts_dir, transcript, "hi", question="Q", summary="S")
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], KeywordEmbedder())

    started = time.monotonic()
    response = handle_user_prompt(
        {"hook_event_name": "UserPromptSubmit", "prompt": "hi"},
        database,
        artifacts_dir,
        embedder_factory=SlowInitEmbedder,
        config=HookConfig(retrieval_timeout_ms=10),
    )
    elapsed_ms = (time.monotonic() - started) * 1000

    assert response == {}
    # Bounded by the configured timeout, not the slow __init__ — proves
    # construction happens inside the timed path, not before it.
    assert elapsed_ms < 300


def test_hook_timeout_bounds_formatting_not_just_construction_and_search(tmp_path, monkeypatch):
    import time

    import session_rag.hook as hook_module

    transcript = tmp_path / "session-123.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    _extract_and_activate(artifacts_dir, transcript, "hi", question="Q", summary="S")
    embedder = KeywordEmbedder()
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder)

    real_format = hook_module._build_context_within_budget

    def slow_format(results, max_tokens):
        time.sleep(0.3)
        return real_format(results, max_tokens)

    # Formatting runs after search returns, so it can only be proven bounded
    # by faking it slow — nothing in the format step itself is naturally
    # slow enough to demonstrate the gap otherwise.
    monkeypatch.setattr(hook_module, "_build_context_within_budget", slow_format)

    started = time.monotonic()
    response = handle_user_prompt(
        {"hook_event_name": "UserPromptSubmit", "prompt": "hi"},
        database,
        artifacts_dir,
        embedder,
        # Must actually surface a candidate to reach the formatting step —
        # global_scope=True so this isn't defeated by the unscoped-record
        # scope gate (Fix 3) before formatting is ever attempted.
        scope=RetrievalScope(global_scope=True),
        config=HookConfig(retrieval_timeout_ms=10),
    )
    elapsed_ms = (time.monotonic() - started) * 1000

    assert response == {}
    # Bounded by the configured timeout, not the slow formatting step —
    # proves formatting runs inside the timed path, not after it.
    assert elapsed_ms < 300


def test_hook_slow_metrics_writer_does_not_delay_the_hook_past_the_retrieval_deadline(tmp_path, monkeypatch):
    import time

    import session_rag.hook as hook_module

    transcript = tmp_path / "session-123.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    _extract_and_activate(artifacts_dir, transcript, "hi", question="Q", summary="S")
    embedder = KeywordEmbedder()
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder)

    real_record_metric = hook_module._record_metric

    def slow_record_metric(*args, **kwargs):
        time.sleep(0.3)
        return real_record_metric(*args, **kwargs)

    monkeypatch.setattr(hook_module, "_record_metric", slow_record_metric)

    # A small retrieval_timeout_ms leaves little remaining budget for
    # metrics after the (near-instant) search — the slow writer's 300ms
    # should be abandoned almost immediately, not awaited. Total latency
    # must stay bounded by retrieval_timeout_ms ALONE, not
    # retrieval_timeout_ms plus however slow the metrics writer is.
    started = time.monotonic()
    response = handle_user_prompt(
        {"hook_event_name": "UserPromptSubmit", "prompt": "hi"},
        database,
        artifacts_dir,
        embedder,
        scope=RetrievalScope(global_scope=True),
        config=HookConfig(retrieval_timeout_ms=50),
    )
    elapsed_ms = (time.monotonic() - started) * 1000

    assert elapsed_ms < 150  # nowhere near the metrics writer's 300ms
    # The response is exactly what a fast metrics write would have produced
    # — metrics failure/slowness never changes it.
    assert response["hookSpecificOutput"]["additionalContext"]


def test_hook_metric_writer_exception_does_not_escape_or_change_response(tmp_path, monkeypatch):
    import session_rag.hook as hook_module

    transcript = tmp_path / "session-123.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    _extract_and_activate(artifacts_dir, transcript, "hi", question="Q", summary="S")
    embedder = KeywordEmbedder()
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder)

    def raising_record_metric(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(hook_module, "_record_metric", raising_record_metric)

    # If the exception escaped handle_user_prompt, this call itself would
    # raise and fail the test — the assertion below only runs if it didn't.
    response = handle_user_prompt(
        {"hook_event_name": "UserPromptSubmit", "prompt": "hi"},
        database,
        artifacts_dir,
        embedder,
        scope=RetrievalScope(global_scope=True),
    )

    assert response["hookSpecificOutput"]["additionalContext"]


def test_hook_fails_open_within_retrieval_timeout_even_when_metrics_writer_is_also_slow(tmp_path, monkeypatch):
    import time

    import session_rag.hook as hook_module

    transcript = tmp_path / "session-123.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    _extract_and_activate(artifacts_dir, transcript, "hi", question="Q", summary="S")
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], KeywordEmbedder())

    real_record_metric = hook_module._record_metric

    def slow_record_metric(*args, **kwargs):
        time.sleep(0.3)
        return real_record_metric(*args, **kwargs)

    monkeypatch.setattr(hook_module, "_record_metric", slow_record_metric)

    # Retrieval itself already times out here (SlowEmbedder, 10ms budget),
    # leaving ~0 remaining budget for metrics — the slow metrics writer must
    # be abandoned immediately, not awaited for its own 300ms on top. Total
    # latency stays bounded by retrieval_timeout_ms alone, proving metrics
    # can never add its own delay when there's no time left to give it.
    started = time.monotonic()
    response = handle_user_prompt(
        {"hook_event_name": "UserPromptSubmit", "prompt": "hi"},
        database,
        artifacts_dir,
        SlowEmbedder(),
        config=HookConfig(retrieval_timeout_ms=10),
    )
    elapsed_ms = (time.monotonic() - started) * 1000

    assert response == {}
    assert elapsed_ms < 100


def test_hook_fails_open_on_retrieval_error(tmp_path):
    transcript = tmp_path / "session-123.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    _extract_and_activate(artifacts_dir, transcript, "hi", question="Q", summary="S")
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], KeywordEmbedder())

    response = handle_user_prompt(
        {"hook_event_name": "UserPromptSubmit", "prompt": "hi"}, database, artifacts_dir, RaisingEmbedder()
    )

    assert response == {}
    metrics = [json.loads(line) for line in (artifacts_dir / "hook_metrics.jsonl").read_text().splitlines()]
    assert metrics[-1]["outcome"] == "error"


def test_hook_records_metric_without_prompt_text_on_success(tmp_path):
    transcript = tmp_path / "session-123.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    _extract_and_activate(artifacts_dir, transcript, "why did rabbitmq reconnect", question="Q", summary="RabbitMQ heartbeat")
    embedder = KeywordEmbedder()
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder)

    handle_user_prompt(
        {"hook_event_name": "UserPromptSubmit", "prompt": "a secret prompt about rabbitmq"},
        database, artifacts_dir, embedder,
        scope=RetrievalScope(global_scope=True),
    )

    log_text = (artifacts_dir / "hook_metrics.jsonl").read_text()
    assert "secret prompt" not in log_text
    last = json.loads(log_text.splitlines()[-1])
    assert last["outcome"] == "ok"
    assert "latency_ms" in last
    assert last["result_count"] == 1


def test_hook_never_injects_more_than_max_injected_records(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    embedder = KeywordEmbedder()
    for i in range(5):
        transcript = tmp_path / f"session-{i}.jsonl"
        _extract_and_activate(artifacts_dir, transcript, "rabbitmq", question=f"Q{i}", summary="rabbitmq heartbeat")
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder)

    response = handle_user_prompt(
        {"hook_event_name": "UserPromptSubmit", "prompt": "rabbitmq"},
        database,
        artifacts_dir,
        embedder,
        scope=RetrievalScope(global_scope=True),
        config=HookConfig(max_injected_records=3, max_injected_tokens=100_000),
    )

    context = response["hookSpecificOutput"]["additionalContext"]
    assert sum(context.count(f"Q{i}") for i in range(5)) <= 3


def test_hook_omits_whole_records_rather_than_truncating_for_token_budget(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    embedder = KeywordEmbedder()
    long_summary = "rabbitmq " + ("word " * 20)
    for i in range(3):
        transcript = tmp_path / f"session-{i}.jsonl"
        _extract_and_activate(artifacts_dir, transcript, "rabbitmq", question=f"Q{i}", summary=long_summary)
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder)

    # Size the budget precisely: room for the intro plus exactly one record,
    # not two — using the module's own token estimate so this isn't a guess.
    global_scope = RetrievalScope(global_scope=True)
    full_response, _trace = search(database, artifacts_dir, "rabbitmq", embedder, scope=global_scope)
    one_record_tokens = _estimate_tokens(_format_record(1, full_response[0]))
    budget = _estimate_tokens(_INTRO) + one_record_tokens + 1

    response = handle_user_prompt(
        {"hook_event_name": "UserPromptSubmit", "prompt": "rabbitmq"},
        database,
        artifacts_dir,
        embedder,
        scope=global_scope,
        config=HookConfig(max_injected_records=3, max_injected_tokens=budget),
    )

    context = response["hookSpecificOutput"]["additionalContext"]
    included = sum(context.count(f"Q{i}") for i in range(3))
    assert included == 1
    # Whichever record made it in is whole — the long summary isn't cut mid-word.
    assert long_summary.strip() in context


def test_hook_config_overridable_via_env(monkeypatch):
    monkeypatch.setenv("SESSION_RAG_RETRIEVAL_TIMEOUT_MS", "9999")
    monkeypatch.setenv("SESSION_RAG_MAX_INJECTED_TOKENS", "42")
    monkeypatch.setenv("SESSION_RAG_MAX_INJECTED_RECORDS", "1")

    config = HookConfig.from_env()

    assert config.retrieval_timeout_ms == 9999
    assert config.max_injected_tokens == 42
    assert config.max_injected_records == 1


def test_hook_timeout_thread_is_daemon_and_does_not_block_process_exit(tmp_path):
    import threading

    transcript = tmp_path / "session-123.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    _extract_and_activate(artifacts_dir, transcript, "hi", question="Q", summary="S")
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], KeywordEmbedder())
    before = {t.ident for t in threading.enumerate()}

    handle_user_prompt(
        {"hook_event_name": "UserPromptSubmit", "prompt": "hi"},
        database,
        artifacts_dir,
        SlowEmbedder(),
        config=HookConfig(retrieval_timeout_ms=10),
    )

    leftover = [t for t in threading.enumerate() if t.ident not in before]
    # The abandoned search() thread (if still alive) must be a daemon —
    # never a plain executor thread that would block interpreter exit.
    assert all(t.daemon for t in leftover)


def test_hook_layers_max_injected_records_on_top_of_env_tuned_retrieval_config(tmp_path, monkeypatch):
    transcript = tmp_path / "session-123.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    # Neither term is "rabbitmq"/"postgres", so KeywordEmbedder gives both
    # query and text the same [0.0, 0.0] vector (distance 0) with no shared
    # substring — qualifies by default (ceiling 1.0), excluded once an
    # operator-tuned env var makes the floor impossibly strict.
    _extract_and_activate(artifacts_dir, transcript, "unrelated", question="Q", summary="totally unrelated content")
    embedder = KeywordEmbedder()
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder)

    monkeypatch.setenv("SESSION_RAG_VECTOR_DISTANCE_CEILING", "-1")
    monkeypatch.setenv("SESSION_RAG_LEXICAL_SCORE_FLOOR", "1000000")

    response = handle_user_prompt(
        {"hook_event_name": "UserPromptSubmit", "prompt": "zzz nothing shared"}, database, artifacts_dir, embedder
    )

    assert response == {}


def test_injected_claim_resolves_to_exact_source_revision_and_evidence_location(tmp_path):
    transcript = tmp_path / "session-123.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    transcript.write_text(json.dumps({"type": "user", "message": {"content": "why did rabbitmq reconnect"}}) + "\n")
    evidence_location = EvidenceLocation(identifier="turn-1", preserved_text="User: why did rabbitmq reconnect")
    record = make_record(
        question="Q",
        summary="RabbitMQ heartbeat",
        source=str(transcript.resolve()),
        source_session_id="session-123",
        evidence_location=evidence_location,
    )
    run(["extract-session", str(transcript), "--artifacts", str(artifacts_dir)], extractor=FakeExtractor([record]))
    embedder = KeywordEmbedder()
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder)

    global_scope = RetrievalScope(global_scope=True)
    response = handle_user_prompt(
        {"hook_event_name": "UserPromptSubmit", "prompt": "why did rabbitmq reconnect"},
        database, artifacts_dir, embedder, scope=global_scope,
    )

    context = response["hookSpecificOutput"]["additionalContext"]
    results, _trace = search(database, artifacts_dir, "rabbitmq", embedder, scope=global_scope)
    result = results[0]
    active_hash = read_active_hash(artifacts_dir, source_type=result["source_type"], source_id=result["source_id"])
    resolved_path = artifact_path(
        artifacts_dir, source_type=result["source_type"], source_id=result["source_id"], hash_value=active_hash
    )

    assert resolved_path.exists()
    assert result["source_hash"] == active_hash
    # The injected citation carries the stable identifier, not a misleading
    # sanitized-rendering line number.
    assert result["evidence_location_id"] == "turn-1"
    assert "evidence turn-1" in context
    assert "line 0" not in context

    # The citation resolves to the exact supporting source record — full
    # source_type/source_id/source_hash/identifier/preserved_text together,
    # read straight from the immutable artifact (never the live transcript).
    resolved = find_record(artifacts_dir, result["id"])
    assert resolved["source_type"] == result["source_type"]
    assert resolved["source_id"] == result["source_id"]
    assert resolved["source_hash"] == active_hash
    assert resolved["evidence_location"] == {"identifier": "turn-1", "preserved_text": "User: why did rabbitmq reconnect"}

    # Remains resolvable after the original transcript changes — the
    # preserved text was snapshotted at extraction time, never re-derived by
    # re-reading the live source.
    transcript.write_text(json.dumps({"type": "user", "message": {"content": "unrelated new content"}}) + "\n")
    still_resolved = find_record(artifacts_dir, result["id"])
    assert still_resolved["evidence_location"]["preserved_text"] == "User: why did rabbitmq reconnect"


def _extract_with_project(artifacts_dir, transcript, content, question, summary, project_id, capsys):
    transcript.write_text(json.dumps({"type": "user", "message": {"content": content}}) + "\n")
    record = make_record(
        question=question,
        summary=summary,
        source=str(transcript.resolve()),
        source_session_id=transcript.stem,
        project=ProjectProvenance(project_id=project_id),
    )
    capsys.readouterr()
    run(["extract-session", str(transcript), "--artifacts", str(artifacts_dir)], extractor=FakeExtractor([record]))
    output = json.loads(capsys.readouterr().out)
    return output["records"][0]["id"]


def test_forget_project_removes_all_sources_for_that_project_and_isolates_others(tmp_path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    transcript_a1 = tmp_path / "session-a1.jsonl"
    transcript_a2 = tmp_path / "session-a2.jsonl"
    transcript_b = tmp_path / "session-b.jsonl"
    record_a1 = _extract_with_project(artifacts_dir, transcript_a1, "hi a1", "QA1", "SA1", "project-a", capsys)
    record_a2 = _extract_with_project(artifacts_dir, transcript_a2, "hi a2", "QA2", "SA2", "project-a", capsys)
    record_b = _extract_with_project(artifacts_dir, transcript_b, "hi b", "QB", "SB", "project-b", capsys)
    embedder = KeywordEmbedder()
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database),], embedder)
    run(["search", "hi", "--database", str(database), "--artifacts", str(artifacts_dir), "--global-scope"], embedder)

    exit_code = run(["forget", "--project", "project-a", "--artifacts", str(artifacts_dir), "--database", str(database)])

    assert exit_code == 0
    # Project A's sources are fully erased.
    assert not (artifacts_dir / "claude_session" / "session-a1").exists()
    assert not (artifacts_dir / "claude_session" / "session-a2").exists()
    assert find_record(artifacts_dir, record_a1) is None
    assert find_record(artifacts_dir, record_a2) is None
    # Project B is untouched.
    assert (artifacts_dir / "claude_session" / "session-b").exists()
    assert find_record(artifacts_dir, record_b) is not None


def test_forget_project_removes_lancedb_rows_and_traces_but_not_other_projects(tmp_path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    transcript_a = tmp_path / "session-a.jsonl"
    transcript_b = tmp_path / "session-b.jsonl"
    record_a = _extract_with_project(artifacts_dir, transcript_a, "alpha content", "QA", "SA project-a", "project-a", capsys)
    record_b = _extract_with_project(artifacts_dir, transcript_b, "beta content", "QB", "SB project-b", "project-b", capsys)
    embedder = KeywordEmbedder()
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder)
    capsys.readouterr()
    run(["search", "SA", "--database", str(database), "--artifacts", str(artifacts_dir), "--global-scope"], embedder)
    trace_log = artifacts_dir / "retrieval_traces.jsonl"
    assert record_a in trace_log.read_text()

    run(["forget", "--project", "project-a", "--artifacts", str(artifacts_dir), "--database", str(database)])

    assert record_a not in trace_log.read_text()
    capsys.readouterr()
    run(["search", "project-b", "--database", str(database), "--artifacts", str(artifacts_dir), "--global-scope"], embedder)
    output = capsys.readouterr().out
    assert "SB project-b" in output


def test_forget_project_does_not_delete_another_projects_active_revision_of_the_same_source(tmp_path, capsys):
    """A source_id's revision history can span more than one project if the
    operator's project config changes between extractions of the same
    transcript. Forgetting the older project must not delete the newer
    project's now-active revision for that same source_id, and must not
    hard-delete its still-valid LanceDB rows either — isolation has to hold
    within one source, not just across separate sources."""

    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    transcript = tmp_path / "session-a.jsonl"

    record_a = _extract_with_project(artifacts_dir, transcript, "v1 content", "QA", "SA project-a", "project-a", capsys)
    # Re-extract the SAME transcript (source_id stays "session-a") under a
    # different project — this becomes the new active revision.
    record_b = _extract_with_project(artifacts_dir, transcript, "v2 content", "QB", "SB project-b", "project-b", capsys)
    embedder = KeywordEmbedder()
    run(["ingest", "--artifacts", str(artifacts_dir), "--database", str(database)], embedder)

    exit_code = run(["forget", "--project", "project-a", "--artifacts", str(artifacts_dir), "--database", str(database)])

    assert exit_code == 0
    # project-b's now-active revision for this same source_id survives.
    assert (artifacts_dir / "claude_session" / "session-a").exists()
    assert find_record(artifacts_dir, record_b) is not None
    # project-a's superseded revision is gone.
    assert find_record(artifacts_dir, record_a) is None

    capsys.readouterr()
    run(["search", "project-b", "--database", str(database), "--artifacts", str(artifacts_dir), "--global-scope"], embedder)
    output = capsys.readouterr().out
    assert "SB project-b" in output


def test_forget_project_removes_job_status_belonging_to_the_forgotten_project(tmp_path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    transcript = tmp_path / "session-a.jsonl"

    # project-a's revision, later superseded.
    _extract_with_project(artifacts_dir, transcript, "v1", "QA", "SA project-a", "project-a", capsys)
    # project-b's revision becomes active.
    _extract_with_project(artifacts_dir, transcript, "v2", "QB", "SB project-b", "project-b", capsys)
    # A later attempt, configured for project-a, that fails — never produces
    # a revision, but does write Extraction Job Status attributed to
    # project-a (see write_job_status).
    transcript.write_text(json.dumps({"type": "user", "message": {"content": "v3"}}) + "\n")
    capsys.readouterr()
    run(
        ["extract-session", str(transcript), "--artifacts", str(artifacts_dir)],
        extractor=FakeExtractor(error=ExtractionPendingRetry("Cursor timed out"), project_id="project-a"),
    )
    job_path = job_status_path(artifacts_dir, source_type="claude_session", source_id="session-a")
    assert job_path.exists()

    run(["forget", "--project", "project-a", "--artifacts", str(artifacts_dir), "--database", str(database)])

    assert not job_path.exists()
    # project-b's active revision is untouched.
    assert read_active_hash(artifacts_dir, source_type="claude_session", source_id="session-a") is not None


def test_forget_project_preserves_job_status_belonging_to_a_different_project(tmp_path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    transcript = tmp_path / "session-a.jsonl"

    _extract_with_project(artifacts_dir, transcript, "v1", "QA", "SA project-a", "project-a", capsys)
    _extract_with_project(artifacts_dir, transcript, "v2", "QB", "SB project-b", "project-b", capsys)
    transcript.write_text(json.dumps({"type": "user", "message": {"content": "v3"}}) + "\n")
    capsys.readouterr()
    run(
        ["extract-session", str(transcript), "--artifacts", str(artifacts_dir)],
        extractor=FakeExtractor(error=ExtractionPendingRetry("Cursor timed out"), project_id="project-a"),
    )
    job_path = job_status_path(artifacts_dir, source_type="claude_session", source_id="session-a")
    original_status = json.loads(job_path.read_text())

    # Forgetting a DIFFERENT project (project-b, which also removes
    # project-b's active revision) must not touch job status attributed to
    # project-a — unrelated project metadata survives.
    run(["forget", "--project", "project-b", "--artifacts", str(artifacts_dir), "--database", str(database)])

    assert job_path.exists()
    assert json.loads(job_path.read_text()) == original_status


def test_forget_project_leaves_no_trace_of_erased_job_status_anywhere_under_artifact_root(tmp_path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    transcript = tmp_path / "session-a.jsonl"

    _extract_with_project(artifacts_dir, transcript, "v1", "QA", "SA project-a", "project-a", capsys)
    _extract_with_project(artifacts_dir, transcript, "v2", "QB", "SB project-b", "project-b", capsys)
    transcript.write_text(json.dumps({"type": "user", "message": {"content": "v3"}}) + "\n")
    capsys.readouterr()
    run(
        ["extract-session", str(transcript), "--artifacts", str(artifacts_dir)],
        extractor=FakeExtractor(
            error=ExtractionPendingRetry("a distinctive failure reason xyz123"), project_id="project-a"
        ),
    )
    job_status_before = json.loads(
        job_status_path(artifacts_dir, source_type="claude_session", source_id="session-a").read_text()
    )
    erased_hash = job_status_before["attempted_hash"]

    run(["forget", "--project", "project-a", "--artifacts", str(artifacts_dir), "--database", str(database)])

    for path in artifacts_dir.rglob("*"):
        if path.is_file():
            content = path.read_text()
            assert erased_hash not in content
            assert "a distinctive failure reason xyz123" not in content


def test_forget_project_discovers_and_erases_a_source_with_only_a_failed_attempt(tmp_path, capsys):
    # A source whose ONLY trace of a project is a failed extraction attempt
    # (never produced an Episode Record) must still be discoverable and
    # erasable by project-scoped forget — otherwise its attempted_hash and
    # failure reason would never be removable for that project at all.
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    transcript = tmp_path / "session-only-failed.jsonl"
    transcript.write_text(json.dumps({"type": "user", "message": {"content": "hi"}}) + "\n")
    run(
        ["extract-session", str(transcript), "--artifacts", str(artifacts_dir)],
        extractor=FakeExtractor(
            error=ExtractionPendingRetry("a distinctive never-succeeded reason"), project_id="project-a"
        ),
    )
    job_path = job_status_path(artifacts_dir, source_type="claude_session", source_id="session-only-failed")
    assert job_path.exists()
    source_dir = artifacts_dir / "claude_session" / "session-only-failed"
    assert source_dir.exists()

    exit_code = run(
        ["forget", "--project", "project-a", "--artifacts", str(artifacts_dir), "--database", str(database)]
    )

    assert exit_code == 0
    assert not source_dir.exists()
    for path in artifacts_dir.rglob("*"):
        if path.is_file():
            assert "a distinctive never-succeeded reason" not in path.read_text()


def test_forget_project_does_not_block_reingestion(tmp_path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    transcript = tmp_path / "session-a.jsonl"
    _extract_with_project(artifacts_dir, transcript, "hi", "QA", "SA", "project-a", capsys)
    run(["forget", "--project", "project-a", "--artifacts", str(artifacts_dir), "--database", str(database)])

    record = make_record(
        question="Q2", summary="S2", source=str(transcript.resolve()), source_session_id="session-a",
        project=ProjectProvenance(project_id="project-a"),
    )
    exit_code = run(
        ["extract-session", str(transcript), "--artifacts", str(artifacts_dir)], extractor=FakeExtractor([record])
    )

    assert exit_code == 0
    assert (artifacts_dir / "claude_session" / "session-a").exists()


def test_forget_requires_exactly_one_of_source_id_or_project(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    database = tmp_path / "memory.lance"

    exit_code = run(["forget", "--artifacts", str(artifacts_dir), "--database", str(database)])

    assert exit_code == 1


def test_import_sessions_dry_run_discovers_all_configured_claude_projects(tmp_path, capsys, monkeypatch):
    artifacts = tmp_path / "artifacts"
    database = tmp_path / "database"
    claude_home = tmp_path / ".claude"
    project_a = tmp_path / "a"
    project_b = tmp_path / "b"
    for project, session_id in ((project_a, "one"), (project_b, "two")):
        transcript_dir = claude_home / "projects" / re.sub(r"[^A-Za-z0-9_-]", "-", str(project.resolve()))
        transcript_dir.mkdir(parents=True)
        (transcript_dir / f"{session_id}.jsonl").write_text('{"type":"user","message":{"content":"hello"}}\n')
    config = tmp_path / "config.toml"
    config.write_text(
        f'operator_id = "ian"\nartifacts = "{artifacts}"\ndatabase = "{database}"\n'
        f'[projects.a]\nroot = "{project_a}"\n[projects.b]\nroot = "{project_b}"\n'
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_home))

    assert run(["--config", str(config), "import-sessions", "--source", "claude", "--configured-projects", "--dry-run"]) == 0

    assert json.loads(capsys.readouterr().out)["discovered"] == 2


def test_import_sessions_project_filter_and_resume_are_safe(tmp_path, capsys, monkeypatch):
    artifacts = tmp_path / "artifacts"
    database = tmp_path / "database"
    claude_home = tmp_path / ".claude"
    project = tmp_path / "lvcore"
    transcript_dir = claude_home / "projects" / re.sub(r"[^A-Za-z0-9_-]", "-", str(project.resolve()))
    transcript_dir.mkdir(parents=True)
    transcript = transcript_dir / "one.jsonl"
    transcript.write_text('{"type":"user","message":{"content":"hello"}}\n')
    config = tmp_path / "config.toml"
    config.write_text(
        f'operator_id = "ian"\nartifacts = "{artifacts}"\ndatabase = "{database}"\n'
        f'[projects.lvcore]\nroot = "{project}"\n'
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_home))
    record = make_record(source=str(transcript), source_session_id="one", project=ProjectProvenance(project_id="lvcore"))
    fake = FakeExtractor([record], project_id="lvcore")

    assert run(["--config", str(config), "import-sessions", "--source", "claude", "--project", "lvcore"], KeywordEmbedder(), fake) == 0
    assert fake.calls == 1
    capsys.readouterr()
    assert run(["--config", str(config), "import-sessions", "--source", "claude", "--project", "lvcore", "--resume"], KeywordEmbedder(), fake) == 0
    assert fake.calls == 1
    assert json.loads(capsys.readouterr().out)["eligible"] == 0


def test_import_sessions_cursor_uses_only_local_rows_and_keeps_them_unscoped(tmp_path, capsys):
    artifacts = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    cursor_db = tmp_path / "conversation-search.db"
    with sqlite3.connect(cursor_db) as connection:
        connection.execute("CREATE TABLE conversations (fts_rowid INTEGER PRIMARY KEY, id TEXT, title TEXT, source TEXT, updated_at REAL)")
        connection.execute("CREATE TABLE conversation_fts (title TEXT, body TEXT)")
        connection.executemany(
            "INSERT INTO conversations VALUES (?, ?, ?, ?, ?)",
            [(1, "local-one", "Local", "local", 1), (2, "cloud-copy", "Cloud", "cloud-cache", 2)],
        )
        connection.executemany(
            "INSERT INTO conversation_fts VALUES (?, ?)",
            [("Local", "local conversation"), ("Cloud", "duplicated cloud conversation")],
        )
    config = tmp_path / "config.toml"
    config.write_text(f'operator_id = "ian"\nartifacts = "{artifacts}"\ndatabase = "{database}"\n')

    assert run(["--config", str(config), "import-sessions", "--source", "cursor", "--cursor-database", str(cursor_db), "--dry-run"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["discovered"] == 1
    assert not (artifacts / ".source-snapshots").exists()


def test_import_sessions_rejects_project_filter_when_cursor_is_included(tmp_path, capsys):
    config = tmp_path / "config.toml"
    config.write_text(
        f'operator_id = "ian"\nartifacts = "{tmp_path / "artifacts"}"\ndatabase = "{tmp_path / "database"}"\n'
        f'[projects.lvcore]\nroot = "{tmp_path / "lvcore"}"\n'
    )

    assert run(["--config", str(config), "import-sessions", "--source", "all", "--project", "lvcore"]) == 3
    assert "cannot yet be assigned trusted project provenance" in capsys.readouterr().err


def test_import_sessions_resume_does_not_silently_process_a_changed_revision(tmp_path, capsys, monkeypatch):
    artifacts = tmp_path / "artifacts"
    claude_home = tmp_path / ".claude"
    project = tmp_path / "lvcore"
    transcript_dir = claude_home / "projects" / re.sub(r"[^A-Za-z0-9_-]", "-", str(project.resolve()))
    transcript_dir.mkdir(parents=True)
    transcript = transcript_dir / "one.jsonl"
    transcript.write_text('{"type":"user","message":{"content":"first"}}\n')
    config = tmp_path / "config.toml"
    config.write_text(
        f'operator_id = "ian"\nartifacts = "{artifacts}"\ndatabase = "{tmp_path / "database"}"\n'
        f'[projects.lvcore]\nroot = "{project}"\n'
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_home))
    failed = FakeExtractor(error=ExtractionPendingRetry("quota"), project_id="lvcore")
    run(["--config", str(config), "import-sessions", "--project", "lvcore"], KeywordEmbedder(), failed)
    transcript.write_text('{"type":"user","message":{"content":"changed"}}\n')
    capsys.readouterr()

    assert run(["--config", str(config), "import-sessions", "--project", "lvcore", "--resume"], KeywordEmbedder(), failed) == 0
    assert failed.calls == 1
    assert json.loads(capsys.readouterr().out)["changed_since_failure"] == 1


def test_ingest_skips_exact_normalized_duplicates_and_preserves_duplicate_link(tmp_path, capsys):
    artifacts = tmp_path / "artifacts"
    database = tmp_path / "database"
    project = ProjectProvenance(project_id="lvcore")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text('{"type":"user","message":{"content":"first"}}\n')
    second.write_text('{"type":"user","message":{"content":"second"}}\n')
    shared = dict(
        question="Where are the LVCore logs?",
        summary="  They are in CLOUDWATCH. ",
        resolution="Use the camerarelay log group.",
        project=project,
    )
    run(["extract-session", str(first), "--artifacts", str(artifacts)], extractor=FakeExtractor([
        make_record(**shared, source=str(first), source_session_id="first")
    ]))
    run(["extract-session", str(second), "--artifacts", str(artifacts)], extractor=FakeExtractor([
        make_record(**{**shared, "summary": "they are in cloudwatch."}, source=str(second), source_session_id="second")
    ]))
    records = load_active_episode_records(artifacts)
    first_id = next(record["id"] for record in records if record["source_id"] == "first")
    second_id = next(record["id"] for record in records if record["source_id"] == "second")
    capsys.readouterr()

    assert run(["ingest", "--artifacts", str(artifacts), "--database", str(database)], KeywordEmbedder()) == 0
    assert "Indexed 1 episode records" in capsys.readouterr().out
    run(["history", second_id, "--artifacts", str(artifacts)])
    state = json.loads(capsys.readouterr().out)
    assert state["duplicate_of"] == first_id
    assert state["duplicate_review_status"] == "exact"
    run(["duplicates", "--project-id", "lvcore", "--artifacts", str(artifacts)])
    assert json.loads(capsys.readouterr().out) == []


def test_ingest_flags_semantic_matches_for_review_without_removing_them(tmp_path, capsys):
    artifacts = tmp_path / "artifacts"
    database = tmp_path / "database"
    project = ProjectProvenance(project_id="lvcore")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text('{"type":"user","message":{"content":"first"}}\n')
    second.write_text('{"type":"user","message":{"content":"second"}}\n')
    run(["extract-session", str(first), "--artifacts", str(artifacts)], extractor=FakeExtractor([
        make_record(question="Why did RabbitMQ reconnect?", summary="A heartbeat timeout caused it.", source=str(first), source_session_id="first", project=project)
    ]))
    run(["extract-session", str(second), "--artifacts", str(artifacts)], extractor=FakeExtractor([
        make_record(question="What caused the RabbitMQ connection reset?", summary="The broker missed its heartbeat.", source=str(second), source_session_id="second", project=project)
    ]))
    records = load_active_episode_records(artifacts)
    first_id = next(record["id"] for record in records if record["source_id"] == "first")
    second_id = next(record["id"] for record in records if record["source_id"] == "second")
    capsys.readouterr()

    run(["ingest", "--artifacts", str(artifacts), "--database", str(database)], KeywordEmbedder())
    assert "Indexed 2 episode records" in capsys.readouterr().out
    run(["history", second_id, "--artifacts", str(artifacts)])
    state = json.loads(capsys.readouterr().out)
    assert state["duplicate_of"] is None
    assert state["duplicate_review_status"] == "possible"
    assert state["reinforces"][0]["record_id"] == first_id
    run(["duplicates", "--project-id", "lvcore", "--artifacts", str(artifacts)])
    queue = json.loads(capsys.readouterr().out)
    assert [item["record_id"] for item in queue] == [second_id]


def test_ingest_never_deduplicates_identical_episodes_across_projects(tmp_path, capsys):
    artifacts = tmp_path / "artifacts"
    database = tmp_path / "database"
    for name, project_id in (("first", "lvcore"), ("second", "beacon")):
        transcript = tmp_path / f"{name}.jsonl"
        transcript.write_text('{"type":"user","message":{"content":"same"}}\n')
        run(["extract-session", str(transcript), "--artifacts", str(artifacts)], extractor=FakeExtractor([
            make_record(question="Where are logs?", summary="CloudWatch", source=str(transcript), source_session_id=name, project=ProjectProvenance(project_id=project_id))
        ]))
    capsys.readouterr()

    run(["ingest", "--artifacts", str(artifacts), "--database", str(database)], KeywordEmbedder())

    assert "Indexed 2 episode records" in capsys.readouterr().out


def test_ingest_does_not_compare_records_without_trusted_project_provenance(tmp_path, capsys):
    artifacts = tmp_path / "artifacts"
    database = tmp_path / "database"
    for name in ("first", "second"):
        transcript = tmp_path / f"{name}.jsonl"
        transcript.write_text('{"type":"user","message":{"content":"same"}}\n')
        run(["extract-session", str(transcript), "--artifacts", str(artifacts)], extractor=FakeExtractor([
            make_record(question="Where are logs?", summary="CloudWatch", source=str(transcript), source_session_id=name)
        ]))
    capsys.readouterr()

    run(["ingest", "--artifacts", str(artifacts), "--database", str(database)], KeywordEmbedder())

    assert "Indexed 2 episode records" in capsys.readouterr().out


def test_health_check_writes_deterministic_review_report_without_changing_record_state(tmp_path, capsys):
    artifacts = tmp_path / "artifacts"
    transcript = tmp_path / "old.jsonl"
    transcript.write_text('{"type":"user","message":{"content":"old behavior"}}\n')
    record = make_record(
        question="How did the old deploy work?",
        summary="It copied jars directly.",
        timestamp="2020-01-01T00:00:00Z",
        temporal_scope="time_sensitive",
        evidence_location=None,
        source=str(transcript),
        source_session_id="old",
        project=ProjectProvenance(project_id="lvcore"),
    )
    run(["extract-session", str(transcript), "--artifacts", str(artifacts)], extractor=FakeExtractor([record]))
    record_id = load_active_episode_records(artifacts)[0]["id"]
    capsys.readouterr()

    assert run(["health-check", "--project-id", "lvcore", "--artifacts", str(artifacts)]) == 0

    result = json.loads(capsys.readouterr().out)
    categories = {finding["category"] for finding in result["findings"]}
    assert {"stale_record", "unsupported_claim"} <= categories
    assert Path(result["report_path"]).exists()
    assert read_state(artifacts, record_id)["verification_status"] == "unreviewed"


def test_health_check_accepts_grounded_cursor_findings_and_rejects_forged_record_links(tmp_path, capsys):
    artifacts = tmp_path / "artifacts"
    transcript = tmp_path / "session.jsonl"
    transcript.write_text('{"type":"user","message":{"content":"deployment"}}\n')
    record = make_record(
        question="How is deployment performed?",
        summary="The build uploads dependencies.",
        source=str(transcript),
        source_session_id="session",
        project=ProjectProvenance(project_id="lvcore"),
    )
    run(["extract-session", str(transcript), "--artifacts", str(artifacts)], extractor=FakeExtractor([record]))
    record_id = load_active_episode_records(artifacts)[0]["id"]
    checker = FakeHealthChecker([
        {
            "category": "contradiction",
            "title": "Deployment descriptions disagree",
            "explanation": "Review current deployment behavior.",
            "record_ids": [record_id],
            "recommended_action": "Verify against the repository.",
        },
        {
            "category": "suggested_article",
            "title": "Forged source",
            "explanation": "Bad link.",
            "record_ids": ["invented-record"],
            "recommended_action": "Do not keep this link.",
        },
    ])
    capsys.readouterr()

    assert run(["health-check", "--project-id", "lvcore", "--ai", "--artifacts", str(artifacts)], health_checker=checker) == 0

    result = json.loads(capsys.readouterr().out)
    assert checker.calls == 1
    contradictions = [finding for finding in result["findings"] if finding["category"] == "contradiction"]
    assert len(contradictions) == 1
    assert contradictions[0]["record_ids"] == [record_id]
    assert all("invented-record" not in finding["record_ids"] for finding in result["findings"])


def test_health_check_reports_unprocessed_sessions_and_project_scoped_empty_retrievals(tmp_path, capsys, monkeypatch):
    artifacts = tmp_path / "artifacts"
    database = tmp_path / "database"
    project = tmp_path / "lvcore"
    claude_home = tmp_path / ".claude"
    transcript_dir = claude_home / "projects" / re.sub(r"[^A-Za-z0-9_-]", "-", str(project.resolve()))
    transcript_dir.mkdir(parents=True)
    (transcript_dir / "never-imported.jsonl").write_text('{"type":"user","message":{"content":"new"}}\n')
    config = tmp_path / "config.toml"
    config.write_text(
        f'operator_id = "ian"\nartifacts = "{artifacts}"\ndatabase = "{database}"\n'
        f'[projects.lvcore]\nroot = "{project}"\n'
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_home))
    run(["--config", str(config), "search", "nothing", "--project-id", "lvcore"], KeywordEmbedder())
    capsys.readouterr()

    run(["--config", str(config), "health-check", "--project-id", "lvcore"])

    categories = {finding["category"] for finding in json.loads(capsys.readouterr().out)["findings"]}
    assert {"unprocessed_source", "weak_retrieval"} <= categories


def test_health_check_report_path_cannot_escape_through_project_id(tmp_path, capsys):
    artifacts = tmp_path / "artifacts"

    run(["health-check", "--project-id", "../../outside", "--artifacts", str(artifacts)])

    report = Path(json.loads(capsys.readouterr().out)["report_path"])
    assert report.is_relative_to(artifacts / "health-checks")
    assert ".." not in report.relative_to(artifacts / "health-checks").parts


def test_health_check_locally_flags_same_question_with_conflicting_answers(tmp_path, capsys):
    artifacts = tmp_path / "artifacts"
    for name, answer in (("old", "Deploy with Ant."), ("new", "Deploy with Gradle.")):
        transcript = tmp_path / f"{name}.jsonl"
        transcript.write_text('{"type":"user","message":{"content":"deploy"}}\n')
        run(["extract-session", str(transcript), "--artifacts", str(artifacts)], extractor=FakeExtractor([
            make_record(
                question="How do we deploy LVCore?", summary=answer, source=str(transcript),
                source_session_id=name, project=ProjectProvenance(project_id="lvcore"),
            )
        ]))
    capsys.readouterr()

    run(["health-check", "--project-id", "lvcore", "--artifacts", str(artifacts)])

    findings = json.loads(capsys.readouterr().out)["findings"]
    assert any(finding["category"] == "possible_contradiction" for finding in findings)


def test_health_check_treats_a_changed_session_revision_as_unprocessed(tmp_path, capsys, monkeypatch):
    artifacts = tmp_path / "artifacts"
    project = tmp_path / "lvcore"
    claude_home = tmp_path / ".claude"
    transcript_dir = claude_home / "projects" / re.sub(r"[^A-Za-z0-9_-]", "-", str(project.resolve()))
    transcript_dir.mkdir(parents=True)
    transcript = transcript_dir / "session.jsonl"
    transcript.write_text('{"type":"user","message":{"content":"old"}}\n')
    config = tmp_path / "config.toml"
    config.write_text(
        f'operator_id = "ian"\nartifacts = "{artifacts}"\n'
        f'[projects.lvcore]\nroot = "{project}"\n'
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_home))
    record = make_record(source=str(transcript), source_session_id="session", project=ProjectProvenance(project_id="lvcore"))
    run(["extract-session", str(transcript), "--artifacts", str(artifacts)], extractor=FakeExtractor([record]))
    transcript.write_text('{"type":"user","message":{"content":"new revision"}}\n')
    capsys.readouterr()

    run(["--config", str(config), "health-check", "--project-id", "lvcore"])

    findings = json.loads(capsys.readouterr().out)["findings"]
    assert any(finding["category"] == "unprocessed_source" for finding in findings)
