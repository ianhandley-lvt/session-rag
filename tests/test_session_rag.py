import json
import shutil
from pathlib import Path

import pytest

from session_rag.artifacts import artifact_path, find_record, job_status_path, read_active_hash
from session_rag.cli import run
from session_rag.extractors.base import ExtractionBlocked, ExtractionError, ExtractionPendingRetry
from session_rag.hook import HookConfig, _INTRO, _estimate_tokens, _format_record, handle_user_prompt
from session_rag.retrieval import search

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
    assert run(["search", "rabbitmq timeout", "--database", str(database), "--artifacts", str(artifacts_dir)], embedder) == 0

    output = capsys.readouterr().out
    assert "heartbeat timeout caused the reconnect" in output
    assert "session-123" in output


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
    run(["search", "rabbitmq timeout", "--database", str(database), "--artifacts", str(artifacts_dir)], embedder)
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

    results, _trace = search(database, artifacts_dir, "rabbitmq", embedder)

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
    run(["search", "new question", "--database", str(database), "--artifacts", str(artifacts_dir)], embedder)
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
    run(["search", "rabbitmq postgres", "--database", str(database), "--artifacts", str(artifacts_dir)], embedder)
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
    metrics = [json.loads(line) for line in (artifacts_dir / "hook_metrics.jsonl").read_text().splitlines()]
    assert metrics[-1]["outcome"] == "timeout"
    assert "prompt" not in json.dumps(metrics[-1])


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
        {"hook_event_name": "UserPromptSubmit", "prompt": "a secret prompt about rabbitmq"}, database, artifacts_dir, embedder
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
    full_response, _trace = search(database, artifacts_dir, "rabbitmq", embedder)
    one_record_tokens = _estimate_tokens(_format_record(1, full_response[0]))
    budget = _estimate_tokens(_INTRO) + one_record_tokens + 1

    response = handle_user_prompt(
        {"hook_event_name": "UserPromptSubmit", "prompt": "rabbitmq"},
        database,
        artifacts_dir,
        embedder,
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
