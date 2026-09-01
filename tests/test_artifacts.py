import json
from pathlib import Path

from session_rag.artifacts import artifact_path, source_hash, write_artifact

from conftest import make_record


def test_source_hash_is_stable_for_unchanged_content(tmp_path):
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("same content")

    assert source_hash(transcript) == source_hash(transcript)


def test_source_hash_changes_when_content_changes(tmp_path):
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("content A")
    hash_a = source_hash(transcript)

    transcript.write_text("content B")
    hash_b = source_hash(transcript)

    assert hash_a != hash_b


def test_write_artifact_creates_expected_path(tmp_path):
    path = write_artifact(
        tmp_path,
        source_type="claude_session",
        source_id="session-123",
        source_uri="/abs/session-123.jsonl",
        hash_value="sha256:deadbeef",
        extractor="cursor",
        extractor_model="auto",
        prompt_version=1,
        episode_records=[make_record()],
    )

    assert path == tmp_path / "claude_session" / "session-123" / "sha256-deadbeef.json"
    assert path.exists()


def test_write_artifact_envelope_contains_provenance_and_records(tmp_path):
    record = make_record(question="Why did it break?")
    path = write_artifact(
        tmp_path,
        source_type="claude_session",
        source_id="session-123",
        source_uri="/abs/session-123.jsonl",
        hash_value="sha256:deadbeef",
        extractor="cursor",
        extractor_model="gemini-3.7-flash-low",
        prompt_version=2,
        episode_records=[record],
    )

    envelope = json.loads(path.read_text())

    assert envelope["schema_version"] == 1
    assert envelope["source_type"] == "claude_session"
    assert envelope["source_id"] == "session-123"
    assert envelope["source_hash"] == "sha256:deadbeef"
    assert envelope["extractor"] == "cursor"
    assert envelope["extractor_model"] == "gemini-3.7-flash-low"
    assert envelope["prompt_version"] == 2
    assert "extracted_at" in envelope
    assert len(envelope["episode_records"]) == 1
    assert envelope["episode_records"][0]["question"] == "Why did it break?"
    assert envelope["episode_records"][0]["id"]


def test_write_artifact_assigns_stable_deterministic_ids(tmp_path):
    records = [make_record(question="Q1"), make_record(question="Q2")]

    path = write_artifact(
        tmp_path,
        source_type="claude_session",
        source_id="session-123",
        source_uri="/abs/session-123.jsonl",
        hash_value="sha256:deadbeef",
        extractor="cursor",
        extractor_model="auto",
        prompt_version=1,
        episode_records=records,
    )

    envelope = json.loads(path.read_text())
    ids = [item["id"] for item in envelope["episode_records"]]

    assert ids[0] != ids[1]
    assert all(id_.startswith("sha256:deadbeef:") for id_ in ids)


def test_write_artifact_is_a_noop_for_an_existing_hash(tmp_path):
    # Artifacts are immutable: a second write for the same (source_id, hash)
    # must not replace content, even if the caller passes different records
    # (e.g. non-deterministic re-extraction) — that would silently reassign
    # an existing Episode Record's id to different content.
    common = dict(
        tmp_path=tmp_path,
        source_type="claude_session",
        source_id="session-123",
        source_uri="/abs/session-123.jsonl",
        hash_value="sha256:deadbeef",
        extractor="cursor",
        extractor_model="auto",
        prompt_version=1,
    )

    path_first = write_artifact(common.pop("tmp_path"), episode_records=[make_record(question="Original?")], **common)
    written_at_first = path_first.stat().st_mtime_ns
    path_second = write_artifact(tmp_path, episode_records=[make_record(question="Updated?")], **common)

    assert path_first == path_second
    assert path_second.stat().st_mtime_ns == written_at_first
    files = list((tmp_path / "claude_session" / "session-123").glob("*.json"))
    assert len(files) == 1
    envelope = json.loads(path_second.read_text())
    assert envelope["episode_records"][0]["question"] == "Original?"


def test_write_artifact_creates_separate_file_for_changed_hash(tmp_path):
    kwargs = dict(
        source_type="claude_session",
        source_id="session-123",
        source_uri="/abs/session-123.jsonl",
        extractor="cursor",
        extractor_model="auto",
        prompt_version=1,
    )

    path_old = write_artifact(tmp_path, hash_value="sha256:old", episode_records=[make_record()], **kwargs)
    path_new = write_artifact(tmp_path, hash_value="sha256:new", episode_records=[make_record()], **kwargs)

    assert path_old != path_new
    assert path_old.exists()
    assert path_new.exists()


def test_write_artifact_path_computed_correctly():
    path = artifact_path(
        Path("/root"),
        source_type="claude_session",
        source_id="session-123",
        hash_value="sha256:abc",
    )

    assert path == Path("/root/claude_session/session-123/sha256-abc.json")
