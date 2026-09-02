import pytest

from session_rag.artifacts import read_artifact, write_artifact
from session_rag.overlay import (
    InvalidTransition,
    SupersedeRequiresReplacement,
    UnknownReplacementRecord,
    filter_retrievable,
    read_state,
    reject,
    supersede,
    verify,
)

from conftest import make_record


def _real_record_id(root, hash_value="sha256:real"):
    """supersede validates the replacement id against a real artifact — only
    the replacement needs to exist, not the record being transitioned."""

    path = write_artifact(
        root,
        source_type="claude_session",
        source_id="session-1",
        source_uri="/abs/session-1.jsonl",
        hash_value=hash_value,
        extractor="cursor",
        extractor_model="auto",
        prompt_version=1,
        episode_records=[make_record()],
    )
    return read_artifact(path)["episode_records"][0]["id"]


def test_read_state_defaults_to_unreviewed(tmp_path):
    assert read_state(tmp_path, "sha256:abc:0") == {"verification_status": "unreviewed", "superseded_by": None}


def test_verify_from_unreviewed(tmp_path):
    verify(tmp_path, "sha256:abc:0")

    assert read_state(tmp_path, "sha256:abc:0")["verification_status"] == "verified"


def test_reject_from_verified(tmp_path):
    verify(tmp_path, "sha256:abc:0")
    reject(tmp_path, "sha256:abc:0")

    assert read_state(tmp_path, "sha256:abc:0")["verification_status"] == "rejected"


def test_cannot_verify_a_rejected_record(tmp_path):
    reject(tmp_path, "sha256:abc:0")

    with pytest.raises(InvalidTransition):
        verify(tmp_path, "sha256:abc:0")


def test_cannot_reject_a_superseded_record(tmp_path):
    replacement_id = _real_record_id(tmp_path)
    supersede(tmp_path, "sha256:abc:0", replacement_id)

    with pytest.raises(InvalidTransition):
        reject(tmp_path, "sha256:abc:0")


def test_supersede_without_replacement_id_raises(tmp_path):
    with pytest.raises(SupersedeRequiresReplacement):
        supersede(tmp_path, "sha256:abc:0", None)
    with pytest.raises(SupersedeRequiresReplacement):
        supersede(tmp_path, "sha256:abc:0", "")


def test_supersede_rejects_a_nonexistent_replacement_id(tmp_path):
    with pytest.raises(UnknownReplacementRecord):
        supersede(tmp_path, "sha256:abc:0", "sha256:doesnotexist:0")


def test_supersede_records_replacement_link(tmp_path):
    replacement_id = _real_record_id(tmp_path)
    supersede(tmp_path, "sha256:abc:0", replacement_id)

    state = read_state(tmp_path, "sha256:abc:0")
    assert state["verification_status"] == "superseded"
    assert state["superseded_by"] == replacement_id


def test_filter_retrievable_drops_rejected_and_superseded(tmp_path):
    replacement_id = _real_record_id(tmp_path)
    records = [{"id": "sha256:a:0"}, {"id": "sha256:b:0"}, {"id": replacement_id}]
    reject(tmp_path, "sha256:a:0")
    supersede(tmp_path, "sha256:b:0", replacement_id)

    kept = filter_retrievable(tmp_path, records)

    assert kept == [{"id": replacement_id}]
