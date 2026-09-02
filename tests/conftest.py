import pytest

from session_rag.extractors.base import StructuredRecord


@pytest.fixture(autouse=True)
def isolate_personal_config(tmp_path, monkeypatch):
    """Tests never inherit the operator's real ~/.config/session-rag file."""

    monkeypatch.setenv("SESSION_RAG_CONFIG", str(tmp_path / "missing-config.toml"))


def make_record(**overrides) -> StructuredRecord:
    defaults = dict(
        question="Why?",
        summary="Because.",
        source="/abs/session.jsonl",
        source_session_id="session",
        source_type="claude_session",
        operator_id="ian",
        prompt_version=1,
    )
    return StructuredRecord(**{**defaults, **overrides})
