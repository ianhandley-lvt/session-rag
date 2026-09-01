from session_rag.extractors.base import StructuredRecord


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
