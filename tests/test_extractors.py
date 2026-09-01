import json
import subprocess
from pathlib import Path

import pytest
from session_rag.extractors.cursor import CursorExtractor
from session_rag.extractors.base import ExtractionError


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
                        "author": "Ian",
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
    assert ["--mode", "ask"] == observed["command"][4:6]
    assert "--sandbox" in observed["command"]
    assert "--workspace" in observed["command"]
    assert "--trust" in observed["command"]
    assert "Why did RabbitMQ reconnect?" in observed["input"]
    assert str(transcript) not in observed["input"]


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
                        "author": None,
                        "timestamp": None,
                        "source": "/forged/path",
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
