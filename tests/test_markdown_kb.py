import json
from pathlib import Path

from session_rag.artifacts import artifact_path, read_active_hash
from session_rag.cli import run
from session_rag.markdown_kb import markdown_source_id


class KeywordEmbedder:
    dimensions = 2
    model_name = "keyword-test"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [float("rabbitmq" in text.lower()), float("postgres" in text.lower())]
            for text in texts
        ]


def _write_article(wiki: Path, body: str) -> Path:
    wiki.mkdir(parents=True, exist_ok=True)
    article = wiki / "lvcore-observability.md"
    article.write_text(body)
    return article


def _import_args(wiki: Path, artifacts: Path) -> list[str]:
    return [
        "import-markdown-kb",
        str(wiki),
        "--knowledge-base-id",
        "lvcore",
        "--project-id",
        "lvcore",
        "--project-root",
        "/work/lvcore",
        "--operator-id",
        "ian",
        "--artifacts",
        str(artifacts),
    ]


def _source_id(wiki: Path, filename: str = "lvcore-observability.md") -> str:
    return markdown_source_id("lvcore", wiki, wiki / filename)


def test_cli_imports_markdown_articles_as_active_artifacts(tmp_path, capsys):
    wiki = tmp_path / "lvcore_kb" / "Wiki"
    artifacts = tmp_path / "artifacts"
    article = _write_article(
        wiki,
        """# LVCore Observability

**Status:** established
**Last updated:** 2026-08-28
**Sources:** [[observability-conventions]], [[log-runbook]]

## Summary

LVCore uses structured logging and OpenTelemetry.

## Body

### Finding logs

Find the EC2 instance ID, then open the matching CloudWatch log stream.

### RabbitMQ alerts

RabbitMQ heartbeat alerts identify stalled consumers.

## Related

- [[lvcore-architecture]]
""",
    )

    assert run(_import_args(wiki, artifacts)) == 0

    output = json.loads(capsys.readouterr().out)
    assert output == {"articles": 1, "records": 3, "activated": 1, "unchanged": 0}
    source_id = _source_id(wiki)
    active_hash = read_active_hash(artifacts, source_type="markdown_knowledge_base", source_id=source_id)
    path = artifact_path(
        artifacts,
        source_type="markdown_knowledge_base",
        source_id=source_id,
        hash_value=active_hash,
    )
    envelope = json.loads(path.read_text())
    assert envelope["source_uri"] == str(article.resolve())
    assert envelope["extractor"] == "markdown"
    assert envelope["extractor_model"] == "deterministic"
    records = envelope["episode_records"]
    assert [record["question"] for record in records] == [
        "LVCore Observability: Summary",
        "LVCore Observability: Body > Finding logs",
        "LVCore Observability: Body > RabbitMQ alerts",
    ]
    assert records[0]["source_type"] == "markdown_knowledge_base"
    assert records[0]["project"]["project_id"] == "lvcore"
    assert records[0]["project"]["project_root"] == "/work/lvcore"
    assert records[0]["operator_id"] == "ian"
    assert records[0]["timestamp"].startswith("2026-08-28")
    assert records[0]["document_status"] == "established"
    assert records[0]["source_references"] == ["observability-conventions", "log-runbook"]
    assert records[1]["evidence_location"] == {
        "identifier": "body/finding-logs",
        "preserved_text": "Find the EC2 instance ID, then open the matching CloudWatch log stream.",
    }


def test_cli_markdown_import_is_no_op_until_article_changes(tmp_path, capsys):
    wiki = tmp_path / "lvcore_kb" / "Wiki"
    artifacts = tmp_path / "artifacts"
    article = _write_article(wiki, "# LVCore\n\n## Summary\n\nOriginal summary.\n")

    run(_import_args(wiki, artifacts))
    first = json.loads(capsys.readouterr().out)
    run(_import_args(wiki, artifacts))
    second = json.loads(capsys.readouterr().out)

    assert first["activated"] == 1
    assert second["unchanged"] == 1
    source_dir = artifacts / "markdown_knowledge_base" / _source_id(wiki)
    assert len(list(source_dir.glob("sha256-*.json"))) == 1

    article.write_text("# LVCore\n\n## Summary\n\nUpdated summary.\n")
    run(_import_args(wiki, artifacts))
    third = json.loads(capsys.readouterr().out)

    assert third["activated"] == 1
    assert len(list(source_dir.glob("sha256-*.json"))) == 2


def test_cli_markdown_import_skips_navigation_files(tmp_path, capsys):
    wiki = tmp_path / "lvcore_kb" / "Wiki"
    artifacts = tmp_path / "artifacts"
    _write_article(wiki, "# LVCore\n\n## Summary\n\nUseful article.\n")
    (wiki / "INDEX.md").write_text("# Index\n\n- [[lvcore-observability]]\n")
    (wiki / "QUESTIONS.md").write_text("# Questions\n\n- What remains unknown?\n")

    run(_import_args(wiki, artifacts))

    output = json.loads(capsys.readouterr().out)
    assert output["articles"] == 1
    assert not (artifacts / "markdown_knowledge_base" / _source_id(wiki, "INDEX.md")).exists()
    assert not (artifacts / "markdown_knowledge_base" / _source_id(wiki, "QUESTIONS.md")).exists()


def test_markdown_source_ids_do_not_collide_after_slugging(tmp_path, capsys):
    wiki = tmp_path / "lvcore_kb" / "Wiki"
    artifacts = tmp_path / "artifacts"
    wiki.mkdir(parents=True)
    first = wiki / "foo bar.md"
    second = wiki / "foo-bar.md"
    first.write_text("# First\n\n## Summary\n\nFirst article.\n")
    second.write_text("# Second\n\n## Summary\n\nSecond article.\n")

    run(_import_args(wiki, artifacts))

    assert json.loads(capsys.readouterr().out)["articles"] == 2
    assert markdown_source_id("lvcore", wiki, first) != markdown_source_id("lvcore", wiki, second)
    source_dirs = list((artifacts / "markdown_knowledge_base").iterdir())
    assert len(source_dirs) == 2
    assert all((source_dir / "active.json").exists() for source_dir in source_dirs)


def test_markdown_records_are_searchable_with_project_scope_and_citations(tmp_path, capsys):
    wiki = tmp_path / "lvcore_kb" / "Wiki"
    artifacts = tmp_path / "artifacts"
    database = tmp_path / "memory.lance"
    _write_article(
        wiki,
        "# LVCore Messaging\n\n## RabbitMQ recovery\n\nRabbitMQ heartbeat alerts identify stalled consumers.\n",
    )
    embedder = KeywordEmbedder()

    run(_import_args(wiki, artifacts))
    capsys.readouterr()
    run(["ingest", "--artifacts", str(artifacts), "--database", str(database)], embedder)
    capsys.readouterr()
    run(
        [
            "search",
            "rabbitmq heartbeat",
            "--database",
            str(database),
            "--artifacts",
            str(artifacts),
            "--project-id",
            "lvcore",
        ],
        embedder,
    )

    output = capsys.readouterr().out
    assert "RabbitMQ heartbeat alerts identify stalled consumers" in output
    assert f"markdown_knowledge_base/{_source_id(wiki)}" in output
    assert "evidence rabbitmq-recovery" in output
    assert "Underlying sources:" not in output


def test_markdown_sections_split_at_paragraph_boundaries_to_fit_injection_budget(tmp_path, capsys):
    wiki = tmp_path / "lvcore_kb" / "Wiki"
    artifacts = tmp_path / "artifacts"
    paragraph = "RabbitMQ consumers need heartbeat monitoring. " * 45
    _write_article(
        wiki,
        "# LVCore Messaging\n\n## Recovery\n\n" + paragraph + "\n\n" + paragraph + "\n",
    )

    run(_import_args(wiki, artifacts))

    source_id = _source_id(wiki)
    active_hash = read_active_hash(artifacts, source_type="markdown_knowledge_base", source_id=source_id)
    path = artifact_path(
        artifacts,
        source_type="markdown_knowledge_base",
        source_id=source_id,
        hash_value=active_hash,
    )
    records = json.loads(path.read_text())["episode_records"]
    assert len(records) == 2
    assert records[0]["evidence_location"]["identifier"] == "recovery/part-1"
    assert records[1]["evidence_location"]["identifier"] == "recovery/part-2"
    assert all(len(record["summary"]) <= 3_000 for record in records)


def test_markdown_oversized_indivisible_paragraph_is_blocked(tmp_path, capsys):
    wiki = tmp_path / "lvcore_kb" / "Wiki"
    artifacts = tmp_path / "artifacts"
    _write_article(wiki, "# LVCore\n\n## Generated token\n\n" + "x" * 3_001 + "\n")

    exit_code = run(_import_args(wiki, artifacts))

    assert exit_code == 2
    assert "blocked" in capsys.readouterr().err
