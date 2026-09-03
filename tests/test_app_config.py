from pathlib import Path

import pytest

from session_rag.app_config import ConfigError, load_app_config, resolve_app_config


def write_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
operator_id = "config-operator"
artifacts = "~/configured-artifacts"
database = "~/configured-lancedb"

[extractor]
provider = "cursor"
mode = "plan"
model = "configured-model"
max_sanitized_chars = 123456

[projects.lvcore]
root = "/work/lvcore"
knowledge_base = "/knowledge/lvcore/Wiki"
""".strip()
        + "\n"
    )


def test_loads_default_config_from_xdg_config_home(tmp_path, monkeypatch):
    config_path = tmp_path / "xdg" / "memory" / "config.toml"
    write_config(config_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("MEMORY_CONFIG", raising=False)
    monkeypatch.delenv("SESSION_RAG_CONFIG", raising=False)

    config = load_app_config()

    assert config.path == config_path
    assert config.operator_id == "config-operator"
    assert config.extractor.max_sanitized_chars == 123456
    assert config.projects["lvcore"].root == Path("/work/lvcore")


def test_memory_environment_overrides_legacy_session_rag_environment(tmp_path):
    config_path = tmp_path / "config.toml"
    write_config(config_path)
    config = load_app_config(config_path)

    resolved = resolve_app_config(
        config,
        cwd=Path("/work/lvcore"),
        environ={
            "MEMORY_OPERATOR_ID": "memory-operator",
            "SESSION_RAG_OPERATOR_ID": "legacy-operator",
        },
    )

    assert resolved.operator_id == "memory-operator"


def test_legacy_config_is_loaded_when_canonical_config_does_not_exist(tmp_path, monkeypatch):
    legacy = tmp_path / "xdg" / "session-rag" / "config.toml"
    write_config(legacy)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("MEMORY_CONFIG", raising=False)
    monkeypatch.delenv("SESSION_RAG_CONFIG", raising=False)

    config = load_app_config()

    assert config.path == legacy


def test_explicit_missing_config_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="config file does not exist"):
        load_app_config(tmp_path / "missing.toml")


def test_current_directory_selects_the_most_specific_configured_project(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[projects.work]
root = "/work"

[projects.lvcore]
root = "/work/lvcore"
""".strip()
        + "\n"
    )
    config = load_app_config(config_path)

    resolved = resolve_app_config(config, cwd=Path("/work/lvcore/src"), environ={})

    assert resolved.project_id == "lvcore"
    assert resolved.project_root == Path("/work/lvcore")


def test_environment_overrides_config_values(tmp_path):
    config_path = tmp_path / "config.toml"
    write_config(config_path)
    config = load_app_config(config_path)

    resolved = resolve_app_config(
        config,
        cwd=Path("/work/lvcore"),
        environ={
            "SESSION_RAG_OPERATOR_ID": "env-operator",
            "SESSION_RAG_ARTIFACTS": "/env/artifacts",
            "SESSION_RAG_DATABASE": "/env/database",
            "SESSION_RAG_CURSOR_MODE": "ask",
            "SESSION_RAG_CURSOR_MODEL": "env-model",
            "SESSION_RAG_MAX_SANITIZED_CHARS": "654321",
        },
    )

    assert resolved.operator_id == "env-operator"
    assert resolved.artifacts == Path("/env/artifacts")
    assert resolved.database == Path("/env/database")
    assert resolved.cursor_mode == "ask"
    assert resolved.cursor_model == "env-model"
    assert resolved.max_sanitized_chars == 654321


def test_explicit_values_override_environment_and_config(tmp_path):
    config_path = tmp_path / "config.toml"
    write_config(config_path)
    config = load_app_config(config_path)

    resolved = resolve_app_config(
        config,
        cwd=Path("/work/lvcore"),
        environ={"SESSION_RAG_OPERATOR_ID": "env-operator"},
        explicit={
            "operator_id": "cli-operator",
            "artifacts": "/cli/artifacts",
            "database": "/cli/database",
            "project_id": "cli-project",
            "project_root": "/cli/project",
            "cursor_mode": "ask",
            "cursor_model": "cli-model",
            "max_sanitized_chars": 777,
        },
    )

    assert resolved.operator_id == "cli-operator"
    assert resolved.artifacts == Path("/cli/artifacts")
    assert resolved.database == Path("/cli/database")
    assert resolved.project_id == "cli-project"
    assert resolved.project_root == Path("/cli/project")
    assert resolved.cursor_model == "cli-model"
    assert resolved.max_sanitized_chars == 777
