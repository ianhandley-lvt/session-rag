from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .envconfig import ENV_PREFIX, LEGACY_ENV_PREFIX


class ConfigError(ValueError):
    """The application configuration exists but is invalid."""


@dataclass(frozen=True)
class ExtractorSettings:
    provider: str = "cursor"
    mode: str = "ask"
    model: str = "auto"
    max_sanitized_chars: int = 20_000


@dataclass(frozen=True)
class ProjectSettings:
    root: Path
    knowledge_base: Path | None = None


@dataclass(frozen=True)
class AppConfig:
    path: Path | None = None
    operator_id: str | None = None
    artifacts: Path | None = None
    database: Path | None = None
    extractor: ExtractorSettings = field(default_factory=ExtractorSettings)
    projects: dict[str, ProjectSettings] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedAppConfig:
    config_path: Path | None
    operator_id: str | None
    artifacts: Path | None
    database: Path | None
    extractor_provider: str
    cursor_mode: str
    cursor_model: str
    max_sanitized_chars: int
    project_id: str | None
    project_root: Path | None
    knowledge_base: Path | None


def default_config_path(environ: Mapping[str, str] | None = None) -> Path:
    env = environ if environ is not None else os.environ
    configured = env.get("MEMORY_CONFIG") or env.get("SESSION_RAG_CONFIG")
    if configured:
        return Path(configured).expanduser()
    xdg_home = env.get("XDG_CONFIG_HOME")
    base = Path(xdg_home).expanduser() if xdg_home else Path.home() / ".config"
    return base / "memory" / "config.toml"


def _legacy_config_path(environ: Mapping[str, str] | None = None) -> Path:
    env = environ if environ is not None else os.environ
    xdg_home = env.get("XDG_CONFIG_HOME")
    base = Path(xdg_home).expanduser() if xdg_home else Path.home() / ".config"
    return base / "session-rag" / "config.toml"


def _optional_path(value: object, *, field_name: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} must be a non-empty path string")
    return Path(value).expanduser()


def _string(value: object, *, field_name: str, default: str | None = None) -> str | None:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} must be a non-empty string")
    return value.strip()


def _integer(value: object, *, field_name: str, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{field_name} must be a positive integer")
    return value


def load_app_config(path: Path | None = None, *, environ: Mapping[str, str] | None = None) -> AppConfig:
    explicit_path = path is not None
    selected_path = path.expanduser() if path is not None else default_config_path(environ)
    env = environ if environ is not None else os.environ
    has_explicit_env_path = bool(env.get("MEMORY_CONFIG") or env.get("SESSION_RAG_CONFIG"))
    if not explicit_path and not has_explicit_env_path and not selected_path.exists():
        legacy_path = _legacy_config_path(environ)
        if legacy_path.exists():
            selected_path = legacy_path
    if not selected_path.exists():
        if explicit_path:
            raise ConfigError(f"config file does not exist: {selected_path}")
        return AppConfig(path=None)
    try:
        raw = tomllib.loads(selected_path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot read {selected_path}: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError(f"{selected_path} must contain a TOML table")

    extractor_raw = raw.get("extractor", {})
    if not isinstance(extractor_raw, dict):
        raise ConfigError("extractor must be a TOML table")
    extractor = ExtractorSettings(
        provider=_string(extractor_raw.get("provider"), field_name="extractor.provider", default="cursor"),
        mode=_string(extractor_raw.get("mode"), field_name="extractor.mode", default="ask"),
        model=_string(extractor_raw.get("model"), field_name="extractor.model", default="auto"),
        max_sanitized_chars=_integer(
            extractor_raw.get("max_sanitized_chars"),
            field_name="extractor.max_sanitized_chars",
            default=20_000,
        ),
    )
    if extractor.mode not in {"ask", "plan"}:
        raise ConfigError("extractor.mode must be 'ask' or 'plan'")

    projects_raw = raw.get("projects", {})
    if not isinstance(projects_raw, dict):
        raise ConfigError("projects must be a TOML table")
    projects: dict[str, ProjectSettings] = {}
    for project_id, project_raw in projects_raw.items():
        if not isinstance(project_raw, dict):
            raise ConfigError(f"projects.{project_id} must be a TOML table")
        root = _optional_path(project_raw.get("root"), field_name=f"projects.{project_id}.root")
        if root is None:
            raise ConfigError(f"projects.{project_id}.root is required")
        projects[project_id] = ProjectSettings(
            root=root,
            knowledge_base=_optional_path(
                project_raw.get("knowledge_base"), field_name=f"projects.{project_id}.knowledge_base"
            ),
        )

    return AppConfig(
        path=selected_path,
        operator_id=_string(raw.get("operator_id"), field_name="operator_id"),
        artifacts=_optional_path(raw.get("artifacts"), field_name="artifacts"),
        database=_optional_path(raw.get("database"), field_name="database"),
        extractor=extractor,
        projects=projects,
    )


def _project_for_cwd(config: AppConfig, cwd: Path) -> tuple[str | None, ProjectSettings | None]:
    matches: list[tuple[int, str, ProjectSettings]] = []
    resolved_cwd = cwd.expanduser().resolve()
    for project_id, project in config.projects.items():
        root = project.root.resolve()
        if resolved_cwd == root or resolved_cwd.is_relative_to(root):
            matches.append((len(root.parts), project_id, project))
    if not matches:
        return None, None
    _, project_id, project = max(matches, key=lambda match: match[0])
    return project_id, project


def _first(*values):
    return next((value for value in values if value is not None and value != ""), None)


def _env(env: Mapping[str, str], suffix: str) -> str | None:
    return env.get(ENV_PREFIX + suffix) or env.get(LEGACY_ENV_PREFIX + suffix)


def _resolved_path(value: object) -> Path | None:
    if value is None or value == "":
        return None
    return Path(str(value)).expanduser()


def resolve_app_config(
    config: AppConfig,
    *,
    cwd: Path,
    environ: Mapping[str, str] | None = None,
    explicit: Mapping[str, object] | None = None,
) -> ResolvedAppConfig:
    env = environ if environ is not None else os.environ
    cli = explicit or {}

    selected_project_id = _first(cli.get("project_id"), _env(env, "PROJECT_ID"))
    matched_id, matched_project = _project_for_cwd(config, cwd)
    if selected_project_id is None:
        selected_project_id = matched_id
    configured_project = config.projects.get(str(selected_project_id)) if selected_project_id else matched_project
    project_root = _resolved_path(
        _first(
            cli.get("project_root"),
            _env(env, "PROJECT_ROOT"),
            configured_project.root if configured_project else None,
        )
    )

    max_chars_raw = _first(
        cli.get("max_sanitized_chars"),
        _env(env, "MAX_SANITIZED_CHARS"),
        config.extractor.max_sanitized_chars,
    )
    try:
        max_chars = int(max_chars_raw)
    except (TypeError, ValueError) as error:
        raise ConfigError("max_sanitized_chars must be a positive integer") from error
    if max_chars <= 0:
        raise ConfigError("max_sanitized_chars must be a positive integer")

    return ResolvedAppConfig(
        config_path=config.path,
        operator_id=_first(cli.get("operator_id"), _env(env, "OPERATOR_ID"), config.operator_id),
        artifacts=_resolved_path(_first(cli.get("artifacts"), _env(env, "ARTIFACTS"), config.artifacts)),
        database=_resolved_path(_first(cli.get("database"), _env(env, "DATABASE"), config.database)),
        extractor_provider=str(
            _first(cli.get("extractor_provider"), _env(env, "EXTRACTOR"), config.extractor.provider)
        ),
        cursor_mode=str(_first(cli.get("cursor_mode"), _env(env, "CURSOR_MODE"), config.extractor.mode)),
        cursor_model=str(_first(cli.get("cursor_model"), _env(env, "CURSOR_MODEL"), config.extractor.model)),
        max_sanitized_chars=max_chars,
        project_id=str(selected_project_id) if selected_project_id else None,
        project_root=project_root,
        knowledge_base=configured_project.knowledge_base if configured_project else None,
    )
