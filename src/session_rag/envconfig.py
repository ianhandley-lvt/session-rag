from __future__ import annotations

import os
from dataclasses import fields

ENV_PREFIX = "SESSION_RAG_"


def config_from_env(cls):
    """Build a frozen dataclass config from SESSION_RAG_<FIELD_NAME> env
    vars, falling back to each field's own default. Shared by every
    provisional, tunable-constants config (RetrievalConfig, HookConfig, ...)
    so env-parsing semantics live in exactly one place."""

    defaults = cls()
    overrides = {}
    for field in fields(cls):
        raw = os.getenv(ENV_PREFIX + field.name.upper())
        if raw is None:
            continue
        overrides[field.name] = int(raw) if field.type == "int" else float(raw)
    return cls(**{**{f.name: getattr(defaults, f.name) for f in fields(cls)}, **overrides})
