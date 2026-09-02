from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import lancedb

from .artifacts import read_active_hash
from .envconfig import config_from_env
from .jsonio import append_json_line
from .overlay import EXCLUDED_FROM_SEARCH, read_state
from .store import Embedder, TABLE_NAME

TRACE_LOG_NAME = "retrieval_traces.jsonl"


@dataclass(frozen=True)
class RetrievalScope:
    """Retrieval Scope (ADR-0004): trusted, application-supplied — never
    derived from the submitted prompt/query text, which is untrusted and
    must never be able to choose or widen its own retrieval scope."""

    project_id: str | None = None
    global_scope: bool = False

    @classmethod
    def from_env(cls) -> "RetrievalScope":
        return cls(
            project_id=os.getenv("SESSION_RAG_PROJECT_ID") or None,
            global_scope=os.getenv("SESSION_RAG_GLOBAL_SCOPE", "").lower() == "true",
        )

    def permits(self, candidate_project_id: str) -> bool:
        if self.global_scope:
            return True
        # CONTEXT.md's own rationale for Retrieval Scope is preventing
        # cross-workspace disclosure, not a blanket opt-in gate: with no
        # current project configured, there is no *other* workspace to leak
        # against, so an unscoped record matching an unscoped query is not
        # the disclosure this boundary exists to stop. A real project_id
        # never matches "" here, so cross-project leakage still can't happen.
        return candidate_project_id == (self.project_id or "")


@dataclass(frozen=True)
class RetrievalConfig:
    """All relevance-gate and ranking constants — provisional defaults, meant
    to be tuned later from real Retrieval Traces and an evaluation corpus,
    not asserted as final here."""

    vector_distance_ceiling: float = 1.0
    lexical_score_floor: float = 0.5
    max_results: int = 4
    unreviewed_durable_half_life_days: float = 90.0
    time_sensitive_half_life_days: float = 21.0
    unknown_age_days: float = 3650.0  # unknown timestamp — treat as very stale, never free-ranked
    fetch_multiplier: int = 3  # candidates fetched per method = max_results * fetch_multiplier

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        return config_from_env(cls)


def _fetch_candidates(database: Path, query: str, embedder: Embedder, fetch_limit: int) -> dict[str, dict]:
    """Raw candidates keyed by id, carrying vector_distance/lexical_score —
    None for whichever search method didn't surface that candidate."""

    candidates: dict[str, dict] = {}
    if not database.exists():
        return candidates
    connection = lancedb.connect(database)
    if TABLE_NAME not in connection.table_names():
        return candidates
    table = connection.open_table(TABLE_NAME)

    query_vector = embedder.embed([query])[0]
    for row in table.search(query_vector, vector_column_name="vector").limit(fetch_limit).to_list():
        entry = candidates.setdefault(row["id"], {**row, "vector_distance": None, "lexical_score": None})
        entry["vector_distance"] = row.get("_distance")
    try:
        fts_rows = table.search(query, query_type="fts", fts_columns="text").limit(fetch_limit).to_list()
    except Exception:
        fts_rows = []
    for row in fts_rows:
        entry = candidates.setdefault(row["id"], {**row, "vector_distance": None, "lexical_score": None})
        entry["lexical_score"] = row.get("_score")
    return candidates


def _normalized_lexical_score(raw_score: float, query: str) -> float:
    """Raw BM25-style FTS scores are unbounded and grow with query term
    count, so a fixed floor isn't comparable across queries — divide by the
    number of query terms to get a per-term average score instead."""

    term_count = max(len(query.split()), 1)
    return raw_score / term_count


def _distinctive_query_tokens(query: str, min_length: int = 6) -> list[str]:
    """Long tokens (paths, identifiers, error names) worth checking
    individually — covers a query that embeds one inside other words,
    e.g. 'how do I fix /src/foo/bar.py:42 KeyError'."""

    return [token for token in query.split() if len(token) >= min_length]


def _qualifies(candidate: dict, query: str, config: RetrievalConfig) -> tuple[bool, str | None]:
    """A candidate qualifies via a real vector-similarity floor or genuine
    lexical strength — mere FTS result-set membership does not qualify on
    its own; either a strong normalized FTS score or an exact phrase/token
    match is required."""

    distance = candidate.get("vector_distance")
    if distance is not None and distance <= config.vector_distance_ceiling:
        return True, "vector"
    score = candidate.get("lexical_score")
    if score is not None and _normalized_lexical_score(score, query) >= config.lexical_score_floor:
        return True, "lexical_score"
    text = candidate.get("text", "").lower()
    stripped = query.strip().lower()
    if stripped and stripped in text:
        return True, "exact_phrase"
    if any(token.lower() in text for token in _distinctive_query_tokens(query)):
        return True, "exact_phrase"
    return False, None


def _age_days(timestamp: str, config: RetrievalConfig) -> float:
    if not timestamp:
        return config.unknown_age_days
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return config.unknown_age_days
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max((datetime.now(timezone.utc) - parsed).total_seconds() / 86400, 0.0)


def _decay_multiplier(
    verification_status: str, temporal_scope: str | None, timestamp: str, config: RetrievalConfig
) -> float:
    """verified+durable: no decay, ever. unreviewed+durable: mild decay — an
    extraction-time 'durable' label must not grant unreviewed content
    permanent ranking strength. time_sensitive (either verification status):
    normal decay, regardless of durable/unreviewed."""

    if verification_status == "verified" and temporal_scope == "durable":
        return 1.0
    half_life = (
        config.unreviewed_durable_half_life_days
        if temporal_scope == "durable"
        else config.time_sensitive_half_life_days
    )
    return 0.5 ** (_age_days(timestamp, config) / half_life)


def _rank_score(candidate: dict) -> float:
    """Higher is better. A candidate matching on both vector and lexical
    signals ranks above one matching on either alone."""

    score = 0.0
    if candidate.get("vector_distance") is not None:
        score += 1.0 / (1.0 + candidate["vector_distance"])
    if candidate.get("lexical_score") is not None:
        score += candidate["lexical_score"]
    return score


def search(
    database: Path,
    artifacts_root: Path,
    query: str,
    embedder: Embedder,
    config: RetrievalConfig | None = None,
    scope: RetrievalScope | None = None,
) -> tuple[list[dict], dict]:
    """Relevance-gated, authority-ranked search. Order: (1) relevance gate on
    raw signals, (2) exclude rejected/superseded/non-active-revision/
    out-of-scope via the Record State Overlay, Active Revision, and
    Retrieval Scope, (3) rank survivors by the authority policy
    (verification_status + temporal_scope + recency, computed here — never
    stored, ADR-0002), (4) cap at max_results.

    `scope` is trusted, application-supplied context (see RetrievalScope) —
    it must never be derived from `query`, which is untrusted prompt text.

    Returns (results, trace) — the Retrieval Trace records every candidate's
    raw scores and why it was excluded or kept, for later threshold tuning.
    """

    config = config or RetrievalConfig.from_env()
    scope = scope or RetrievalScope.from_env()
    trace: dict = {"query": query, "candidates": []}
    if not query.strip():
        return [], trace

    raw_candidates = _fetch_candidates(database, query, embedder, config.max_results * config.fetch_multiplier)
    survivors: list[tuple[float, dict]] = []

    for candidate in raw_candidates.values():
        entry = {
            "id": candidate["id"],
            "vector_distance": candidate.get("vector_distance"),
            "lexical_score": candidate.get("lexical_score"),
        }
        qualifies, reason = _qualifies(candidate, query, config)
        entry["qualification_reason"] = reason
        if not qualifies:
            entry["excluded_reason"] = "below_relevance_gate"
            trace["candidates"].append(entry)
            continue

        state = read_state(artifacts_root, candidate["id"])
        if state["verification_status"] in EXCLUDED_FROM_SEARCH:
            entry["excluded_reason"] = f"verification_status:{state['verification_status']}"
            trace["candidates"].append(entry)
            continue

        active_hash = read_active_hash(
            artifacts_root, source_type=candidate["source_type"], source_id=candidate["source_id"]
        )
        if active_hash != candidate.get("source_hash"):
            entry["excluded_reason"] = "non_active_revision"
            trace["candidates"].append(entry)
            continue

        if not scope.permits(candidate.get("project_id", "")):
            entry["excluded_reason"] = "out_of_scope"
            trace["candidates"].append(entry)
            continue

        decay = _decay_multiplier(
            state["verification_status"], candidate.get("temporal_scope"), candidate.get("timestamp", ""), config
        )
        final_score = _rank_score(candidate) * decay
        entry["excluded_reason"] = None
        entry["verification_status"] = state["verification_status"]
        entry["decay_multiplier"] = decay
        entry["final_score"] = final_score
        trace["candidates"].append(entry)
        survivors.append((final_score, candidate))

    survivors.sort(key=lambda pair: pair[0], reverse=True)
    results = [candidate for _, candidate in survivors[: config.max_results]]
    trace["returned_ids"] = [result["id"] for result in results]

    # Persisted so later threshold tuning has real data to work from — but
    # never the query text itself, matching the "never store prompt text"
    # posture the hook otherwise holds to.
    persisted_trace = {key: value for key, value in trace.items() if key != "query"}
    persisted_trace["logged_at"] = datetime.now(timezone.utc).isoformat()
    append_json_line(artifacts_root / TRACE_LOG_NAME, persisted_trace)

    return results, trace


def purge_traces(artifacts_root: Path, record_ids: set[str]) -> None:
    """Scrub these record ids from the persisted Retrieval Trace log — part
    of forget's erasure guarantee. Rewrites each trace entry rather than
    dropping it outright, since one search's trace can name records from
    many sources, most of which aren't being forgotten."""

    path = artifacts_root / TRACE_LOG_NAME
    if not record_ids or not path.exists():
        return
    kept_lines = []
    for line in path.read_text().splitlines():
        if not line:
            continue
        entry = json.loads(line)
        entry["candidates"] = [c for c in entry.get("candidates", []) if c.get("id") not in record_ids]
        entry["returned_ids"] = [i for i in entry.get("returned_ids", []) if i not in record_ids]
        kept_lines.append(json.dumps(entry))
    path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""))
