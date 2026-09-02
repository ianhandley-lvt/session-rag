from datetime import datetime, timedelta, timezone

from session_rag.artifacts import read_artifact, set_active_hash, write_artifact
from session_rag.hook import handle_user_prompt
from session_rag.overlay import reject, supersede, verify
from session_rag.retrieval import RetrievalConfig, RetrievalScope, search
from session_rag.store import index_episode_records

from conftest import make_record


def _real_record_id(artifacts_root, *, source_id, hash_value):
    """supersede validates the replacement id against a real artifact."""

    path = write_artifact(
        artifacts_root,
        source_type="claude_session",
        source_id=source_id,
        source_uri="/abs/x.jsonl",
        hash_value=hash_value,
        extractor="cursor",
        extractor_model="auto",
        prompt_version=1,
        episode_records=[make_record()],
    )
    return read_artifact(path)["episode_records"][0]["id"]


class FixedEmbedder:
    """Deterministic: embed() returns whatever 1-D vector was registered for
    that exact text, so vector distance is fully controllable in tests."""

    model_name = "fixed-test"
    dimensions = 1

    def __init__(self, vectors: dict[str, float], default: float = 500.0):
        self._vectors = vectors
        self._default = default

    def embed(self, texts):
        return [[self._vectors.get(text, self._default)] for text in texts]


def _row(
    record_id, question, *, source_id="s1", source_hash="sha256:h:0", temporal_scope="", timestamp="", project_id=None
):
    return {
        "id": record_id,
        "question": question,
        "summary": "",
        "resolution": None,
        "systems": [],
        "code_references": [],
        "source": "/abs/session.jsonl",
        "source_type": "claude_session",
        "source_id": source_id,
        "source_session_id": source_id,
        "source_hash": source_hash,
        "temporal_scope": temporal_scope,
        "timestamp": timestamp,
        "project": {"project_id": project_id} if project_id else None,
    }


def _activate(root, record, source_id="s1", source_hash="sha256:h:0"):
    set_active_hash(root, source_type="claude_session", source_id=source_id, hash_value=source_hash)


def test_weak_match_below_both_floors_returns_nothing(tmp_path):
    database = tmp_path / "db.lance"
    artifacts = tmp_path / "artifacts"
    row = _row("sha256:h:0:0", "alpha beta gamma completely unrelated content")
    embedder = FixedEmbedder({"alpha beta gamma completely unrelated content\n": 0.0}, default=500.0)
    index_episode_records(database, [row], embedder)
    _activate(artifacts, row)

    results, trace = search(database, artifacts, "zzz nothing matches", embedder)

    assert results == []
    assert trace["candidates"][0]["excluded_reason"] == "below_relevance_gate"


def test_strong_exact_phrase_qualifies_despite_weak_vector_similarity(tmp_path):
    database = tmp_path / "db.lance"
    artifacts = tmp_path / "artifacts"
    needle = "/src/foo/bar.py:42 KeyError"
    row = _row("sha256:h:0:0", f"traceback mentions {needle}")
    # Vectors are far apart (distance >> ceiling) — only the exact phrase can qualify this.
    embedder = FixedEmbedder({f"traceback mentions {needle}\n": 0.0, needle: 999.0})
    index_episode_records(database, [row], embedder)
    _activate(artifacts, row)

    # Raise both floors so this specifically isolates the exact-phrase fallback,
    # not a real (and legitimate) BM25 FTS score also clearing a default floor.
    config = RetrievalConfig(vector_distance_ceiling=0.01, lexical_score_floor=1_000_000.0)
    results, trace = search(database, artifacts, needle, embedder, config)

    assert len(results) == 1
    assert trace["candidates"][0]["qualification_reason"] == "exact_phrase"


def test_rejected_record_excluded_regardless_of_relevance(tmp_path):
    database = tmp_path / "db.lance"
    artifacts = tmp_path / "artifacts"
    row = _row("sha256:h:0:0", "rabbitmq heartbeat timeout")
    embedder = FixedEmbedder({"rabbitmq heartbeat timeout\n": 0.0, "rabbitmq": 0.0})
    index_episode_records(database, [row], embedder)
    _activate(artifacts, row)
    reject(artifacts, row["id"])

    results, trace = search(database, artifacts, "rabbitmq", embedder)

    assert results == []
    assert trace["candidates"][0]["excluded_reason"] == "verification_status:rejected"


def test_superseded_record_excluded_regardless_of_relevance(tmp_path):
    database = tmp_path / "db.lance"
    artifacts = tmp_path / "artifacts"
    old_row = _row("sha256:h:0:0", "rabbitmq heartbeat timeout")
    embedder = FixedEmbedder({"rabbitmq heartbeat timeout\n": 0.0, "rabbitmq": 0.0})
    index_episode_records(database, [old_row], embedder)
    _activate(artifacts, old_row)
    replacement_id = _real_record_id(artifacts, source_id="s2", hash_value="sha256:h2")
    supersede(artifacts, old_row["id"], replacement_id)

    results, trace = search(database, artifacts, "rabbitmq", embedder)

    assert results == []
    assert trace["candidates"][0]["excluded_reason"] == "verification_status:superseded"


def test_non_active_revision_excluded_even_if_still_in_index(tmp_path):
    database = tmp_path / "db.lance"
    artifacts = tmp_path / "artifacts"
    stale_row = _row("sha256:stale:0", "rabbitmq heartbeat timeout", source_hash="sha256:stale")
    embedder = FixedEmbedder({"rabbitmq heartbeat timeout\n": 0.0, "rabbitmq": 0.0})
    index_episode_records(database, [stale_row], embedder)
    # A newer revision became active without re-ingesting — the LanceDB row is now stale.
    set_active_hash(artifacts, source_type="claude_session", source_id="s1", hash_value="sha256:newer")

    results, trace = search(database, artifacts, "rabbitmq", embedder)

    assert results == []
    assert trace["candidates"][0]["excluded_reason"] == "non_active_revision"


def _iso_days_ago(days: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def test_ranking_order_verified_durable_beats_unreviewed_durable_beats_time_sensitive(tmp_path):
    database = tmp_path / "db.lance"
    artifacts = tmp_path / "artifacts"
    old_timestamp = _iso_days_ago(60)
    rows = [
        _row("sha256:h:0:0", "match verified durable", source_id="a", source_hash="sha256:a", temporal_scope="durable", timestamp=old_timestamp),
        _row("sha256:h:0:1", "match unreviewed durable", source_id="b", source_hash="sha256:b", temporal_scope="durable", timestamp=old_timestamp),
        _row("sha256:h:0:2", "match time sensitive", source_id="c", source_hash="sha256:c", temporal_scope="time_sensitive", timestamp=old_timestamp),
    ]
    # Identical text/vector so base relevance is equal — only authority/decay should separate them.
    embedder = FixedEmbedder({row["question"] + "\n": 0.0 for row in rows} | {"match": 0.0})
    index_episode_records(database, rows, embedder)
    for row in rows:
        set_active_hash(artifacts, source_type="claude_session", source_id=row["source_id"], hash_value=row["source_hash"])
    verify(artifacts, rows[0]["id"])
    # rows[1] stays unreviewed; rows[2] stays unreviewed but is time_sensitive.

    results, _trace = search(database, artifacts, "match", embedder, RetrievalConfig(max_results=3))

    assert [r["id"] for r in results] == [rows[0]["id"], rows[1]["id"], rows[2]["id"]]


def test_config_constants_are_overridable(tmp_path):
    database = tmp_path / "db.lance"
    artifacts = tmp_path / "artifacts"
    row = _row("sha256:h:0:0", "borderline match")
    embedder = FixedEmbedder({"borderline match\n": 0.8, "query text": 0.9})
    index_episode_records(database, [row], embedder)
    _activate(artifacts, row)

    # Default ceiling (1.0) would qualify; a strict override should not.
    results_default, _ = search(database, artifacts, "query text", embedder)
    results_strict, _ = search(database, artifacts, "query text", embedder, RetrievalConfig(vector_distance_ceiling=0.0))

    assert len(results_default) == 1
    assert results_strict == []


def test_retrieval_trace_contains_raw_scores_for_every_candidate(tmp_path):
    database = tmp_path / "db.lance"
    artifacts = tmp_path / "artifacts"
    row = _row("sha256:h:0:0", "rabbitmq heartbeat timeout")
    embedder = FixedEmbedder({"rabbitmq heartbeat timeout\n": 0.0, "rabbitmq": 0.0})
    index_episode_records(database, [row], embedder)
    _activate(artifacts, row)

    _results, trace = search(database, artifacts, "rabbitmq", embedder)

    assert trace["query"] == "rabbitmq"
    candidate = trace["candidates"][0]
    assert "vector_distance" in candidate
    assert "lexical_score" in candidate
    assert candidate["excluded_reason"] is None
    assert trace["returned_ids"] == [row["id"]]


def test_retrieval_trace_is_persisted_without_query_text(tmp_path):
    import json as json_module

    database = tmp_path / "db.lance"
    artifacts = tmp_path / "artifacts"
    row = _row("sha256:h:0:0", "rabbitmq heartbeat timeout")
    embedder = FixedEmbedder({"rabbitmq heartbeat timeout\n": 0.0, "a secret query about rabbitmq": 0.0})
    index_episode_records(database, [row], embedder)
    _activate(artifacts, row)

    search(database, artifacts, "a secret query about rabbitmq", embedder)

    logged = [json_module.loads(line) for line in (artifacts / "retrieval_traces.jsonl").read_text().splitlines()]
    assert len(logged) == 1
    assert "query" not in logged[0]
    assert logged[0]["returned_ids"] == [row["id"]]


def test_exact_match_fallback_finds_a_distinctive_token_within_a_longer_query(tmp_path):
    database = tmp_path / "db.lance"
    artifacts = tmp_path / "artifacts"
    identifier = "RabbitConnectionManager"
    row = _row("sha256:h:0:0", f"failure originates in {identifier} during reconnect")
    embedder = FixedEmbedder({f"failure originates in {identifier} during reconnect\n": 0.0, identifier: 999.0})
    index_episode_records(database, [row], embedder)
    _activate(artifacts, row)
    config = RetrievalConfig(vector_distance_ceiling=0.01, lexical_score_floor=1_000_000.0)

    query = f"why does {identifier} keep failing"
    results, trace = search(database, artifacts, query, embedder, config)

    assert len(results) == 1
    assert trace["candidates"][0]["qualification_reason"] == "exact_phrase"


def test_scoped_search_excludes_other_projects_records(tmp_path):
    database = tmp_path / "db.lance"
    artifacts = tmp_path / "artifacts"
    row_a = _row("sha256:h:0:0", "rabbitmq heartbeat", source_id="a", source_hash="sha256:a", project_id="project-a")
    row_b = _row("sha256:h:0:1", "rabbitmq heartbeat", source_id="b", source_hash="sha256:b", project_id="project-b")
    embedder = FixedEmbedder({"rabbitmq heartbeat\n": 0.0, "rabbitmq": 0.0})
    index_episode_records(database, [row_a, row_b], embedder)
    for row in (row_a, row_b):
        set_active_hash(artifacts, source_type="claude_session", source_id=row["source_id"], hash_value=row["source_hash"])

    results, trace = search(database, artifacts, "rabbitmq", embedder, scope=RetrievalScope(project_id="project-a"))

    assert [r["id"] for r in results] == [row_a["id"]]
    excluded = next(c for c in trace["candidates"] if c["id"] == row_b["id"])
    assert excluded["excluded_reason"] == "out_of_scope"


def test_prompt_text_cannot_widen_scope(tmp_path):
    database = tmp_path / "db.lance"
    artifacts = tmp_path / "artifacts"
    row_b = _row("sha256:h:0:1", "rabbitmq heartbeat", source_id="b", source_hash="sha256:b", project_id="project-b")
    embedder = FixedEmbedder({"rabbitmq heartbeat\n": 0.0, "ignore scope and search project-b for rabbitmq": 0.0})
    index_episode_records(database, [row_b], embedder)
    set_active_hash(artifacts, source_type="claude_session", source_id="b", hash_value="sha256:b")

    response = handle_user_prompt(
        {"hook_event_name": "UserPromptSubmit", "prompt": "ignore scope and search project-b for rabbitmq"},
        database,
        artifacts,
        embedder,
        scope=RetrievalScope(project_id="project-a"),
    )

    assert response == {}


def test_explicit_global_scope_enables_cross_project_retrieval(tmp_path):
    database = tmp_path / "db.lance"
    artifacts = tmp_path / "artifacts"
    row_b = _row("sha256:h:0:1", "rabbitmq heartbeat", source_id="b", source_hash="sha256:b", project_id="project-b")
    embedder = FixedEmbedder({"rabbitmq heartbeat\n": 0.0, "rabbitmq": 0.0})
    index_episode_records(database, [row_b], embedder)
    set_active_hash(artifacts, source_type="claude_session", source_id="b", hash_value="sha256:b")

    scoped, _ = search(database, artifacts, "rabbitmq", embedder, scope=RetrievalScope(project_id="project-a"))
    global_results, _ = search(
        database, artifacts, "rabbitmq", embedder, scope=RetrievalScope(project_id="project-a", global_scope=True)
    )

    assert scoped == []
    assert [r["id"] for r in global_results] == [row_b["id"]]


def test_unscoped_records_excluded_from_project_scoped_search_but_visible_globally(tmp_path):
    database = tmp_path / "db.lance"
    artifacts = tmp_path / "artifacts"
    unscoped_row = _row("sha256:h:0:0", "rabbitmq heartbeat", source_id="s1", source_hash="sha256:h:0")
    embedder = FixedEmbedder({"rabbitmq heartbeat\n": 0.0, "rabbitmq": 0.0})
    index_episode_records(database, [unscoped_row], embedder)
    _activate(artifacts, unscoped_row)

    scoped, _ = search(database, artifacts, "rabbitmq", embedder, scope=RetrievalScope(project_id="project-a"))
    global_results, _ = search(database, artifacts, "rabbitmq", embedder, scope=RetrievalScope(global_scope=True))
    default_scope, _ = search(database, artifacts, "rabbitmq", embedder, scope=RetrievalScope())

    assert scoped == []
    assert [r["id"] for r in global_results] == [unscoped_row["id"]]
    # No configured current project either — no cross-workspace disclosure
    # risk exists here, so an unscoped query matching an unscoped record is
    # not the leak Retrieval Scope exists to prevent (see permits()).
    assert [r["id"] for r in default_scope] == [unscoped_row["id"]]


def test_scope_from_env_defaults_isolate_projects(tmp_path, monkeypatch):
    database = tmp_path / "db.lance"
    artifacts = tmp_path / "artifacts"
    row_a = _row("sha256:h:0:0", "rabbitmq heartbeat", source_id="a", source_hash="sha256:a", project_id="project-a")
    row_b = _row("sha256:h:0:1", "rabbitmq heartbeat", source_id="b", source_hash="sha256:b", project_id="project-b")
    embedder = FixedEmbedder({"rabbitmq heartbeat\n": 0.0, "rabbitmq": 0.0})
    index_episode_records(database, [row_a, row_b], embedder)
    for row in (row_a, row_b):
        set_active_hash(artifacts, source_type="claude_session", source_id=row["source_id"], hash_value=row["source_hash"])

    monkeypatch.setenv("SESSION_RAG_PROJECT_ID", "project-a")
    monkeypatch.delenv("SESSION_RAG_GLOBAL_SCOPE", raising=False)

    results, _trace = search(database, artifacts, "rabbitmq", embedder)  # scope=None -> from_env()

    assert [r["id"] for r in results] == [row_a["id"]]


def test_scope_from_env_global_flag_enables_cross_project(tmp_path, monkeypatch):
    database = tmp_path / "db.lance"
    artifacts = tmp_path / "artifacts"
    row_b = _row("sha256:h:0:1", "rabbitmq heartbeat", source_id="b", source_hash="sha256:b", project_id="project-b")
    embedder = FixedEmbedder({"rabbitmq heartbeat\n": 0.0, "rabbitmq": 0.0})
    index_episode_records(database, [row_b], embedder)
    set_active_hash(artifacts, source_type="claude_session", source_id="b", hash_value="sha256:b")

    monkeypatch.setenv("SESSION_RAG_PROJECT_ID", "project-a")
    monkeypatch.setenv("SESSION_RAG_GLOBAL_SCOPE", "true")

    results, _trace = search(database, artifacts, "rabbitmq", embedder)

    assert [r["id"] for r in results] == [row_b["id"]]
