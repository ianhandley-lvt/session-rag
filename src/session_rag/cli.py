from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import find_record, find_sources_by_project, forget_source, load_active_episode_records
from .embeddings import FastEmbedder
from .extractors import create_extractor
from .extractors.base import KnowledgeExtractor
from .hook import format_context, handle_user_prompt
from .overlay import (
    InvalidTransition,
    SupersedeRequiresReplacement,
    UnknownReplacementRecord,
    filter_retrievable,
    forget_records,
    read_state,
    reject,
    supersede,
    verify,
)
from .pipeline import run_extraction
from .retrieval import RetrievalScope
from .retrieval import purge_traces
from .retrieval import search as retrieval_search
from .store import Embedder, delete_by_source_id, index_episode_records


def _add_record_command_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("record_id")
    subparser.add_argument("--artifacts", type=Path, required=True)


def _add_scope_args(subparser: argparse.ArgumentParser) -> None:
    # Deliberately CLI/env-only — never derived from the query/prompt text
    # itself (ADR-0004). Unset means "fall back to SESSION_RAG_PROJECT_ID /
    # SESSION_RAG_GLOBAL_SCOPE" (RetrievalScope.from_env()).
    subparser.add_argument("--project-id")
    subparser.add_argument("--global-scope", action="store_true", default=None)


def _scope_from_args(args: argparse.Namespace) -> RetrievalScope | None:
    if args.project_id is None and args.global_scope is None:
        return None
    return RetrievalScope(project_id=args.project_id, global_scope=bool(args.global_scope))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="session-rag")
    commands = result.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--artifacts", type=Path, required=True)
    ingest.add_argument("--database", type=Path, required=True)
    extract = commands.add_parser("extract-session")
    extract.add_argument("transcript", type=Path)
    extract.add_argument("--artifacts", type=Path, required=True)
    extract.add_argument("--extractor", default="cursor", choices=["cursor"])
    extract.add_argument("--cursor-mode", choices=["ask", "plan"])
    extract.add_argument("--cursor-model")
    search_cmd = commands.add_parser("search")
    search_cmd.add_argument("query")
    search_cmd.add_argument("--database", type=Path, required=True)
    search_cmd.add_argument("--artifacts", type=Path, required=True)
    _add_scope_args(search_cmd)
    hook = commands.add_parser("hook")
    hook.add_argument("--database", type=Path, required=True)
    hook.add_argument("--artifacts", type=Path, required=True)
    _add_scope_args(hook)
    _add_record_command_args(commands.add_parser("verify"))
    _add_record_command_args(commands.add_parser("reject"))
    supersede_cmd = commands.add_parser("supersede")
    _add_record_command_args(supersede_cmd)
    supersede_cmd.add_argument("replacement_id")
    _add_record_command_args(commands.add_parser("history"))
    forget_cmd = commands.add_parser("forget")
    forget_cmd.add_argument("source_id", nargs="?")
    forget_cmd.add_argument("--project")
    forget_cmd.add_argument("--artifacts", type=Path, required=True)
    forget_cmd.add_argument("--database", type=Path, required=True)
    return result


def _forget_sources(artifacts_root: Path, database: Path, source_ids: list[str]) -> int:
    """Run the full forget sequence (artifact, overlay, trace, index) for
    each source — shared by single-source and project-wide forget so the
    erasure guarantee is enforced identically either way."""

    total = 0
    for source_id in source_ids:
        record_ids = forget_source(artifacts_root, source_id)
        forget_records(artifacts_root, record_ids)
        purge_traces(artifacts_root, set(record_ids))
        delete_by_source_id(database, source_id)
        total += len(record_ids)
    return total


def run(
    arguments: list[str] | None = None,
    embedder: Embedder | None = None,
    extractor: KnowledgeExtractor | None = None,
) -> int:
    args = parser().parse_args(arguments)
    if args.command == "ingest":
        selected_embedder = embedder or FastEmbedder()
        records = filter_retrievable(args.artifacts, load_active_episode_records(args.artifacts))
        count = index_episode_records(args.database, records, selected_embedder)
        print(f"Indexed {count} episode records in {args.database}")
    elif args.command == "extract-session":
        try:
            selected_extractor = extractor or create_extractor(
                args.extractor,
                cursor_mode=args.cursor_mode,
                cursor_model=args.cursor_model,
            )
        except ValueError as error:
            print(f"configuration error: {error}", file=sys.stderr)
            return 3

        outcome = run_extraction(selected_extractor, args.transcript, args.artifacts)

        if outcome.status == "blocked":
            print(f"blocked: {outcome.reason}", file=sys.stderr)
            return 2
        if outcome.status == "pending_retry":
            print(f"pending_retry: {outcome.reason}", file=sys.stderr)
            return 4
        if outcome.status == "failed":
            print(f"failed: {outcome.reason}", file=sys.stderr)
            return 1

        if outcome.orphaned_questions:
            print(
                f"note: {len(outcome.orphaned_questions)} record(s) from the previous revision have no "
                "obvious counterpart in this extraction — review for verification, rejection, or supersession:",
                file=sys.stderr,
            )
            for question in outcome.orphaned_questions:
                print(f"  - {question}", file=sys.stderr)

        envelope = json.loads(outcome.artifact_path.read_text())
        print(
            json.dumps(
                {"artifact_path": str(outcome.artifact_path), "records": envelope["episode_records"]},
                indent=2,
            )
        )
    elif args.command == "search":
        selected_embedder = embedder or FastEmbedder()
        results, _trace = retrieval_search(
            args.database, args.artifacts, args.query, selected_embedder, scope=_scope_from_args(args)
        )
        print(format_context(results) if results else "No relevant session memory found.")
    elif args.command == "hook":
        # Construction is deferred to inside handle_user_prompt's timed
        # daemon thread (embedder_factory) rather than built eagerly here —
        # otherwise model init/process startup would fall outside
        # retrieval_timeout_ms, defeating the fail-open guarantee. A
        # caller-supplied embedder (e.g. tests) bypasses the factory and is
        # used directly.
        scope = _scope_from_args(args)
        try:
            event = json.load(sys.stdin)
            result = handle_user_prompt(
                event,
                args.database,
                args.artifacts,
                embedder=embedder,
                embedder_factory=None if embedder else FastEmbedder,
                scope=scope,
            )
            print(json.dumps(result))
        except Exception:
            print("{}")
    elif args.command in {"verify", "reject", "supersede", "history"}:
        record = find_record(args.artifacts, args.record_id)
        if record is None:
            print(f"no such record: {args.record_id}", file=sys.stderr)
            return 1

        if args.command == "history":
            state = read_state(args.artifacts, args.record_id)
            print(json.dumps({**record, **state}, indent=2))
            return 0

        verb = {"verify": "verified", "reject": "rejected", "supersede": "superseded"}[args.command]
        try:
            if args.command == "verify":
                verify(args.artifacts, args.record_id)
            elif args.command == "reject":
                reject(args.artifacts, args.record_id)
            else:
                supersede(args.artifacts, args.record_id, args.replacement_id)
        except (InvalidTransition, SupersedeRequiresReplacement, UnknownReplacementRecord) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        print(f"{verb} {args.record_id}")
    elif args.command == "forget":
        if bool(args.source_id) == bool(args.project):
            print("forget requires exactly one of <source-id> or --project", file=sys.stderr)
            return 1
        if args.source_id:
            source_ids = [args.source_id]
            label = f"source {args.source_id}"
        else:
            source_ids = find_sources_by_project(args.artifacts, args.project)
            label = f"project {args.project}"
        total_records = _forget_sources(args.artifacts, args.database, source_ids)
        # Terminal-only, one-time — never written to a file, matching the
        # erasure guarantee (no record of the deletion itself is retained).
        print(f"forgot {total_records} record(s) across {len(source_ids)} source(s) for {label}")
    return 0


def main() -> None:
    raise SystemExit(run())
