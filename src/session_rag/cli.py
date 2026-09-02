from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import find_record, load_active_episode_records
from .embeddings import FastEmbedder
from .extractors import create_extractor
from .extractors.base import KnowledgeExtractor
from .hook import format_context, handle_user_prompt
from .overlay import (
    InvalidTransition,
    SupersedeRequiresReplacement,
    UnknownReplacementRecord,
    filter_retrievable,
    read_state,
    reject,
    supersede,
    verify,
)
from .pipeline import run_extraction
from .retrieval import search as retrieval_search
from .store import Embedder, index_episode_records


def _add_record_command_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("record_id")
    subparser.add_argument("--artifacts", type=Path, required=True)


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
    hook = commands.add_parser("hook")
    hook.add_argument("--database", type=Path, required=True)
    hook.add_argument("--artifacts", type=Path, required=True)
    _add_record_command_args(commands.add_parser("verify"))
    _add_record_command_args(commands.add_parser("reject"))
    supersede_cmd = commands.add_parser("supersede")
    _add_record_command_args(supersede_cmd)
    supersede_cmd.add_argument("replacement_id")
    _add_record_command_args(commands.add_parser("history"))
    return result


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
        results, _trace = retrieval_search(args.database, args.artifacts, args.query, selected_embedder)
        print(format_context(results) if results else "No relevant session memory found.")
    elif args.command == "hook":
        selected_embedder = embedder or FastEmbedder()
        try:
            event = json.load(sys.stdin)
            print(json.dumps(handle_user_prompt(event, args.database, args.artifacts, selected_embedder)))
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
    return 0


def main() -> None:
    raise SystemExit(run())
