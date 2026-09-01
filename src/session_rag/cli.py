from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .embeddings import FastEmbedder
from .extractors import create_extractor
from .extractors.base import KnowledgeExtractor
from .hook import format_context, handle_user_prompt
from .pipeline import run_extraction
from .store import Embedder, index_memories, search_memories
from .transcripts import load_sessions


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="session-rag")
    commands = result.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest-sessions")
    ingest.add_argument("directory", type=Path)
    ingest.add_argument("--database", type=Path, required=True)
    extract = commands.add_parser("extract-session")
    extract.add_argument("transcript", type=Path)
    extract.add_argument("--artifacts", type=Path, required=True)
    extract.add_argument("--extractor", default="cursor", choices=["cursor"])
    extract.add_argument("--cursor-mode", choices=["ask", "plan"])
    extract.add_argument("--cursor-model")
    search = commands.add_parser("search")
    search.add_argument("query")
    search.add_argument("--database", type=Path, required=True)
    hook = commands.add_parser("hook")
    hook.add_argument("--database", type=Path, required=True)
    return result


def run(
    arguments: list[str] | None = None,
    embedder: Embedder | None = None,
    extractor: KnowledgeExtractor | None = None,
) -> int:
    args = parser().parse_args(arguments)
    if args.command == "ingest-sessions":
        selected_embedder = embedder or FastEmbedder()
        count = index_memories(args.database, load_sessions(args.directory), selected_embedder)
        print(f"Indexed {count} session memories in {args.database}")
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
        results = search_memories(args.database, args.query, selected_embedder)
        print(format_context(results) if results else "No relevant session memory found.")
    elif args.command == "hook":
        selected_embedder = embedder or FastEmbedder()
        try:
            event = json.load(sys.stdin)
            print(json.dumps(handle_user_prompt(event, args.database, selected_embedder)))
        except Exception:
            print("{}")
    return 0


def main() -> None:
    raise SystemExit(run())
