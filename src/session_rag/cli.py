from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .embeddings import FastEmbedder
from .extractors import create_extractor
from .extractors.base import ExtractionBlocked, ExtractionError
from .hook import format_context, handle_user_prompt
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
    extract.add_argument("--extractor", default="cursor", choices=["cursor"])
    extract.add_argument("--cursor-mode", choices=["ask", "plan"])
    extract.add_argument("--cursor-model")
    search = commands.add_parser("search")
    search.add_argument("query")
    search.add_argument("--database", type=Path, required=True)
    hook = commands.add_parser("hook")
    hook.add_argument("--database", type=Path, required=True)
    return result


def run(arguments: list[str] | None = None, embedder: Embedder | None = None) -> int:
    args = parser().parse_args(arguments)
    if args.command == "ingest-sessions":
        selected_embedder = embedder or FastEmbedder()
        count = index_memories(args.database, load_sessions(args.directory), selected_embedder)
        print(f"Indexed {count} session memories in {args.database}")
    elif args.command == "extract-session":
        extractor = create_extractor(
            args.extractor,
            cursor_mode=args.cursor_mode,
            cursor_model=args.cursor_model,
        )
        try:
            records = extractor.extract(args.transcript)
        except ExtractionBlocked as error:
            # Distinct from a failed attempt: the input was rejected before extraction ran.
            # Full pending_retry/failed/blocked job-status persistence is ticket #5's job —
            # this is the minimal CLI-surface signal this ticket's acceptance criteria need.
            print(f"blocked: {error.reason}", file=sys.stderr)
            return 2
        except ExtractionError as error:
            print(f"failed: {error}", file=sys.stderr)
            return 1
        print(json.dumps({"records": [record.model_dump(mode="json") for record in records]}, indent=2))
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
