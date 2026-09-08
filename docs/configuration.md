# Configuration

Memory reads `~/.config/memory/config.toml` by default. Set
`MEMORY_CONFIG` or pass the global `--config PATH` option to select a
different file.

Configuration precedence is:

1. Command-line option
2. `MEMORY_*` environment variable (`SESSION_RAG_*` remains a compatibility alias)
3. TOML configuration
4. Built-in default

The initial schema is:

```toml
operator_id = "ian"
artifacts = "/Users/ian.handley/.local/share/memory/artifacts"
database = "/Users/ian.handley/.local/share/memory/lancedb"

[extractor]
provider = "cursor"
mode = "ask"
model = "gemini-3.7-flash-low"
max_sanitized_chars = 500000

[projects.lvcore]
root = "/Users/ian.handley/src/work/lvcore"
knowledge_base = "/Users/ian.handley/src/personal/second-brain/lvcore_kb/Wiki"
```

When the current directory is inside a configured project root, Memory
selects the most specific matching project. Prompt text can never select or
widen this Retrieval Scope.

Inspect the single global configuration, including every registered project,
with:

```sh
memory config show
```

Inspect only the project resolved from the current directory with:

```sh
memory config current
```

With storage paths configured, commands no longer need repeated `--artifacts`
or `--database` options:

```sh
memory ingest
memory search "Where are LVCore logs?"
```

Capture the newest Claude session for the configured current project and
rebuild the index in one command:

```sh
cd /Users/ian.handley/src/work/lvcore
memory capture --latest
```

## Batch ingestion

Preview first, then remove `--dry-run` to extract and index:

```sh
memory import-sessions --source claude --all-projects --dry-run
memory import-sessions --source claude --project lvcore --since 2026-09-01 --dry-run
memory import-sessions --source claude --project current --dry-run
memory import-sessions --source cursor --dry-run
```

The batch continues past individual failures and rebuilds the index once at
the end. Resume never silently retries a different mutable revision: changed
sources are reported as `changed_since_failure` and require a fresh normal
import. Cursor is read through a temporary, read-only database snapshot.
Because its project fingerprint is opaque, Cursor conversations are imported
without project provenance and require `--global-scope` during retrieval.
Memory intentionally does not combine Claude and Cursor discovery under a
`--source all` option because their scope and provenance rules differ.

Each index rebuild automatically skips exact normalized duplicates inside the
same project and flags probable semantic matches without removing them. Inspect
the review queue with `memory duplicates --project-id PROJECT_ID`. Tune the
provisional cosine floor with `MEMORY_SEMANTIC_DUPLICATE_THRESHOLD` (default
`0.92`) only against evaluated examples.

Run `memory health-check --project-id PROJECT_ID` for a local, read-only audit.
Add `--ai` only when you explicitly want structured Episode Records sent to the
configured Cursor model for contradiction, coverage-gap, and article
suggestions. Both modes write a durable report and apply no changes.

`capture` refuses to run unless the selected project ID and root match a
project registered in the TOML file. This prevents an environment variable
from silently attaching one project's provenance to another project's
transcript.
