# Memory

Memory is a local, evidence-backed knowledge system for AI-assisted work. It
turns Claude Code sessions, Cursor conversations, and curated Markdown into
structured Episode Records, stores immutable source-linked artifacts, builds a
hybrid search index, and gives relevant evidence to Claude before it responds.

Everything stays on your computer except extraction: by default, Memory asks
the Cursor agent and its configured model to turn sanitized conversations into
structured records. LanceDB and FastEmbed perform storage and retrieval
locally.

## Concept map

```mermaid
flowchart LR
    A[Claude transcripts] --> S[Sanitize]
    B[Cursor conversations] --> S
    C[Curated Markdown] --> D[Deterministic parser]
    S --> E[Cursor extraction model]
    E --> R[Episode Records]
    D --> R
    R --> F[Immutable artifacts<br/>source of derived memory]
    F --> U[Exact + semantic<br/>duplicate analysis]
    U --> I
    I[LanceDB index<br/>semantic + exact-text search]
    P[Your Claude prompt] --> H[UserPromptSubmit hook]
    H --> I
    I --> K[Ranked, project-scoped evidence<br/>with citations]
    K --> M[Claude reasons with your prompt<br/>plus retrieved evidence]
```

The original transcript or document remains the authority. Artifacts preserve
structured records and citations. LanceDB is a disposable index that can be
rebuilt with `memory ingest`.

## Requirements

- macOS or another Unix-like environment
- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/) for installation
- `cursor-agent` installed and authenticated for conversation extraction
- Claude Code if you want automatic prompt-time retrieval

Confirm the external commands are available:

```sh
uv --version
cursor-agent --help
claude --version
```

## 1. Install Memory

Clone the repository, enter it, and install the command as an editable uv tool:

```sh
git clone https://github.com/ianhandley-lvt/session-rag.git
cd session-rag
uv tool install --editable .
```

Confirm installation:

```sh
memory --help
```

If your shell cannot find `memory`, add uv's tool directory to your `PATH`.
With direnv, put this in the repository's `.envrc` instead of `.zshrc`:

```sh
PATH_add "$HOME/.local/bin"
```

Then approve it once:

```sh
direnv allow
```

The former `session-rag` command remains an alias for compatibility.

## 2. Configure Memory

Create `~/.config/memory/config.toml`:

```toml
operator_id = "your-name"
artifacts = "/Users/you/.local/share/memory/artifacts"
database = "/Users/you/.local/share/memory/lancedb"

[extractor]
provider = "cursor"
mode = "ask"
model = "gemini-3.7-flash-low"
max_sanitized_chars = 500000

[projects.my-project]
root = "/Users/you/src/work/my-project"

[projects.another-project]
root = "/Users/you/src/personal/another-project"
knowledge_base = "/Users/you/knowledge/another-project/Wiki"
```

Use any stable name for each project table, such as `lvcore`. `root` is the
local repository directory. `knowledge_base` is optional.

You can register projects without editing TOML by hand. From anywhere inside a
Git repository, this finds its root and uses the directory name as the project
ID:

```sh
memory config add-project
```

Register a specific directory, or override its inferred ID:

```sh
memory config add-project ~/src/work/lvcore
memory config add-project ~/src/work/schedule-management-service --id schedule-service
```

The command preserves the existing config file, refuses conflicting IDs or
roots, and reports `already_registered` when the same project is added again.

Create the storage directories and inspect the single global configuration:

```sh
mkdir -p ~/.local/share/memory/artifacts ~/.local/share/memory/lancedb
memory config show
```

`config show` lists every registered project. To see which project Memory
resolves from the directory you are currently in, run:

```sh
memory config current
```

Configuration precedence is command option, `MEMORY_*` environment variable,
TOML value, then built-in default. `SESSION_RAG_*` variables and the old config
path remain fallback compatibility mechanisms.

## 3. Put knowledge into Memory

### Preview first

Extraction uses Cursor model quota. A dry run discovers candidates without
calling the model:

```sh
memory import-sessions --source claude --all-projects --dry-run
memory import-sessions --source claude --project my-project --dry-run
memory import-sessions --source claude --project current --dry-run
memory import-sessions --source cursor --dry-run
```

### Import Claude sessions

Import every transcript belonging to configured projects:

```sh
memory import-sessions --source claude --all-projects
```

Limit the import to one project or recent sessions:

```sh
memory import-sessions --source claude --project my-project
memory import-sessions --source claude --project current
memory import-sessions --source claude --project my-project --since 2026-09-01
```

Claude imports always require an explicit project scope. `--project current`
uses the registered project containing the current directory; it fails clearly
when the directory is not inside one. `--all-projects` imports every registered
Claude project.

To retry recorded extraction failures without retrying everything:

```sh
memory import-sessions --source claude --project my-project --resume
```

If a failed transcript changed afterward, Memory reports
`changed_since_failure` instead of silently retrying different content. Run a
normal import to treat the changed file as a new revision.

For the newest Claude session in the project you are currently inside:

```sh
cd /Users/you/src/work/my-project
memory capture --latest
```

This extracts the session and rebuilds the index in one operation.

### Import Cursor conversations

```sh
memory import-sessions --source cursor --dry-run
memory import-sessions --source cursor
```

Memory reads a temporary, read-only snapshot of Cursor's local conversation
search database and ignores duplicate cloud-cache rows. Cursor does not expose
a trustworthy project path there, so these records are deliberately unscoped.
They appear only in searches that explicitly use `--global-scope`.

There is intentionally no `--source all` command. Claude imports are
project-scoped while Cursor imports are global and unscoped, so combining them
made discovery totals misleading. Run the two explicit commands separately.

### Import a curated Markdown knowledge base

```sh
memory import-markdown-kb /path/to/Wiki \
  --knowledge-base-id team-wiki \
  --project-id my-project \
  --temporal-scope durable

memory ingest
```

Markdown import is deterministic and does not spend Cursor model quota.

### Import one transcript manually

```sh
memory extract-session /absolute/path/to/session.jsonl \
  --project-id my-project \
  --project-root /Users/you/src/work/my-project

memory ingest
```

`extract-session` creates and activates an artifact; `memory ingest` rebuilds
LanceDB from all active artifacts.

## 4. Test retrieval

Project-scoped search:

```sh
memory search "Where are the application logs?" --project-id my-project
```

Global search, including unscoped Cursor records:

```sh
memory search "How did we fix the deployment?" --global-scope
```

A weak match returns `No relevant session memory found.` Memory combines
semantic similarity with exact-text retrieval, then applies verification,
temporal, relevance, and project-scope rules.

## 5. Connect Memory to Claude Code

The repository includes the fail-open hook wrapper at
`scripts/claude-user-prompt-submit`. Make it executable:

```sh
chmod +x /absolute/path/to/memory/scripts/claude-user-prompt-submit
```

Add this to the target project's `.claude/settings.local.json`, merging it with
any existing settings:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/absolute/path/to/memory/scripts/claude-user-prompt-submit",
            "args": ["--project-id", "my-project"],
            "statusMessage": "Searching project memory..."
          }
        ]
      }
    ]
  }
}
```

Restart Claude Code after changing its settings. Every submitted prompt then
triggers a bounded local retrieval before Claude starts reasoning. Relevant
records are returned as `additionalContext`; no match, error, or timeout returns
an empty object and allows Claude to continue normally. Extraction never runs
inside this hook.

Use `--global-scope` instead of the project arguments only if that Claude
workspace should be allowed to retrieve records from every project and
unscoped Cursor conversations.

## Everyday command reference

| Goal | Command |
| --- | --- |
| Show the global configuration and all projects | `memory config show` |
| Show the project resolved for this directory | `memory config current` |
| Register the current project | `memory config add-project` |
| Register another project | `memory config add-project PATH [--id ID]` |
| Preview all configured Claude sessions | `memory import-sessions --source claude --all-projects --dry-run` |
| Import one project's Claude sessions | `memory import-sessions --source claude --project ID` |
| Import the current project's Claude sessions | `memory import-sessions --source claude --project current` |
| Capture the current project's newest session | `memory capture --latest` |
| Preview/import Cursor | `memory import-sessions --source cursor --dry-run` / remove `--dry-run` |
| Retry unchanged failed revisions | `memory import-sessions --source claude --project ID --resume` |
| Rebuild the derived index | `memory ingest` |
| Search one project | `memory search "question" --project-id ID` |
| Search everything | `memory search "question" --global-scope` |
| Inspect a record and its status | `memory history RECORD_ID` |
| Review duplicate candidates | `memory duplicates --project-id ID` |
| Run a local knowledge audit | `memory health-check --project-id ID` |
| Add Cursor contradiction/gap analysis | `memory health-check --project-id ID --ai` |
| Mark a record trustworthy | `memory verify RECORD_ID` |
| Remove a bad record from retrieval | `memory reject RECORD_ID` |
| Replace an old record | `memory supersede OLD_RECORD_ID NEW_RECORD_ID` |
| Erase one source and its index rows | `memory forget SOURCE_ID` |
| Erase a project's sources | `memory forget --project ID` |
| Show help for any operation | `memory COMMAND --help` |

Record IDs live in the immutable artifact JSON files under the configured
artifact directory. Search citations identify the source type, source ID,
source hash, and evidence location needed to trace a result back to its source.

## How records are treated

- New records begin as `unreviewed` and remain searchable.
- Exact normalized duplicates receive a `duplicate_of` link and are omitted
  from the index; their immutable source evidence is retained.
- Probable semantic duplicates remain searchable and receive scored
  `reinforces` links for review with `memory duplicates`.
- `verified` records receive a ranking boost.
- `rejected` and `superseded` records remain in history but leave retrieval.
- Durable verified knowledge resists time decay; time-sensitive observations
  lose ranking strength as they age.
- Retrieval is project-scoped unless global scope is explicitly enabled.
- Prompt text cannot widen its own scope.

## Improve the knowledge base

Run the fully local audit whenever you want a maintenance report:

```sh
memory health-check --project-id my-project
```

It reports possible duplicates or contradictions, stale time-sensitive records, missing evidence
locations, failed ingestion jobs, and retrievals that returned nothing. Reports
are saved under `artifacts/health-checks/PROJECT_ID/`. The command never changes
or verifies records.

For contradiction, coverage-gap, and suggested-article analysis, explicitly
authorize one Cursor synthesis call:

```sh
memory health-check --project-id my-project --ai
```

This sends structured, previously sanitized Episode Record content to the
configured Cursor model. It does not send raw transcripts or project-root
paths. Suggested record links are accepted only when they resolve to real
records in the selected project. Review the report, then use `verify`, `reject`,
or `supersede` yourself; no recommendation is applied automatically.

## Troubleshooting

**`memory: command not found`**

Ensure `~/.local/bin` is on `PATH`, then rerun `uv tool install --editable .`.

**No Claude sessions are discovered**

Confirm the project `root` exactly matches the path used when Claude Code ran.
Use `memory config show` to inspect all registered roots and `memory config
current` from inside the project to confirm that it resolves correctly.

**Extraction is blocked because the session is too large**

Raise `extractor.max_sanitized_chars` deliberately, or start with newer/smaller
sessions using `--since`. Memory refuses silent truncation.

**Cursor is unavailable, times out, or runs out of quota**

The session is marked `pending_retry`; authenticate or wait for quota, then use
`--resume`. Memory does not fall back to raw turn indexing or another provider.

**Search finds nothing after a manual extraction**

Run `memory ingest`. Batch imports and `capture --latest` rebuild the index
automatically; `extract-session` and Markdown import do not.

**Cursor memories do not appear in a project search**

This is intentional. Use `--global-scope`; Cursor records are unscoped until a
trustworthy Cursor-to-project mapping is available.

## Development

```sh
uv sync
uv run pytest
```

Architecture vocabulary and invariants are in [CONTEXT.md](CONTEXT.md). The
implemented design decisions are in
[docs/rag-design-decisions.md](docs/rag-design-decisions.md), and configuration
details are in [docs/configuration.md](docs/configuration.md).
