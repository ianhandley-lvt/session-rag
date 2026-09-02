# Configuration

Session RAG reads `~/.config/session-rag/config.toml` by default. Set
`SESSION_RAG_CONFIG` or pass the global `--config PATH` option to select a
different file.

Configuration precedence is:

1. Command-line option
2. `SESSION_RAG_*` environment variable
3. TOML configuration
4. Built-in default

The initial schema is:

```toml
operator_id = "ian"
artifacts = "/Users/ian.handley/.local/share/session-rag/artifacts"
database = "/Users/ian.handley/.local/share/session-rag/lancedb"

[extractor]
provider = "cursor"
mode = "ask"
model = "gemini-3.7-flash-low"
max_sanitized_chars = 500000

[projects.lvcore]
root = "/Users/ian.handley/src/work/lvcore"
knowledge_base = "/Users/ian.handley/src/personal/second-brain/lvcore_kb/Wiki"
```

When the current directory is inside a configured project root, Session RAG
selects the most specific matching project. Prompt text can never select or
widen this Retrieval Scope.

Inspect the effective configuration for the current directory with:

```sh
session-rag config show
```

With storage paths configured, commands no longer need repeated `--artifacts`
or `--database` options:

```sh
session-rag ingest
session-rag search "Where are LVCore logs?"
```
