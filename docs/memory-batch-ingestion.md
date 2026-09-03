# Memory rename and session batch ingestion

Memory is the public product name and canonical command. The Python package
remains `session_rag` internally, while the old command, configuration path,
and environment prefix remain compatibility aliases.

The `import-sessions` command discovers Claude transcripts from registered
project roots and Cursor conversations from a read-only snapshot of Cursor's
local search index. It supports source/project selection, dry runs, date
filtering, and explicit retry of recorded failures. Every source still passes
through sanitization, structured extraction, immutable artifacts, active
revision selection, and the shared LanceDB rebuild.

Cursor's opaque root fingerprint is not treated as trusted project identity.
Cursor records remain unscoped and require explicit global retrieval.
