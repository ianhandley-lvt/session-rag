# Markdown Knowledge Bases

**Status:** Implemented

Session RAG can import a curated Markdown Wiki without sending it through Cursor. This path is intended for knowledge bases that already contain synthesized articles, meaningful headings, dates, and citations to original material.

## Source contract

Pass the Wiki directory, not the knowledge-base root or its `RAW/` directory. The importer:

- recursively reads Markdown articles;
- skips `INDEX.md` and `QUESTIONS.md` as navigation/work-queue files;
- treats each article as one immutable, content-hashed source revision;
- creates Episode Records at Markdown heading boundaries and skips navigation-only `Related` sections;
- groups oversized sections at paragraph boundaries, then sentence boundaries only when one paragraph is too large; a paragraph with no semantic boundary fails clearly rather than being arbitrarily word-chunked;
- preserves article `Status`, `Last updated`, and `Sources` metadata;
- uses the article path, heading path, preserved section text, and source hash as cited evidence;
- defaults every imported section to `time_sensitive` unless the operator explicitly selects another Temporal Scope;
- initializes lifecycle state as `unreviewed`, like every other source.

The source type is `markdown_knowledge_base`. The deterministic adapter is recorded as extractor `markdown` with model `deterministic`; no extraction-provider request is made.

## Import and index

```bash
uv run session-rag import-markdown-kb /path/to/knowledge-base/Wiki \
  --knowledge-base-id lvcore \
  --project-id lvcore \
  --project-root /path/to/lvcore \
  --artifacts ~/.local/share/session-rag/artifacts

uv run session-rag ingest \
  --artifacts ~/.local/share/session-rag/artifacts \
  --database ~/.local/share/session-rag/lancedb
```

`SESSION_RAG_OPERATOR_ID` supplies the operator unless `--operator-id` is passed. Import is idempotent: unchanged article content reuses the existing artifact, while changed content produces and activates a new immutable revision.

## Search

```bash
uv run session-rag search "How do I find LVCore logs in CloudWatch?" \
  --artifacts ~/.local/share/session-rag/artifacts \
  --database ~/.local/share/session-rag/lancedb \
  --project-id lvcore
```

Normal retrieval remains project-scoped. The article's own status, such as `established` or `emerging`, is preserved as document metadata but does not automatically set Verification Status or grant a ranking boost.

## LVCore mapping

For the knowledge base at `/Users/ian.handley/src/personal/second-brain/lvcore_kb`:

- Wiki source: `lvcore_kb/Wiki`
- RAW provenance: `lvcore_kb/RAW` (not indexed directly)
- knowledge-base ID: `lvcore`
- retrieval project ID: `lvcore`
- project root: `/Users/ian.handley/src/work/lvcore`

The current Wiki contains 17 evidence articles after excluding its two navigation files. Counts may change as the Wiki evolves.
