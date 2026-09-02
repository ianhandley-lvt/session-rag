# Session RAG: Design Brief

**Status:** HISTORICAL. This was the seed document for the initial design ("grill-with-docs") session, written before any v1 code existed. It's preserved for the original framing and trade-offs, not as a description of the current system — its "Current implementation status" and "Open design questions" sections below reflect that starting point and are now superseded. For current behavior, see [`CONTEXT.md`](../CONTEXT.md) (canonical vocabulary), the [ADRs](adr/) (decision rationale), [`rag-design-decisions.md`](rag-design-decisions.md) (the consolidated, implemented v1 design), and closed [issue #1](https://github.com/ianhandley-lvt/session-rag/issues/1) (final acceptance criteria and history).<br>
**Purpose:** Seed document for a `grill-with-docs` session<br>
**Last updated:** 2026-09-01

## Summary

Session RAG is a local-first knowledge system that turns completed AI coding sessions and other work sources into durable, searchable evidence for future Claude Code prompts. It should collect knowledge where work already happens, distill noisy conversations into structured records, retrieve the most relevant and trustworthy records, and inject them before Claude begins reasoning. The system is intended to reduce repeated investigation and recover decisions, explanations, and resolutions that would otherwise disappear inside old sessions.

This is not intended to replace source systems such as Claude transcripts, Notion, repositories, or issue trackers. Those remain the sources of record. Session RAG maintains a derived index with citations back to them.

## Problem

Useful knowledge is routinely created during coding sessions:

- Why a system behaves a certain way.
- Which approaches failed and why.
- What decision was made and under which constraints.
- How an incident or implementation problem was resolved.
- Which services, files, symbols, tickets, and people were involved.

Claude Code stores session transcripts locally, but those transcripts are noisy and difficult to search as organizational memory. They mix user prompts, intermediate reasoning, tool activity, command output, generated patches, corrections, and final answers. Simple vector search over raw transcript chunks would preserve too much noise and would not distinguish a tentative suggestion from a verified resolution.

## Goals

1. Collect knowledge automatically from its original sources.
2. Convert noisy conversations into structured knowledge records.
3. Combine semantic retrieval with exact-text search.
4. Prefer authoritative and recent evidence when relevance is otherwise similar.
5. Inject retrieved evidence before Claude Code processes a submitted prompt.
6. Preserve citations back to the original source.
7. Keep extraction-model and embedding-model choices replaceable.
8. Fail without preventing normal Claude Code use.

## Non-goals

- Replacing the original transcript, document, repository, or issue tracker.
- Treating model-generated summaries as authoritative source material.
- Automatically allowing retrieved text to issue instructions to Claude.
- Building a shared multi-user enterprise service in the first version.
- Indexing all available work history before retrieval quality is evaluated on a small corpus.
- Using LanceDB as the system of record; it is a rebuildable derived index.

## Users and initial scope

The initial user is one developer on one Mac. The first source is local Claude Code session history for explicitly selected projects. Notion, repository history, issue trackers, and other sources may be added later through the same ingestion boundary.

The first retrieval consumer is a Claude Code `UserPromptSubmit` hook. A manual search command remains important for testing, debugging, and inspecting why a record was retrieved.

## Proposed architecture

```text
Original sources
  Claude sessions | Notion | repositories | issue trackers
        |
        v
Source adapters
  discover changes, read source content, attach trusted provenance
        |
        v
Knowledge extractor
  Cursor initially; local or other providers later
        |
        v
Schema validation and provenance enforcement
        |
        v
Structured knowledge records
        |
        +------------------------+
        |                        |
        v                        v
Local embeddings           Exact-text index
  FastEmbed initially        LanceDB FTS
        |                        |
        +-----------+------------+
                    v
             LanceDB derived index
                    |
                    v
       Hybrid retrieval and ranking
  semantic + lexical + authority + recency
                    |
                    v
          Compact cited evidence bundle
                    |
                    v
       Claude Code UserPromptSubmit hook
```

### Component responsibilities

#### Source adapter

A source adapter discovers new or changed source material and converts it into a provider-neutral conversation or document envelope. It owns trusted provenance such as source path, source identifier, project, timestamps available directly from the source, and content hashes. It must not ask an extraction model to invent or reproduce these values.

#### Knowledge extractor

A knowledge extractor interprets noisy source content and proposes zero or more structured knowledge records. The interface should not expose provider-specific details to ingestion:

```python
class KnowledgeExtractor(Protocol):
    def extract(self, transcript: Path) -> list[StructuredRecord]: ...
```

Cursor is the initial provider because it is already available and has unused quota. It runs non-interactively in read-only Ask mode. Configuration is supplied through:

```text
SESSION_RAG_CURSOR_MODE=ask
SESSION_RAG_CURSOR_MODEL=gemini-3.7-flash-low
```

The current provider boundary is incomplete: the interface still accepts a transcript path rather than a fully provider-neutral source envelope, and provider registration is hard-coded. These are deliberate topics for design review.

#### Validator and provenance enforcer

Model-generated fields must pass a strict schema. Unknown fields are rejected. Source path, source session ID, and authority are supplied or overwritten by trusted application code. Invalid output remains pending or failed and must not enter the searchable index silently.

#### Embedder

The embedder converts the retrieval-oriented representation of each record and each query into vectors. FastEmbed with `BAAI/bge-small-en-v1.5` is the initial implementation. Embedding is separate from extraction: extraction interprets a conversation; embedding represents text for similarity search.

#### Store

LanceDB stores structured fields, embedding vectors, and a full-text index. The database is local and derived. It should be possible to delete and rebuild it from extraction artifacts or original sources.

The existing prototype overwrites the table during ingestion. Production behavior should be incremental and idempotent, using stable record identifiers and content hashes.

#### Retriever and ranker

Retrieval should combine at least:

- Semantic similarity for paraphrases and conceptually related language.
- Full-text relevance for file names, symbols, ticket IDs, error strings, and exact terminology.
- Authority weighting so a verified decision can outrank a speculative working-session observation.
- Recency weighting so newer evidence is preferred where appropriate without erasing older durable decisions.

The current prototype combines vector and full-text result lists using reciprocal-rank fusion. Authority and recency ranking are not implemented yet.

#### Context injector

The Claude Code `UserPromptSubmit` hook receives the submitted prompt, retrieves evidence, and returns `additionalContext`. It should inject only a small number of high-confidence records and clearly label them as potentially stale evidence rather than instructions.

The hook must fail open: an unavailable model, missing index, timeout, malformed record, or retrieval exception must not prevent Claude from processing the original prompt.

## Canonical knowledge unit

The canonical unit is a **knowledge episode**: one resumable unit of durable knowledge about a decision, explanation, or problem and its resolution. Episode boundaries are semantic and may span however many conversational turns were required. A single session may contain zero, one, or many knowledge episodes.

The extractor identifies episodes; the transcript parser must not derive them mechanically from user-to-assistant turn boundaries. Turn-level records are too noisy and session-level records blur unrelated subjects, authority, and recency.

## Proposed knowledge record

```json
{
  "id": "stable-derived-id",
  "record_type": "problem_resolution",
  "question": "Why was RabbitMQ repeatedly reconnecting?",
  "summary": "The service missed broker heartbeats while the event loop was blocked.",
  "resolution": "Move the CPU-heavy task to a worker.",
  "systems": ["RabbitMQ", "liveunit-service"],
  "code_references": ["DemoQueueConsumer", "src/messaging/consumer.ts"],
  "author": "Ian",
  "source_type": "claude_session",
  "source_uri": "/absolute/path/to/session.jsonl",
  "source_id": "session-uuid",
  "source_locations": ["message-id-a", "message-id-b"],
  "source_created_at": "2026-08-27T10:00:00Z",
  "extracted_at": "2026-09-01T12:00:00Z",
  "authority": "working_session",
  "verification_status": "unreviewed",
  "extractor": "cursor",
  "extractor_model": "gemini-3.7-flash-low",
  "source_hash": "sha256:...",
  "schema_version": 1
}
```

This is a proposal, not the current implemented schema. Record granularity is resolved as one record per knowledge episode. Several other fields still require decisions during design review, especially authority, verification, exact source locations, and stable identity.

## Extraction artifact persistence

Every successful extraction is persisted outside LanceDB as an **extraction artifact** before embedding or indexing. An artifact is a versioned JSON envelope for one source revision, containing trusted source metadata, extraction metadata, and all knowledge episodes derived from that revision.

```json
{
  "schema_version": 1,
  "source": {
    "type": "claude_session",
    "id": "session-uuid",
    "uri": "/absolute/path/to/session.jsonl",
    "hash": "sha256:..."
  },
  "extraction": {
    "provider": "cursor",
    "model": "gemini-3.7-flash-low",
    "prompt_version": 1,
    "extracted_at": "2026-09-01T12:00:00Z"
  },
  "records": []
}
```

Use one atomic JSON document per source revision rather than bare JSONL. The records and their shared provenance must validate, replace, and replay together. JSONL remains a possible export or streaming format, but is not the canonical artifact format.

The intended logical layout is:

```text
artifacts/<source_type>/<source_id>/<source_hash>.json
```

The physical data root is configurable and remains to be decided. It should not be committed to the application repository. Atomic writes use a temporary sibling followed by rename so a crash cannot leave a partially valid artifact.

LanceDB consumes extraction artifacts only after they are durable. Rebuilding the index means deleting the LanceDB directory and replaying the active artifact set, without invoking an extraction model.

## Data flow

### Initial ingestion

1. The user explicitly selects one Claude Code project-history directory.
2. The source adapter discovers completed JSONL sessions.
3. Each source is fingerprinted to support idempotency.
4. A transcript parser reconstructs coherent turns and removes or labels tool noise.
5. The configured extractor proposes structured records.
6. The application validates output and attaches trusted provenance.
7. Validated records and trusted provenance are atomically persisted as a versioned extraction artifact.
8. Retrieval text is assembled from fields such as question, summary, resolution, systems, and code references.
9. FastEmbed creates vectors locally.
10. LanceDB consumes the durable artifact, upserts structured rows, and updates its full-text index.

### Incremental ingestion

The intended steady state is event-driven or periodically triggered ingestion of newly completed sessions. Extraction should not run synchronously inside `UserPromptSubmit`; prompt-time work should be limited to embedding the query, searching, ranking, and formatting evidence.

A failed extraction should be recorded with enough diagnostic information to retry. It should not cause the source to be marked successfully ingested.

### Prompt-time retrieval

1. Claude Code emits a `UserPromptSubmit` event.
2. The hook extracts the prompt and applies any retrieval gate.
3. The prompt is embedded locally.
4. LanceDB runs semantic and exact-text searches.
5. Results are fused and adjusted for authority and recency.
6. Low-confidence or duplicate evidence is removed.
7. A bounded evidence bundle is returned with citations.
8. Claude reasons over the original prompt plus the injected evidence.

## Trust and security model

- Source content is untrusted data. A transcript may contain prompt-injection text.
- Cursor runs in read-only Ask mode, with its sandbox enabled, against an isolated temporary workspace.
- Cursor receives transcript contents through standard input rather than a command-line argument or prompt file.
- Model output cannot choose trusted provenance fields.
- Retrieved evidence is data, not executable instruction.
- Secrets and raw tool output require filtering before extraction and before indexing.
- The local index contains derived work information and should be protected like the source material.
- Deleting or losing access to source material must eventually remove its derived records.
- Local-first extraction does not automatically mean offline: Cursor sends provided transcript content to Cursor's service. This requires the same organizational approval as other use of that content in Cursor.

## Failure behavior

| Failure | Desired behavior |
|---|---|
| Cursor unavailable or quota exhausted | Mark extraction pending/failed; retry later |
| Invalid extractor JSON | Reject record; retain diagnostic without indexing content |
| Embedding model unavailable | Leave validated record pending embedding |
| LanceDB unavailable | Do not block Claude; manual search reports unavailable index |
| Hook exceeds latency budget | Return no additional context and allow Claude to continue |
| Source deleted or access revoked | Remove or tombstone derived records |
| Conflicting records retrieved | Show both with sources or prefer verified authority; do not silently merge |
| No sufficiently relevant evidence | Inject nothing |

## Current implementation status

### Implemented

- Isolated Python project managed by `uv`.
- FastEmbed adapter.
- LanceDB storage and full-text index prototype.
- Semantic and exact-text retrieval combined with reciprocal-rank fusion.
- Parser for local Claude session JSONL files.
- Provider-neutral extraction protocol and strict structured-record validation.
- Cursor extraction adapter with configurable mode and model.
- Trusted source path and session ID attachment.
- Manual `extract-session` preview command.
- `UserPromptSubmit` hook response formatter and fail-open handler.
- Automated tests using synthetic transcripts.

### Partial or disconnected

- Claude transcript collection is manually triggered rather than automatic.
- Cursor extraction produces a validated preview but is not connected to LanceDB ingestion.
- Citations identify transcript path, session, and timestamp but not exact message ranges.
- Provider switching requires small source changes rather than configuration-only registration.
- The hook handler exists but is not installed in Claude Code settings.

### Missing

- Complete turn reconstruction for real Claude sessions.
- Durable extraction artifacts and extraction audit trail.
- Incremental/idempotent upsert and deletion handling.
- Authority and verification lifecycle.
- Recency-aware ranking.
- Retrieval thresholds, deduplication, and latency budget.
- Automatic session-end ingestion.
- Notion and other source adapters.
- Evaluation corpus and measurable retrieval-quality criteria.

## Open design questions

These are intentionally unresolved inputs for the grilling session.

1. What evidence is authoritative, and who or what can promote a working-session record to a verified decision?
2. Should recency affect every record type, or should durable architectural decisions resist age decay?
3. How should corrections supersede earlier records without destroying historical provenance?
4. What is the deletion contract when a source disappears, a project is no longer authorized, or a user requests erasure?
5. Should retrieval run on every prompt or only when a deterministic or learned gate predicts useful memory?
6. What latency and token budgets are acceptable for prompt-time retrieval?
7. How many records may be injected, and what confidence threshold should suppress weak results?
8. Should Cursor receive full sessions, bounded episodes, or pre-cleaned turns?
9. What transcript content must be removed before it is sent to an extraction provider?
10. How are authors identified in sessions containing user, assistant, tool, and subagent messages?
11. How should code references be validated against the relevant repository and commit?
12. How should multiple knowledge sources with different permission models be isolated?
13. What is the fallback if Cursor is unavailable: queue for retry, deterministic raw indexing, another provider, or no ingestion?
14. How will extraction and retrieval quality be evaluated before indexing broad work history?

## Proposed evaluation approach

Create a small, intentionally selected corpus of completed sessions and a set of questions whose expected evidence is known. Evaluate the pipeline in layers:

- **Extraction validity:** required fields are present, citations are correct, and no facts are invented.
- **Extraction usefulness:** durable decisions and resolutions are captured while routine chatter is omitted.
- **Retrieval recall:** expected evidence appears in the candidate set.
- **Retrieval precision:** injected evidence is relevant enough to help rather than distract.
- **Citation fidelity:** every returned claim can be traced to exact source content.
- **Freshness behavior:** newer corrections outrank stale records where intended.
- **Authority behavior:** verified decisions outrank speculative session observations where intended.
- **Operational behavior:** ingestion is idempotent and prompt-time failure remains fail-open.
- **Performance:** retrieval stays within the agreed latency and context budgets.

The initial evaluation should use synthetic and deliberately reviewed work sessions. Broad automatic ingestion should wait until the evaluation exposes acceptable behavior and failure modes.

## Candidate milestones

1. Resolve the domain vocabulary and record granularity.
2. Persist trusted, validated extraction artifacts for one selected session.
3. Upsert those structured records into LanceDB without overwriting unrelated records.
4. Create a small labeled evaluation corpus.
5. Implement authority, recency, deduplication, and relevance thresholds.
6. Verify manual retrieval with citations against the evaluation corpus.
7. Install the `UserPromptSubmit` hook with a strict latency budget.
8. Add automatic completed-session discovery and retryable background ingestion.
9. Add a second extractor to prove provider replaceability.
10. Add another source only after the Claude-session pipeline is trustworthy.

## Decisions already made

- The first deployment is local and single-user.
- LanceDB is the initial derived search index.
- FastEmbed is the initial local embedding implementation.
- Extraction and embedding are separate provider boundaries.
- Cursor is the first extraction provider.
- Cursor defaults to Ask mode and `gemini-3.7-flash-low`.
- Trusted provenance is attached by application code, not accepted from model output.
- Source systems remain the sources of record.
- Prompt-time retrieval must fail open.
- The canonical knowledge unit is a knowledge episode, identified semantically by the extractor rather than mechanically from turn or session boundaries.
- Validated extraction artifacts persist outside LanceDB as atomic, versioned JSON envelopes; LanceDB is rebuilt by replaying them without re-extraction.

These decisions may be challenged during grilling. If a decision changes and is hard to reverse, surprising, and based on a real trade-off, it should be captured as an ADR after agreement—not before.
