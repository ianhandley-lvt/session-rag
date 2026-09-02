# Session RAG v1 Design Decisions

**Status:** Implemented. v1 acceptance criteria are complete — see closed [issue #1](https://github.com/ianhandley-lvt/session-rag/issues/1) and merged PR #19.<br>
**Decided:** 2026-09-01 · **Implemented:** 2026-09-02<br>
**Scope:** Local Claude Code session knowledge for one operator on one Mac

This is the readable, consolidated statement of the agreed v1 design. For the original pre-implementation framing and the open questions this document answers, see [`design-brief.md`](design-brief.md) (historical). For canonical domain vocabulary, see [`CONTEXT.md`](../CONTEXT.md). For the rationale behind specific hard-to-reverse decisions, see the ADRs linked throughout.

## Purpose

Session RAG turns useful knowledge created during Claude Code sessions into durable, structured evidence that can be retrieved during later work. It is intended to recover decisions, explanations, resolved problems, and current-system observations without treating an entire raw conversation as trustworthy memory.

The v1 pipeline will:

1. Read selected local Claude Code sessions.
2. Sanitize each session before sending it to the configured extraction provider.
3. Use Cursor initially to extract semantic Knowledge Episodes.
4. Validate the result and attach trusted provenance in application code.
5. Persist an immutable Extraction Artifact before indexing anything.
6. Build a rebuildable hybrid semantic and exact-text index in LanceDB.
7. Search that index before every non-empty Claude Code prompt.
8. Inject only relevant, in-scope, cited evidence within strict latency and token budgets.

These decisions describe v1 behavior as implemented and merged, not merely intended. Independent review after the initial implementation caught a few places where the code briefly diverged from what's agreed here before being corrected — notably default Retrieval Scope permitting an unscoped record it should have denied, and the whole-hook timeout not yet bounding metrics recording. What's described below is the final, corrected behavior.

## End-to-end architecture

```text
Local Claude session
        |
        v
Source adapter attaches trusted source and project provenance
        |
        v
Sanitization, secret redaction, and input-size validation
        |
        v
Cursor extracts zero or more Knowledge Episodes
        |
        v
Schema validation and trusted-provenance enforcement
        |
        v
Immutable Extraction Artifact written atomically
        |
        v
Source revision activated atomically
        |
        v
FastEmbed vectors + LanceDB full-text index
        |
        v
Claude Code UserPromptSubmit hook
        |
        v
Project-scoped hybrid retrieval and absolute relevance gate
        |
        v
Verification, temporal, and recency-aware ranking
        |
        v
Up to three cited Episode Records within the token budget
        |
        v
Claude receives evidence before handling the original prompt
```

Extraction does not run in the prompt hook. Prompt-time work is limited to query embedding, retrieval, filtering, ranking, and evidence formatting.

## Domain model

The canonical vocabulary is maintained in [`CONTEXT.md`](../CONTEXT.md). The essential distinctions are summarized here because they determine the architecture.

### Knowledge Episode

A Knowledge Episode is one resumable unit of useful knowledge: a decision, explanation, resolved problem, or system observation. Its boundary is semantic. It may span multiple conversational turns, and one session may produce zero, one, or many episodes.

The system must not mechanically equate an episode with one user-assistant turn or with an entire session.

### Episode Record

An Episode Record is the validated, schema-checked, provenance-attached persisted representation of one Knowledge Episode. Cursor proposes content; application code supplies trusted provenance.

### Extraction Artifact

An Extraction Artifact is the immutable JSON envelope containing all Episode Records extracted from one source revision. It is durable and replayable. LanceDB is derived from these artifacts and is not the system of record.

### Source revision and Active Revision

A source revision is identified by a source ID and content hash. Multiple revisions may be retained, but only one successfully extracted revision per source is the Active Revision eligible for normal retrieval.

### Verification Status

Verification Status is a human-controlled lifecycle state on an Episode Record:

- `unreviewed`
- `verified`
- `rejected`
- `superseded`

### Temporal Scope

Temporal Scope describes how a record's correctness behaves over time:

- `durable`: expected to remain valid until explicitly rejected or superseded
- `time_sensitive`: may become stale as the underlying system changes

Temporal Scope is independent of content type. A problem resolution may be durable, while an explanation of current behavior may be time-sensitive.

### Provenance, attribution, and authority

These are separate concepts:

- **Provenance** records trusted facts about where and under what project context the evidence was captured.
- **Operator ID** identifies the configured operator who captured the session.
- **Attribution** is an untrusted, cited content claim that another person originated an idea or decision.
- **Authority** is a retrieval-time ranking signal computed from policy. It is not a fixed field supplied by the extractor.

## Durable artifacts and the derived index

Every successful extraction is persisted before embedding or indexing. The logical artifact layout is:

```text
artifacts/<source_type>/<source_id>/<source_hash>.json
```

Each versioned envelope includes at least:

- schema version
- trusted source identity, URI, type, and content hash
- trusted project and operator provenance
- extraction provider, model, prompt version, and timestamp
- all validated Episode Records produced from that revision

Artifacts are written atomically using a temporary sibling followed by rename. A failed or partial extraction never becomes a valid artifact.

LanceDB consumes durable artifacts. Rebuilding the index means deleting the derived database and replaying the active artifact set without invoking Cursor again. See [ADR 0001: Extraction Artifacts are the system of record](adr/0001-extraction-artifacts-are-the-system-of-record.md).

## Verification and authority policy

The extractor initializes new Episode Records as `unreviewed`. It may not promote, reject, or supersede them.

Only an explicit operator action may change Verification Status. The intended CLI surface is conceptually:

```text
session-rag verify <record-id>
session-rag reject <record-id>
session-rag supersede <old-record-id> <replacement-record-id>
```

Supersession requires a replacement record ID so the provenance chain remains navigable.

Normal retrieval treats the states as follows:

- `verified`: retrievable with a ranking boost
- `unreviewed`: retrievable without that boost
- `rejected`: excluded
- `superseded`: excluded

Verification Status and supersession links live in a durable Record State Overlay, separate from the immutable artifact — editing the overlay is how `verify`/`reject`/`supersede` take effect without ever rewriting an artifact. See [`CONTEXT.md`](../CONTEXT.md) for the full Record State Overlay definition.

Rejected and superseded records remain available through explicit record or history lookup. Source-type weighting is deferred until a second source type exists. See [ADR 0002: Authority is computed, not stored](adr/0002-authority-is-computed-not-stored.md).

## Temporal and recency policy

Recency does not apply uniformly:

| Verification | Temporal scope | Normal retrieval treatment |
|---|---|---|
| `verified` | `durable` | No recency decay |
| `unreviewed` | `durable` | Mild recency decay |
| `verified` or `unreviewed` | `time_sensitive` | Normal recency decay |
| `rejected` or `superseded` | Either | Excluded |

The mild decay on unreviewed durable records prevents an extraction-time classification from giving unreviewed content permanent ranking strength.

## Source revision changes

When the content hash for an already known source changes:

1. Sanitize, extract, and validate the new revision completely.
2. Persist its immutable artifact.
3. Atomically make the new source revision active.
4. Make records from prior revisions ineligible for normal retrieval.
5. Preserve prior revisions and their record statuses for history and provenance.
6. Report an informational diff identifying old records that may require review.

Activating a source revision does not alter any Episode Record's Verification Status. If extraction, validation, artifact persistence, or activation fails, the prior Active Revision remains retrievable. Automatic record matching and supersession reconciliation are deferred. See [ADR 0003: Active revision is decoupled from verification](adr/0003-active-revision-decoupled-from-verification.md).

## Deletion contract

Deletion in v1 is explicit and manual:

```text
session-rag forget <source-id>
session-rag forget --project <project-id>
```

An explicit erasure removes:

- every artifact revision in scope
- corresponding LanceDB rows
- cached embeddings and extraction outputs
- identifying source metadata and provenance links

The command may print a one-time completion count, but it does not retain a durable audit entry identifying the erased source. A persistent record of erasure would weaken the erasure guarantee.

A missing file or lost authorization is not automatically an erasure request. Those states will be designed when automatic synchronization exists. Forgetting a source also does not prevent later re-ingestion if the original remains available; permanent exclusion is a separate feature.

## Input scope, sanitization, and extraction trust

### Whole-session extraction

Cursor receives one whole sanitized session in v1. Pre-detecting episode boundaries would duplicate part of the extractor's job and is deferred until real sessions demonstrate the need for chunking.

A configurable maximum sanitized input size is mandatory. Oversized sessions fail clearly rather than being silently truncated, because truncation could remove the final resolution. Semantic chunking and cross-chunk episode reconciliation are deferred.

### Sanitization before sending

All content is sanitized before it reaches Cursor. The sanitizer must:

- redact secret-shaped values across user, assistant, command, diff, and tool content
- support configurable sensitive-path exclusions
- bound large command output and generated diffs
- preserve safe, useful metadata such as tool name, command, referenced paths, exit status, and bounded relevant excerpts
- replace omitted material with an explicit bounded marker

Pattern-based secret detection is best-effort and must not be described as a guarantee.

### Trusted fields

Cursor may propose episode content, code references, Temporal Scope, and cited content-level attribution. It may not choose trusted values such as:

- source identity, path, type, or hash
- source revision
- project provenance
- Operator ID
- Verification Status beyond the application-defined initial value
- active-revision state

Unknown or forged trusted fields are rejected rather than silently accepted.

## Operator and project provenance

The application attaches `operator_id` from explicit configuration. It is not inferred from Git identity, which may vary between repositories and may expose an email address unnecessarily.

The source adapter also attaches trusted project context:

- `project_id`: stable configured identifier
- `project_root`: local path retained only in local provenance
- `repository_revision`: commit SHA when available
- `working_tree_dirty`: whether uncommitted changes existed when available

These fields are optional for sessions outside a Git repository. They describe where a session occurred; they do not validate extracted code references.

Code references remain unverified lexical hints in v1. File names, paths, error strings, and symbols improve exact-text retrieval, but the system does not yet prove that they existed at the recorded revision.

## Retrieval scope

Full multi-source and multi-user permission isolation is deferred. Nevertheless, a single operator may have unrelated or differently sensitive projects on one machine, so v1 enforces a local Retrieval Scope:

- Normal hook retrieval is limited to the current trusted `project_id`.
- Cross-project retrieval requires explicit configuration or an explicit query mode.
- Projectless records are available only through an intentionally configured global scope.
- The hook supplies the trusted project context.
- Prompt text cannot choose or widen its own retrieval scope.

This is a local disclosure boundary, not an ACL system. See [ADR 0004: Prompt text cannot widen retrieval scope](adr/0004-prompt-cannot-widen-retrieval-scope.md).

## Prompt-time retrieval and injection

### Retrieval gate

Retrieval runs for every non-empty submitted prompt. Prompt length is not a useful deterministic gate: short prompts such as "why?" or "what did we decide?" may depend heavily on prior knowledge.

Always retrieving does not mean always injecting. Evidence is injected only when at least one result clears the absolute relevance gate.

### Candidate qualification

A candidate qualifies through either:

- vector similarity above a configurable floor, or
- strong lexical evidence, such as an exact phrase, code identifier, file path, error message, or sufficiently strong normalized full-text score

Merely appearing in the full-text result set is insufficient. Reciprocal-rank fusion describes relative ordering and cannot determine that every candidate is irrelevant.

### Retrieval pipeline

The prompt-time order is:

1. Gather semantic and lexical candidates.
2. Apply the absolute relevance gate using raw retrieval signals.
3. Exclude records outside Retrieval Scope, from inactive revisions, or with rejected/superseded status.
4. Rank survivors using hybrid relevance, verification, Temporal Scope, and recency.
5. Deduplicate and format cited evidence.
6. Inject only the records that fit the count and token budgets.

Zero qualifying candidates means zero injected context.

### Provisional budgets

Initial configurable defaults are:

| Setting | Provisional default |
|---|---:|
| Whole-hook retrieval deadline | 500 ms |
| Maximum injected records | 3 |
| Maximum injected evidence | 1,000 tokens |

These values are starting points, not settled domain decisions. The time budget includes process startup, query embedding, search, filtering, ranking, and formatting. If loading FastEmbed in a new Python process makes the target consistently unattainable, a persistent local process or other model-caching architecture should be evaluated.

Metrics recording is best-effort and shares this same deadline rather than adding its own: it gets whatever time remains once retrieval finishes, and is abandoned rather than awaited if that's ~0. A slow or failing metrics writer can therefore never delay the hook past its deadline or change its response.

The token budget controls the final bundle. Records are truncated or omitted only at explicit boundaries; citations and structured fields are never cut mid-value.

On timeout or retrieval failure, the hook returns no additional context and allows Claude Code to process the original prompt.

## Citations and retrieval traces

Every injected Episode Record must cite the exact source revision (`source_type`, `source_id`, `source_hash`) and a stable Evidence Location within it: an identifier the source adapter's sanitizer itself produced, plus the exact text it pointed to, snapshotted at extraction time — never a position in the transient per-extraction sanitized rendering, which isn't preserved and can shift. See [`CONTEXT.md`](../CONTEXT.md) for the full Evidence Location definition. Retrieved text is labeled as evidence, not as an instruction to Claude.

Retrieval traces retain the raw vector and lexical scores, qualification/exclusion reasons, final ranking, latency, and what was injected. They are used to evaluate and tune retrieval. Prompt text itself is not retained merely for operational latency metrics.

## Extraction failure behavior

Extraction attempts use these states:

- `pending_retry`: Cursor unavailable, timed out, or quota-exhausted
- `failed`: invalid extractor output after bounded retries
- `blocked`: unsupported or rejected input, such as an oversized sanitized session, with an actionable reason

In every case:

- no partial Extraction Artifact is written
- no new LanceDB rows are added
- the prior Active Revision remains retrievable
- raw turn-level indexing is never used as a fallback
- another provider is never selected automatically

Retry is explicit and manual in v1. Non-sensitive job metadata may be retained to explain the failure, retry the correct source revision, and — since a failed attempt never produces an Episode Record of its own — record which project (if any) it was attempted under, so project-scoped erasure can still find and remove it. The extractor interface remains replaceable, but automatic provider fallback waits until a second provider is intentionally supported.

## Evaluation strategy

The existing evaluation layers remain:

- extraction validity and usefulness
- retrieval recall and precision
- citation fidelity
- freshness and authority behavior
- operational correctness
- prompt-time performance

The corpus must also include explicit acceptance cases for:

- verified durable, unreviewed durable, and time-sensitive decay behavior
- exclusion of rejected and superseded records
- successful Active Revision swaps
- preservation of the prior revision when replacement extraction fails
- same-project retrieval and prevention of cross-project leakage
- semantic, strong lexical, weak-match, and no-match relevance gates
- contradictory and stale records
- timeout, invalid-output, and oversized-input atomicity
- citations resolving to the correct source revision and evidence location

Extraction, retrieval, and end-to-end fixtures remain separate so extractor variability can be distinguished from deterministic retrieval regressions. Exact corpus contents, metric targets, relevance thresholds, and final budget values are implementation work informed by observed data.

## Primary testing seams

The project uses two primary behavioral seams.

### `session_rag.cli.run()`

This is the black-box seam for the local pipeline. Tests use real temporary artifact storage, filesystem behavior, and LanceDB with controlled injected dependencies:

- deterministic stub embeddings
- fixed clock
- fixed operator and project context
- configurable relevance thresholds and budgets

It covers artifact persistence, active revisions, lifecycle commands, erasure, hybrid retrieval, ranking, scoping, and hook output.

### `CursorExtractor.extract()` with an injected Runner

This is the external-process seam. Tests replace the real Cursor subprocess and verify:

- only sanitized input reaches Cursor
- secrets and excluded paths are absent
- malformed output and forged provenance are rejected
- trusted provenance is attached only by application code
- timeout and invalid output produce no artifact

Internal pure-function tests may support edge cases, but they do not establish additional architectural seams.

## Deferred work

The following are intentionally outside v1 or wait for evidence:

- semantic chunking of oversized sessions and cross-chunk reconciliation
- automatic session-end ingestion or background source monitoring
- automatic verification, heuristic promotion, or inferred supersession
- LLM- or embedding-assisted reconciliation between source revisions
- source-type authority weighting
- validation of code references against a repository revision
- multi-source ACLs and authorization synchronization
- automatic extractor-provider fallback
- a durable do-not-reingest exclusion list
- Notion and other source adapters

Each should be introduced in response to a demonstrated requirement or when the relevant second source/provider exists.

## Decision index

| Question | Decision |
|---|---|
| Q1 | The canonical unit is a semantic Knowledge Episode, not a turn or session. |
| Q2 | Persist immutable, versioned Extraction Artifacts before LanceDB indexing. |
| Q3a | Verification Status changes only through explicit operator action. |
| Q3b | Rejected/superseded records are excluded; verified records are boosted; source weighting is deferred. |
| Q4 | Temporal Scope controls decay independently of episode content type. |
| Q5 | Preserve all source revisions but retrieve only from the atomically selected Active Revision. |
| Q6 | Explicit erasure hard-deletes derived data and leaves no identifying durable audit trace. |
| Q7 | Retrieve on every non-empty prompt; inject only qualifying evidence. |
| Q8 | Latency and token budgets are configurable provisional defaults measured against real traces. |
| Q9 | Apply an absolute semantic-or-strong-lexical relevance gate before policy ranking; inject at most three. |
| Q10 | Use whole sanitized sessions in v1; defer semantic chunking. |
| Q11 | Mandatory best-effort secret redaction and bounded tool/diff content occur before extraction. |
| Q12 | Application-attached Operator ID is trusted provenance; content Attribution remains untrusted and cited. |
| Q13 | Add trusted project provenance now; defer code-reference validation. |
| Q14 | Defer full permission modeling, but isolate normal retrieval by project. |
| Q15 | Cursor failures preserve the prior active revision and never fall back to raw indexing or another provider. |
| Q16 | Retain the evaluation layers and add explicit policy-specific acceptance cases. |
| Testing | Keep two primary seams: CLI black box and Cursor external-process boundary. |

## Guiding principle

Extraction may be probabilistic. Provenance, persistence, revision activation, verification, retrieval boundaries, erasure, citations, and failure behavior must be deterministic and controlled by the application.
