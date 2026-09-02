# Session RAG

Session RAG turns knowledge created in work sessions and other source systems into durable evidence that can be retrieved for later work.

## Language

**Knowledge Episode**:
One resumable unit of durable knowledge about a decision, explanation, or problem and its resolution. A knowledge episode may span any number of conversational turns, while one session may contain multiple episodes.
_Avoid_: Turn, session, chunk, problem-resolution episode

**Episode Record**:
The validated, schema-checked, provenance-attached persisted representation of one Knowledge Episode. What the extractor's raw draft becomes once trusted provenance is attached and it passes the strict schema.
_Avoid_: Episode (when the persisted/validated artifact specifically is meant, not the underlying real-world unit), record (ambiguous alone)

**Source Type**:
Immutable provenance category of an Episode Record's origin — `claude_session`, `adr`, `notion_page`, `external_doc`, etc. Set once at extraction time from the source adapter, never changes for a given record.
_Avoid_: Authority (source type says where evidence came from, not how much to trust it)

**Verification Status**:
Mutable lifecycle state of one Episode Record — `unreviewed`, `verified`, `rejected`, `superseded`. Changes after extraction, independent of source type.
_Avoid_: Authority

**Temporal Scope**:
Whether an Episode Record's correctness is expected to erode with time. `durable` — stays valid until explicitly rejected or superseded, no time decay. `time_sensitive` — describes current system state or circumstances that can go stale with no supersession event to catch it, subject to recency decay in ranking. Orthogonal to Verification Status and to any future content-kind taxonomy (decision/explanation/problem-resolution).
_Avoid_: Episode kind, record type (those would classify *content*; this classifies *decay behavior*)

**Authority**:
A ranking-time signal computed from source type, verification status, temporal scope, and recency — not a stored field on the record. Determines how much a piece of evidence should outrank another during retrieval.

v1 policy: `rejected`/`superseded` excluded from normal retrieval (still reachable via explicit record/history lookup). Among retrievable records: `verified` + `durable` → no recency decay. `unreviewed` + `durable` → mild recency decay (an extraction-time `durable` label must not grant unreviewed content permanent ranking strength). `time_sensitive` (either verification status) → normal recency decay. Source-type weighting deferred until a second source type ships, with the policy versioned/configurable so ranking changes stay testable.
_Avoid_: Storing this as a fixed enum on the record (rejected — see [ADR-0002](docs/adr/0002-authority-is-computed-not-stored.md)); `working_session`/`verified_decision` as record-level values

**Record State Overlay**:
Durable, mutable storage — separate from any Extraction Artifact — holding `verification_status` and supersession links, keyed by each Episode Record's stable ID. Exists because artifacts are immutable and verification is not: a `verify`/`reject`/`supersede` command edits the overlay, never the artifact. Retrieval reads an Episode Record's content from its artifact and its lifecycle state from the overlay. Must survive a LanceDB rebuild — proof it isn't merely baked into derived index rows.
_Avoid_: Editing the artifact in place, storing verification status only in LanceDB

**Extraction Job Status**:
The outcome of one extraction attempt on a source revision — `pending_retry` (Cursor unavailable, timed out, or quota-exhausted), `failed` (invalid extractor output after bounded retries), or `blocked` (unsupported/rejected input, e.g. an oversized sanitized session, with an actionable reason). Distinct from Verification Status, which applies per Episode Record, not per extraction attempt. No case writes a partial Extraction Artifact, adds LanceDB rows, or changes the Active Revision — the prior active revision stays retrievable throughout. Retry is explicit/manual in v1.
_Avoid_: Verification Status (per-record lifecycle, not per-job outcome)

**Retrieval Scope**:
A local boundary — not a permission system — limiting a query to the current `project_id` by default, using trusted Project Provenance, to prevent accidental disclosure between unrelated workspaces on one machine. Cross-project or global-scope retrieval requires explicit configuration; the hook passes the trusted project context itself, and the submitted prompt text can never choose or override scope (closes an injection path — retrieved evidence must stay data, never a lever to widen its own future retrieval). See [ADR-0004](docs/adr/0004-prompt-cannot-widen-retrieval-scope.md).
_Avoid_: Permission, ACL, access control (those are the deferred, real multi-source authorization problem — this is narrower)

**Project Provenance**:
Trusted, source-adapter-attached context describing which codebase a session concerns: `project_id` (stable configured identifier), `project_root` (local path, retained only in local provenance, never sent to an extraction provider), `repository_revision` (commit SHA when available), `working_tree_dirty` (whether uncommitted changes existed). Optional for sessions outside a Git repository. Describes source context only — does not validate that any extracted code reference actually exists at that revision.
_Avoid_: Code reference validation (a separate, deferred concept — project provenance says where a session happened, not whether what it mentions is still true)

**Operator ID**:
Trusted provenance answering "who captured this session?" — attached by application code from explicit configuration, never inferred from Git identity (varies per repo, leaks email unnecessarily) and never supplied by the extractor model. Tool and subagent messages within a session are not separate operators; the whole session is attributed to the one configured operator.
_Avoid_: Author (ambiguous between provenance and content-level claims — see Attribution)

**Attribution**:
An untrusted, content-level claim the extractor may propose when a transcript credits a decision or idea to a specific person, backed by a source citation. Answers "whom does the source claim originated this information?" — distinct from Operator ID, and never promoted to trusted provenance automatically.
_Avoid_: Author

**Retrieval Trace**:
A record of one retrieval's raw vector and lexical scores, which candidates qualified or were excluded and why, and what was ultimately injected. Captured so the evaluation corpus can tune relevance-gate and ranking constants against real data rather than argument.
_Avoid_: Log (implies generic operational logging; this is structured data the evaluation approach specifically consumes)

**Active Revision**:
The one source revision (by source hash) whose Episode Records are eligible for normal retrieval, per source ID. Set atomically only after a new Extraction Artifact is fully extracted and validated; if extraction/validation fails, the previous revision stays active. Ineligibility from a non-active revision is a source-level fact, distinct from and does not alter any individual record's Verification Status. See [ADR-0003](docs/adr/0003-active-revision-decoupled-from-verification.md).
_Avoid_: Latest revision (implies mere recency, not the atomic all-or-nothing swap), current version

**Extraction Artifact**:
The versioned, immutable JSON envelope holding all Episode Records produced from one source revision — keyed by source ID and source hash so re-extraction never destroys a prior revision. The durable, replayable input to the LanceDB index; LanceDB itself stays a rebuildable derived index, never the system of record. See [ADR-0001](docs/adr/0001-extraction-artifacts-are-the-system-of-record.md).
_Avoid_: Extraction result, cache (implies disposable; this is durable), vector row, index row

**Evidence Location**:
A stable pointer into the exact source revision an Episode Record's claim is drawn from, beyond source type/ID/hash alone: an `identifier` (the source's own stable per-turn id when it provides one, otherwise a deterministic position in the immutable raw file — never a position in the transient, per-extraction sanitized rendering, which shifts whenever a turn is skipped or renders as more than one line) plus `preserved_text`, a snapshot of that turn's sanitized text captured at extraction time. The model may only select an identifier the source adapter's sanitizer itself produced for that exact revision — application code rejects anything else, including a plausible-looking invented one — and never supplies `preserved_text` itself. Persisted inside the Episode Record's own Extraction Artifact, so a citation stays resolvable even after the live source at `source` has since changed or been deleted; resolution never re-reads the live source.
_Avoid_: Sanitized line number (the transient rendering it replaces — not stable, not preserved, not what the citation should ever display)
