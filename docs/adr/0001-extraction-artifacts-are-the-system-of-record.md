# Extraction Artifacts are the system of record; LanceDB is a disposable derived index

Validated Episode Records are persisted first as immutable, versioned JSON envelopes — one per source revision, keyed by source ID + source hash — under `artifacts/`, before anything reaches LanceDB. LanceDB ingestion is a pure downstream replay of those files: it holds the searchable index, but it must be fully rebuildable from Extraction Artifacts without re-running extraction.

Use one atomic JSON document per source revision rather than bare JSONL, so trusted source metadata, extraction metadata, schema version, and all Episode Records for that revision validate, replace, and replay together as a unit — JSONL doesn't give atomic whole-unit replacement or clean schema evolution.

We picked this over writing directly into LanceDB during extraction because Cursor extraction costs real quota/spend; losing or corrupting the index would otherwise mean paying for extraction again just to recover it.

**Considered options:** write extraction output directly into LanceDB with no separate artifact (simpler, but ties rebuildability to LanceDB's own durability); append-only JSONL per source (rejected — these artifacts need atomic whole-unit replacement, provenance metadata, and schema evolution, which JSONL doesn't give cleanly).
