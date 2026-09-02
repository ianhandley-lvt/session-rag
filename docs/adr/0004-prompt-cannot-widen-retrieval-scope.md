# The submitted prompt can never choose or override Retrieval Scope

Retrieval Scope defaults to the current `project_id`, established from trusted Project Provenance and passed into the hook by the application — never derived from, or overridable by, the text of the submitted prompt. This is deliberately narrower than a real permission system (multi-source ACLs remain out of scope until a source with a distinct access model is added), but it closes a concrete injection path: since retrieved evidence is untrusted data that could contain adversarial text (per the system's own trust model), a design where prompt content could request "search other projects" would let injected instructions from one project's evidence pull in and potentially exfiltrate another project's evidence in the same response.

We rejected letting an explicit query mode be requested via the prompt itself; cross-project or global-scope retrieval requires explicit configuration outside the prompt channel.
