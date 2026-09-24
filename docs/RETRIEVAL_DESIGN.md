# Phase 7 Retrieval Design (Phase 7A)

Phase 7 measures retrieval quality separately from reasoning quality. This
document covers Phase 7A only: contracts, the corpus boundary, a
deterministic plumbing-check retriever, and retrieval-only metrics. No
embedding model, vector database, or reranker is implemented yet.

## Oracle context vs. retrieval context

Every case in `synthetic-v1`/`synthetic-v2` ships with `EvaluationCase.evidence_pool`
— the small (1-4 record), hand-curated set of evidence the benchmark author
attached to that one case, deliberately including distractors within that
set. This is the **oracle context**: what Phases 1-6 hand the agent
directly, with no search step at all.

Phase 7 introduces a **retrieval context**: instead of handing the agent a
pre-curated pool, a `Retriever` searches a much larger corpus and selects
what it believes is relevant. The agent then reasons over the retriever's
selection, not the oracle pool.

## Why `case.evidence_pool` cannot be the retrieval corpus

If a retriever were allowed to search only within `case.evidence_pool`, it
would only ever have to rank 1-4 pre-selected documents — never distinguish
genuinely relevant evidence from a large pool of irrelevant material. That
would make retrieval trivial by construction and measure nothing real.

`case.evidence_pool` remains meaningful, but for a different job: it (via
`ExpectedOutcome.required_evidence_ids`) still tells the **evaluator** which
evidence was genuinely necessary for the correct answer. It must never be
used to *constrain* what a retriever is allowed to search over.

`app.retrieval.corpus.build_full_version_corpus` builds the **FULL VERSION
CORPUS**: every `Evidence` record reachable by a dataset version (72 records
for `synthetic-v2`, vs. 1-4 per case) — the union across every case in that
version, never one case's own pool.

## Retrieval leakage boundary

`app.retrieval.contracts.RetrievalQuery` is the retrieval analogue of
`app.agent.interface.AgentInput`: the *only* view of an `EvaluationCase` a
`Retriever` implementation may see. Allowed: `case_id` (opaque tracing
metadata only — never consulted by scoring logic), `Control`,
`scenario_description`, `period`. `extra="forbid"` means a caller cannot
smuggle `ExpectedOutcome`, `required_evidence_ids`, `should_abstain`,
`rationale`, `failure_mode_tag`, evaluator results, or a previous
benchmark answer into a query even by accident — the schema has no field
for any of them.

Unlike `AgentInput`, `RetrievalQuery` carries no `evidence` field: a
retrieval query is a *request for* evidence, not a container that already
holds it.

`RetrievalResult` (below) is checked the same way: its schema has no field
for any ground-truth or relevance label, so a persisted result can never
leak one.

## The `Retriever` abstraction

```python
class Retriever(Protocol):
    retriever_config_id: str
    def retrieve(self, query: RetrievalQuery, corpus: EvidenceCorpus, *, top_k: int) -> RetrievalResult: ...
```

Mirrors `AgentRunner`'s own minimal shape deliberately: one identity
attribute, one method, provider-agnostic. A future vector-embedding
retriever or a reranking retriever implements this exact interface without
it changing — and without a caller ever depending on a concrete class.

## Result contract

`RetrievalResult` (`case_id`, `query`, `candidates: list[RetrievedEvidence]`,
`top_k`, `retriever_config_id`) is validated, not just typed:

- `case_id` must match `query.case_id`.
- `candidates` must not exceed `top_k`.
- ranks must be dense, sequential, and in order: `1..N` with no gaps,
  duplicates, or reordering.
- no `evidence_id` may appear twice.

Each `RetrievedEvidence` carries `evidence_id`, `document_id`, `score`,
`rank`, and `retrieval_method` (a plain string identifying which method
produced *this* candidate — relevant once a future hybrid/reranked result
mixes candidates from more than one method).

`RetrievalResult` doubles as the persistable retrieval trace: every field
Phase 7's later persistence work needs (case ID, retriever config, top K,
ranked evidence/document IDs, scores, method) is already present, and
nothing evaluator-only (a relevance label, `required_evidence_ids`) can
enter it. Timing/latency provenance (mirroring `ExecutionTrace.latency_ms`)
is deliberately deferred until a `Retriever` is actually wired into a
traced pipeline in a later Phase 7 sub-phase — adding it now would be
speculative.

## `LexicalBaselineRetriever`

A deterministic token-overlap retriever, implemented **only** to validate
the query → corpus → ranked-candidates → result plumbing above. It is
**not** the Phase 7 experimental retrieval method and must never be tuned
against `AC-001`-`AC-030` — doing so would turn a plumbing check into an
overfit oracle.

Scores every corpus record by counting overlapping tokens between the
query's control text + scenario description and each evidence record's
title + content. Ties break on `evidence_id` ascending — never corpus
insertion order, and never `query.case_id`, which the scoring logic never
reads at all.

## Retrieval metrics

Computed in `app.retrieval.metrics`, independently of
`app.evaluation.metrics` (agent finding metrics). Both use `evidence_id`
identity — the same space `ExpectedOutcome.required_evidence_ids` uses —
never `document_id`.

**Recall@K** (per case):

```
recall_at_k = |unique required evidence_ids present among candidates with rank <= k|
              --------------------------------------------------------------------
              |unique required evidence_ids|
```

**MRR** (per case) — MRR has no single universal definition once a case
can have *more than one* required item, so this project's convention is
explicit:

```
MRR = 1 / rank_of_the_first-ranked candidate whose evidence_id is required
```

Only the earliest-ranked relevant hit counts — this is *not* an average
over every required item's individual rank (that would be a different
metric, "mean rank of all required items," and is not implemented). If no
required item is retrieved at any rank, `MRR = 0.0` for that case — a real,
measured failure.

**N/A convention** (both metrics, matching
`compute_required_evidence_recall`'s existing convention exactly): a case
with an empty `required_evidence_ids` returns `None`, not `0.0` — nothing
was required, so the metric is not applicable, not a measured failure.
Aggregation (`aggregate_recall_at_k`, `aggregate_mrr`) excludes `None`
cases from the mean's denominator entirely; only when *zero* cases are
applicable does the aggregate itself fall back to `0.0`.

## Relationship to the existing agent/evaluator pipeline

The transformation Phase 7B needs is:

```
RetrievalResult -> selected Evidence[] (app.retrieval.integration.select_evidence_for_agent)
                 -> existing AgentInput (unmodified)
                 -> existing AgentRunner (unmodified)
```

`AgentInput.evidence` is already `list[Evidence]` regardless of how that
list was chosen — today from `case.evidence_pool` via
`Dataset.evidence_for_case`, tomorrow from retrieval instead. Neither
`AgentInput` nor any `AgentRunner` implementation needs to change, and
`tests/unit/test_retrieval_agent_integration.py` demonstrates this
end-to-end (using `DeterministicBaselineAgent`, zero network).

One caveat, not a change to `AgentInput` itself:
`app.agent.interface.build_agent_input`'s own oracle-mode check
(`evidence_ids != set(case.evidence_pool)` → raise) exists specifically to
guarantee the *oracle* pipeline never silently sees a different evidence
set than the case defines — exactly the guarantee a real retrieval
pipeline must **not** have. Phase 7B will need its own, separate
constructor alongside `build_agent_input` (not a modification to it) once
real wiring lands.

## Future work (not implemented in Phase 7A)

- A real embedding-based retriever (OpenAI/other), implementing the same
  `Retriever` protocol.
- A reranking stage consuming `RetrievalResult.candidates` and producing a
  re-scored `RetrievalResult`.
- Wiring a `Retriever` into a traced pipeline (adding timing/latency
  provenance to the persisted result at that point).
- The separate, retrieval-aware `AgentInput` constructor described above.
- A real Phase 7 experiment run against `synthetic-v2`, once the above
  exists and is reviewed.

**Update:** every item above was later implemented (Phases 7B–8) and
run for real, once, against `synthetic-v2` — see `docs/RESULTS.md`
(Sections 3–5) for the preserved run IDs, artifact hashes, and metrics.
This section is left as written at the time for historical accuracy.
