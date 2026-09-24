# Phase 7C Reranking Design (Foundation Only)

No real reranker model, no network call, and no production reranking
experiment exist yet. This covers contracts, the `Reranker` interface, a
deterministic pass-through baseline, before/after metrics, and a
provenance schema for a future real experiment.

## Two-stage architecture

```
query -> frozen vector retriever -> candidate set (Top-N) -> reranker -> reranked candidates -> retrieval metrics
```

Candidate generation (`app.retrieval`) and reranking (`app.reranking`) are
independently measurable: a reranker never re-embeds anything and never
sees ground truth; it only reorders (and optionally trims) a candidate set
it is handed.

## Contracts

`RerankInput` reuses `RetrievalQuery` and `RetrievedEvidence` directly
rather than redefining equivalent types -- the candidate set a reranker
receives *is* a retriever's output, unchanged. `candidate_depth` records
the exact size of that set explicitly.

`RerankResult` embeds the `RerankInput` it came from, so it is
self-checkable: `case_id` must match, `reranked_rank`s must be dense and
sequential, no `evidence_id` may repeat, and — the critical invariant —
**output evidence_ids must be a subset of input evidence_ids**. A
reranker's output *may* be smaller than its input (a trimmed final K), but
can never contain anything the candidate set didn't already have.
`original_rank`/`original_score` on each `RerankedEvidence` are also
checked against the input's own recorded values, so provenance can't be
silently altered.

## `Reranker` interface

```python
class Reranker(Protocol):
    reranker_config_id: str
    def rerank(self, rerank_input: RerankInput, *, final_k: Optional[int] = None) -> RerankResult: ...
```

Mirrors `Retriever`'s shape. `final_k=None` keeps the whole candidate set,
reordered; an explicit value trims to that many — this is how "vector
Top-10 → reranker → final Top-5" gets expressed without the reranker
choosing the number itself.

## `PassThroughReranker`

Preserves the incoming order exactly (`reranker_score == original_score`,
`reranked_rank == original_rank` when untrimmed). Exists purely as a
control condition: proof that running candidates through the reranking
pipeline, by itself, produces zero metric delta. Contains no
lexical/entity/case-specific logic.

## Metrics

`app.reranking.metrics` mirrors `app.retrieval.metrics`'s Recall@K/MRR
conventions exactly, applied to `reranked_rank`. `compute_delta(before,
after)` is one generic helper for both Recall@K and MRR deltas (`None` if
either side is `None`).

`compute_required_evidence_rank_movement` tracks each required evidence
item's rank before/after, and distinguishes two failure modes that must
never be conflated:

- `in_candidate_set=False` — candidate generation never supplied this
  item at all. Not attributable to the reranker; `rank_delta` is `None`.
- `in_candidate_set=True`, `reranked_rank=None` — the reranker's own
  output dropped it (e.g. by trimming). A genuine reranking-stage outcome.

End-to-end `recall_at_k`/`mrr` still correctly score `0`/low for missing
required evidence regardless of which failure mode caused it — the
distinction exists for *diagnosis*, not to hide the effect from the
headline metric.

## Provenance (schema only)

`RerankExperimentProvenance` records everything needed to answer "exactly
which frozen retrieval result produced the candidate set this reranker
evaluated?" (`retriever_config_id` + `retrieval_result_run_id`), plus this
reranking run's own `run_id`/`git_commit_sha`/dataset and corpus
fingerprints/`reranker_config_id`/`final_k_values`. Reranker
provider/model fields are `None` for a pass-through or other local
reranker — never fabricated. No runner produces one of these yet.

## Reusing a preserved retrieval artifact

`app.reranking.artifact_loader.load_preserved_retrieval_artifact` reads a
`results/retrieval_experiments/*.json` file, verifies its SHA-256 (and
any other `expected_*` provenance field the caller supplies) before
parsing it as a `RetrievalReport`, and fails closed on any mismatch. It
has no write path — a future reranking experiment can reuse frozen,
already-paid-for candidate lists (`RetrievalResult.candidates`) without
a second embedding call, but this loader itself never rewrites or
normalizes the artifact.

## Not yet implemented

- A real reranker (cross-encoder or API-based).
- A runner/CLI that produces an immutable `RerankExperimentProvenance`
  and a persisted reranking report.
- Any choice of candidate depth `N` or final `K` — those are experimental
  decisions for a future phase, not fixed here.
