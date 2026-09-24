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

## Update: Cohere adapter, frozen baseline, and orchestration skeleton

Since the "Not yet implemented" list above was written, the following
were added, reviewed, and (for the adapter) compatibility-tested with
exactly one real request — but **no official reranking experiment has
been run against synthetic-v2**:

- `app.reranking.cohere_reranker.CohereReranker` — a real `Reranker`
  implementing provider (isolates Cohere's index-based response shape;
  constructed via dependency-injected client, never importing `cohere`
  itself except in the separate `build_cohere_client` factory).
- `configs/reranker-baseline-v1.json` / `app.reranking.baseline_config` —
  a frozen, non-secret, `extra="forbid"` configuration (`config_id =
  cohere-rerank-v4-pro-baseline-v1`) naming the candidate depth (10),
  output depth (10), K values (`[1, 3, 5, 10]`), and — critically — the
  exact preserved retrieval artifact (by run ID + SHA-256 + retriever
  config ID) that must supply its candidates. `served_model` is `null`,
  matching the compatibility check's own finding that Cohere's rerank API
  does not authoritatively report one.
- `app.reranking.candidate_source` — validates that preserved artifact
  (hash, dataset/corpus fingerprint, retriever config ID, run ID, git
  provenance, exactly 30 results, sufficient candidate depth per case)
  and truncates it to `candidate_depth`, all before any provider call.
- `app.reranking.runner.run_reranking_evaluation` — the before/after
  orchestration (Recall@K/MRR before vs. after, deltas, required-evidence
  rank movement), operating only on already-preserved candidates -- no
  embedding calls, no re-retrieval.
- `scripts/run_reranking_eval.py` — the CLI skeleton enforcing the full
  safety order (config → candidate-source validation → git SHA → run ID
  → immutable output path → collision check → explicit real-API
  authorization → *only then* reranker construction and the real
  provider call), writing to `results/reranking_experiments/`.

**Update (Phase 7D, after this design doc was written):** this script
was later run for real, once, with a real Cohere credential, against
the frozen `synthetic-v2` candidate set. That result is preserved —
see `docs/RESULTS.md` (Section 4) for the run ID, artifact hash, and
Recall@K/MRR before vs. after reranking. The candidate depth `N`/final
`K` frozen in `configs/reranker-baseline-v1.json` were never re-derived
or tuned after seeing that result.
