# Incident: first official Cohere reranking run failed on trial-key rate limiting

**Date:** 2026-09-24
**Severity:** Low — no data was lost; no artifact, benchmark, config, or
source-retrieval file was created, modified, or deleted.
**Status:** Classified. Execution hardening addressed separately (see
`configs/reranker-execution-cohere-trial-v1.json` and `app.reranking.pacing`).
**Classification: execution/instrumentation failure, NOT an experimental result.**

## What happened

The first attempt at the official, frozen Cohere reranking evaluation
(`configs/reranker-baseline-v1.json`, `config_id =
cohere-rerank-v4-pro-baseline-v1`) against the preserved
`synthetic-v2` retrieval candidates was run via
`scripts/run_reranking_eval.py` at:

| Field | Value |
|---|---|
| Frozen reranker commit | `efe0360451b970fea58993fa38c56b69a35fac28` |
| Frozen reranker tag | `reranker-baseline-v1-frozen` |
| Intended case count | 30 (`AC-001`–`AC-030`) |
| Provider | Cohere (trial API key) |
| Requested model | `rerank-v4.0-pro` |

The run failed partway through with:

```
HTTP 429
"You are using a Trial key, which is limited to 10 API calls / minute."
```

## What is known, and what is not

- **Exact successful case-call count: NOT instrumented, and therefore
  unknown.** The runner at the time had no per-case progress logging, and
  the exception propagated out of the per-case loop before any partial
  state could be captured or persisted.
- **"Approximately 10" is an inference only**, based on Cohere's own error
  message stating the trial key's limit is 10 calls/minute and the loop
  issuing calls back-to-back with no pacing between them. It is **not a
  measured value** and must never be represented as one in any later
  analysis.
- **The `run_id` for this attempt was generated internally (before the
  reranker was ever invoked) but was never printed or persisted anywhere**,
  since the CLI at the time only printed `run_id` after a successful run.
  It is not recoverable from any artifact, log, or file. It is not
  reconstructed or guessed here.
- **No result artifact was written.** `write_reranking_report_atomically`
  is only reached after `run_reranking_evaluation` returns successfully;
  the exception occurred before that point. `results/reranking_experiments/`
  did not exist after the failure.
- **No benchmark, config, or source-retrieval-artifact file was mutated.**
  Confirmed directly: `git status` was clean, `results/` file hashes were
  byte-identical before and after the attempt, `configs/reranker-baseline-v1.json`
  and the preserved retrieval artifact
  (`results/retrieval_experiments/synthetic-v2__retrieval-baseline-v1__5b363b87-f452-4d1b-90e1-1f2cd1c2ef36.json`)
  were both unchanged.

## Explicit statements

- **No retry was attempted.** The run was not re-invoked after the failure.
- **No second official experiment was started.**
- **Nothing was manually patched or fabricated** to stand in for a
  completed run — no partial result, no invented metric, no reconstructed
  `run_id`.
- **No aggregate metric (Recall@K, MRR, or any delta) was calculated or
  reported** for this attempt — there is no completed run to compute one
  from.

## Root cause

The Cohere account in use is provisioned with a **trial** API key, which
Cohere enforces at 10 requests/minute. `scripts/run_reranking_eval.py`'s
per-case loop issued one `rerank()` call per case with no delay between
calls, so it exceeded that quota well before completing all 30 cases.

This is an **execution/infrastructure limitation**, not a finding about
the reranker's quality, the frozen candidate set, or the benchmark. It says
nothing about Recall@K, MRR, or ranking behavior, because no such metric
was ever computed.

## Remediation (tracked separately, Phase 7D.1)

1. `app.reranking.pacing.CallPacer` enforces a minimum interval between
   provider-call *start* times (not response-to-next-call), read from a
   new, separate, non-secret execution-policy config
   (`configs/reranker-execution-cohere-trial-v1.json`,
   `execution_policy_id = cohere-trial-pacing-v1`,
   `minimum_call_start_interval_seconds = 7.0`) — kept deliberately
   distinct from the frozen *semantic* reranker baseline config, which this
   incident's remediation never touches.
2. `run_id` and the future immutable artifact path are now printed by the
   CLI *before* the first provider call, so a future failure still leaves
   the operator able to identify which attempted run failed.
3. Per-case progress (`case N/30: starting` / `completed`) and a structured
   failure summary (attempted/completed case counts, the case ID in flight
   at failure, exception category, `artifact_written: false`) are now
   emitted on any provider failure.
4. `max_attempts_per_case` remains `1` — no automatic retry was added or is
   planned; recovery from any future provider failure remains a human
   decision.

## What would have prevented this

Pacing calls to stay under the trial key's published rate limit would have
prevented this specific failure. It would not have been prevented by
anything already in place before this incident, since rate limiting is an
account/tier property Cohere enforces server-side, not a client-side bug.
