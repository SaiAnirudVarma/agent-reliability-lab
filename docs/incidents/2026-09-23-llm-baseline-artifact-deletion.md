# Incident: `results/llm-baseline.json` deleted by an integration test

**Date:** 2026-09-23
**Severity:** High — an irreplaceable experimental artifact was permanently lost.
**Status:** Root cause fixed (test isolation). Artifact NOT recovered.

## What was lost

The official, first (and at the time of loss, only) real LLM baseline run
against `synthetic-v1`.

| Field | Value |
|---|---|
| Run ID | `3de6e897-872f-47a2-8f0c-82d858d552fb` |
| Provider | OpenAI |
| Requested model | `gpt-5.4-mini-2026-03-17` |
| Prompt version | `llm-baseline-v1` |
| Temperature | 0.0 |
| Max completion tokens | 800 |
| Dataset | `synthetic-v1` |
| Cases | 10 (AC-001–AC-010) |
| Known aggregate result | Finding accuracy, macro F1, exception precision/recall, citation validity, schema validity, abstention precision/recall/F1 — **all reported as 1.000** |
| Known totals | Input tokens: 12,023. Output tokens: 1,889. |
| Known latency | P50 ≈ 2357.2 ms, P95 ≈ 3177.2 ms (using the percentile implementation in effect at the time — see the percentile-correction work elsewhere in Phase 6; this P95 figure has NOT been recomputed against the corrected implementation, since the underlying per-trace latency data no longer exists to recompute it from) |

These figures are transcribed from the conversation in which the run was
originally reported, not from the artifact itself (which is gone). They are
**reported measurements that survive in the experiment log**, not a
machine-readable, independently re-verifiable record.

## Root cause

`tests/integration/test_cli.py`, in the `TestCliCompare` class (written
during Phase 5, before any official LLM baseline existed), contained:

```python
def test_compare_without_llm_results_fails_clearly(self):
    if LLM_RESULTS_PATH.exists():
        LLM_RESULTS_PATH.unlink()
    ...
```

`LLM_RESULTS_PATH` was a module-level constant pointing directly at the
**production** path `results/llm-baseline.json`. When this test was
written, no real artifact existed at that path, so the unconditional
`unlink()` was harmless — it just guaranteed a clean "file absent" starting
state for the test's own assertion.

Once Phase 5 produced the real official artifact at that exact path, this
test became a landmine: any full `pytest` run would delete it, with no
confirmation, no backup, and no distinction between "a test fixture at this
path" and "the one real experimental result at this path." The adjacent
test, `test_compare_with_both_files_present_prints_a_table`, compounded the
risk by overwriting the same path with a synthetic copy and deleting it
again in its own `finally` block.

During Phase 6 development, the full test suite was run to verify unrelated
changes (dataset versioning, `required_evidence_recall`). That run silently
deleted the artifact. The loss was not noticed until a `results/` directory
listing several steps later.

## Explicit statements

- **The real experiment was NOT rerun.** No additional OpenAI API request
  was made in response to this incident.
- **The JSON artifact was NOT reconstructed.** No file was fabricated or
  hand-assembled to stand in for the original `results/llm-baseline.json`.
- **The original machine-readable artifact is unavailable.** There is no
  copy on this machine, and it was never committed to version control (the
  repository had zero commits at the time of the run — see
  `docs/DEVELOPMENT_CHECKPOINTS.md` for the policy adopted specifically to
  prevent this from being possible again).
- The aggregate numbers and totals above **survive only in the conversation
  log** where the run was originally reported. They are not independently
  re-verifiable against a machine-readable source now, and must not be
  treated as equivalent to the original artifact for any future analysis
  that would require the artifact itself (e.g. recomputing a metric that
  didn't exist yet at the time of the run, such as `required_evidence_recall`,
  or reapplying the corrected percentile implementation to the real
  per-trace latencies).
- This run **predates repository checkpointing** entirely — no commit SHA,
  dataset fingerprint, or experiment manifest exists for it, because none
  of that infrastructure existed yet when it was executed. Any future
  reference to this run must note it has no verifiable provenance beyond
  this document and the conversation transcript.

## Remediation

1. `tests/integration/test_cli.py` was refactored so no test reads, writes,
   or deletes anything under the repository's real `results/` directory —
   every CLI subprocess launched by a test is redirected to a pytest
   `tmp_path` via a new environment-variable override
   (`ARL_RESULTS_DIR`), mirroring the existing `ARL_DOTENV_PATH` pattern.
2. `.gitignore`'s blanket `results/*.json` rule (which meant no experiment
   artifact — including the one lost here — was ever eligible for git
   tracking, even if a commit had been made) was replaced with narrow
   rules that ignore only the regenerable, deterministic outputs
   (`results/baseline.json`, `results/deterministic-*.json`) and leave real
   experiment artifacts trackable.
3. The CLI now fails closed (refuses to overwrite, non-zero exit) instead
   of silently overwriting an existing real LLM experiment result file,
   unless an explicit override flag is passed.
4. See `docs/ARTIFACT_POLICY.md` and `docs/DEVELOPMENT_CHECKPOINTS.md` for
   the durable policies adopted as a result of this incident.

## What would have prevented this

Any one of the three remediations above, independently, would have
prevented this specific loss:
- If the test had used a temporary directory instead of the production
  path, it could never have touched the real file.
- If `.gitignore` had not blanket-excluded `results/*.json`, an earlier
  commit could have captured the artifact even without any other fix.
- If the CLI had refused to let anything overwrite/delete an existing
  real-experiment result file, the test's `unlink()` call would still have
  been a bug, but the file's actual removal would have required the same
  explicit override a human would need — not a bare, unconfirmed
  `Path.unlink()` reachable from routine test execution.
