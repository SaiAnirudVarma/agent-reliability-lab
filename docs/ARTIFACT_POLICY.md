# Artifact Policy

This policy exists because of
[`docs/incidents/2026-09-23-llm-baseline-artifact-deletion.md`](incidents/2026-09-23-llm-baseline-artifact-deletion.md):
the one real LLM experiment result this project had was permanently lost to
a test that manipulated the production results path directly. Every rule
below traces back to preventing a repeat of that specific failure mode.

## Three categories of artifact

### A. Immutable experiment artifacts

Real outputs of a genuine experiment against a real, paid model — most
importantly, any `--agent llm` run of `scripts/run_eval.py`. Once produced,
these must **never** be overwritten, deleted, or "corrected." They are not
reproducible on demand: reproducing one costs real money and real API
calls, and even then would be a *new* run, not a recovery of the old one,
because hosted models are stochastic and can change server-side over time
(see `served_model_name` in `ExecutionTrace` — a rolling model alias can
resolve to a different snapshot on a later date).

Rules:
- Never delete. Never `unlink()` from code, test or otherwise, without a
  human explicitly asking for that specific file to be removed.
- Never silently overwrite. `scripts/run_eval.py` fails closed (non-zero
  exit, no write) if an `--agent llm` run's destination file already
  exists, unless `--force-overwrite` is explicitly passed (development use
  only — never pass this to preserve a result you might want later).
- Should be committed to git once produced (see `.gitignore` — real LLM
  result files are deliberately **not** in any ignore rule).
- Referenced by run ID in any analysis or report, not by "the file that
  happened to be at this path."

### B. Regenerable artifacts

Deterministic outputs: any `DeterministicBaselineAgent` run, for any
dataset version. Rerunning `scripts/run_eval.py` (no `--agent llm`)
produces byte-for-byte equivalent results (same dataset, same code, same
rules), so these carry no information a rerun can't reproduce.

Rules:
- Freely overwritten by default — no confirmation, no `--force` flag
  needed. Requiring one would be friction with no safety benefit.
- Ignored by git (`results/baseline.json`, `results/deterministic-*.json`
  in `.gitignore`) — regenerating on demand is cheaper and more reliable
  than diffing stale copies.
- If a deterministic result is ever worth preserving permanently (e.g. as
  a comparison baseline for a paper or report), copy it somewhere outside
  `results/` and commit that copy explicitly — don't rely on it surviving
  in the gitignored working path.

### C. Temporary test artifacts

Anything a test creates to exercise the CLI or the loader. Must live
**only** under a pytest-managed temporary directory (`tmp_path` /
`tmp_path_factory`), never under the repository's real `results/`.

Rules:
- Every CLI subprocess a test launches must set `ARL_RESULTS_DIR` (see
  `scripts/run_eval.py`) to an isolated temporary directory. This is not
  optional style guidance — `tests/integration/conftest.py` enforces it at
  the session level: it hashes the real `results/` directory before and
  after the whole test session and fails the suite if anything in it
  changed.
- Test cleanup is handled entirely by pytest's temp-directory lifecycle
  (automatic; do not hand-write `finally: path.unlink()` against any path
  that isn't itself inside a `tmp_path`).
- No test may assume, depend on, or read the *content* of a real artifact
  in `results/` for its assertions — if a test needs a `RunReport`-shaped
  fixture, it builds or copies one into its own temp directory.

## Future real-experiment naming

For real LLM experiments going forward, prefer a unique, collision-proof
filename over the current fixed `results/llm-baseline.json` path:

```
results/experiments/<dataset-version>__<agent-config-id>__<run-id>.json
```

Example:

```
results/experiments/synthetic-v2__llm-baseline-v1__3de6e897-872f-47a2-8f0c-82d858d552fb.json
```

This makes every real result inherently collision-proof (a run ID is a
UUID) — the overwrite protection in `scripts/run_eval.py` becomes a
second, redundant safety net rather than the only one. A convenience alias
such as `results/experiments/latest.json` (a symlink or a small copy) may
exist for quick access, but it must **never** be the sole copy of an
experiment — the run-ID-named file is always the source of truth.

**Scope note:** this experiments/ directory and naming scheme are not yet
wired into `scripts/run_eval.py` — this document describes the intended
direction, adopted as *policy* now so that when Phase 6 LLM experimentation
resumes, the CLI change to produce these paths is a small, reviewable diff
against an already-agreed design, not an on-the-spot decision made under
time pressure the way the original fixed-path design was.

## Experiment manifest (required before any future real LLM run)

Before running a real experiment, record — and after running it, extend
with the run ID:

| Field | Source |
|---|---|
| Git commit SHA | `git rev-parse HEAD` at the time of the run |
| Dataset version | e.g. `synthetic-v2` |
| Dataset fingerprint | `Dataset.fingerprint` (SHA-256; see `app/datasets/loader.py`) |
| Prompt version | `LLMConfig.prompt_version` |
| Requested model | `MODEL_NAME` |
| Temperature | `LLMConfig.temperature` |
| Max output tokens | `LLMConfig.max_output_tokens` |
| Agent config ID | `ExecutionTrace.agent_config_id` |
| Evaluator version | the evaluator/metrics module's state at commit time (tie to the same commit SHA above until a dedicated version string exists) |
| Timestamp | wall-clock time of the run |
| Run ID | assigned by `run_evaluation`; only known after execution |

This gives every future experimental result full provenance back to the
exact source code, dataset content, and configuration that produced it —
something the lost `3de6e897-872f-47a2-8f0c-82d858d552fb` run does not have,
because none of this infrastructure (including git commits at all — see
`docs/DEVELOPMENT_CHECKPOINTS.md`) existed yet when it ran. Its incident
record says so explicitly rather than fabricating a commit SHA after the
fact.

## `.gitignore` rule (do not regress this)

```gitignore
results/baseline.json
results/deterministic-*.json
!results/.gitkeep
```

Only these two *regenerable* filename patterns are ignored. Everything
else under `results/` — most importantly a real `llm-baseline.json` or
anything under `results/experiments/` — is deliberately left trackable.

**Never reintroduce a blanket `results/*.json` rule.** That exact rule is
what made the original `results/llm-baseline.json` loss permanent: even if
a commit had existed, that file would never have been eligible for git to
track, so there would have been nothing to recover it from either way.
