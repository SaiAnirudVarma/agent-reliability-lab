# Regression Gate (Phase 9)

## Why regression cases exist

Phase 8's official, frozen reranked-evidence LLM baseline
(`results/reranked_llm_experiments/synthetic-v2__reranked-llm-baseline-v1__12457851-fd75-4f45-8e58-79d95c3e13a1.json`,
run `12457851-fd75-4f45-8e58-79d95c3e13a1`) produced 30 real,
non-fabricated outcomes against `synthetic-v2`. Six of those cases are
worth locking in as persistent checks:

- **Five wrong final findings**: `AC-004`, `AC-013`, `AC-019`, `AC-028`,
  `AC-030`.
- **One correct final finding with degraded evidence grounding**:
  `AC-016` (finding correct; `required_evidence_recall` dropped to 0.5
  despite both required items being present in the Top-5 reranked
  evidence).

Without a regression gate, a future code, prompt, or pipeline change
could make any of these six cases silently *worse* — for example,
turning a wrong-but-fully-grounded answer into a schema-invalid one, or
letting `AC-016`'s already-correct finding regress — and nothing in the
existing test suite would notice, because none of it evaluates against
real observed model behavior.

## Where they came from

Every floor in `regression/regression_manifest_v1.json` is read
directly off that one preserved artifact — never invented, never
tuned, never copied from hidden model reasoning. The manifest records:

- `source_run_id` / `source_artifact_sha256` / `source_config_id`:
  exactly which official run and artifact the floors were derived from,
  so anyone can re-derive or re-verify them from that file directly.
- Per case: the benchmark's own ground truth (`observed_expected_finding`,
  `observed_should_abstain` — the same `ExpectedOutcome` fields every
  evaluator in this project already uses), what the official run
  actually produced, and a set of **floors**:
  `min_schema_valid`, `min_citation_validity`,
  `min_required_evidence_recall`, and optionally
  `require_correct_finding` / `require_abstention_correct`.

`require_correct_finding` is `true` for **exactly one** case
(`AC-016`) — the only one of the six that was already correct in the
official run. It is `false` for the other five: **the manifest never
demands a case become correct that wasn't already**. See
"What CI does NOT prove" below.

## What CI checks

`scripts/run_regression_gate.py` reads two JSON files — the manifest
and a candidate evaluation artifact (any report with top-level
`dataset_version` and `results: list[EvaluationResult]` fields, e.g. a
`RunReport` or a `RerankedLLMReport`) — and, for each manifest case,
compares the candidate's `EvaluationResult` for that `case_id` against
the case's floors:

- `schema_valid` must still be `True` if `min_schema_valid` is `True`.
- `citation_validity` must not drop below `min_citation_validity`.
- `required_evidence_recall` must not drop below
  `min_required_evidence_recall` (when set).
- `correct_finding` must still be `True` if `require_correct_finding`
  is `True`.
- `abstention_correct` must still match `require_abstention_correct`
  when that field is set.
- A manifest case absent from the candidate's results is reported as
  **missing**, not silently skipped.
- The candidate's own `dataset_version`, when present, must match the
  manifest's — a provenance mismatch fails the whole gate before any
  per-case check is trusted.

The CLI exits `0` only if every check above holds; any regression,
missing case, or provenance mismatch exits non-zero and prints a
concise pass/fail report per case. CI (`.github/workflows/ci.yml`)
runs the full offline test suite and then runs this exact gate against
the preserved official Phase 8 artifact as a self-consistency check.

## What CI does NOT prove

- **CI does not prove model quality.** Passing the gate means "no
  further regression below an already-observed baseline" — it does
  not mean the agent's answers are correct, and five of the six
  manifest cases are *already wrong* by design (their `require_correct_finding`
  is `false`). A green CI run says nothing about whether those five
  cases have improved.
- **CI does not re-run any real model.** The gate is a pure,
  deterministic comparison over already-computed `EvaluationResult`
  fields from a JSON file. It never calls OpenAI, Cohere, or any
  embedding provider, and never re-executes retrieval, reranking, or
  an LLM agent.
- **CI does not claim these five failures are fixed.** That claim is
  explicitly not made anywhere in the manifest, the gate, or this
  document.
- **CI does not evaluate hidden reasoning.** Every check operates on
  already-produced, already-scored fields (`schema_valid`,
  `citation_validity`, `required_evidence_recall`, `correct_finding`,
  `abstention_correct`) — never on a model's raw chain-of-thought or
  unstructured output.

## How future official failures would be added

1. Run the relevant official, frozen evaluation pipeline (never inside
   this gate — the gate only reads an already-produced artifact).
2. Identify the case(s) worth locking in, and record their observed
   `EvaluationResult` fields as a new `RegressionCase` entry (or a new
   manifest version file, e.g. `regression_manifest_v2.json`, if the
   existing entries should be preserved unchanged) — with floors set
   to exactly what was observed, never a fabricated target.
3. Set `require_correct_finding` / `require_abstention_correct` only
   for a property that was **already true** in that observed run.
4. Point `source_run_id` / `source_artifact_sha256` / `source_config_id`
   at that exact preserved artifact.
5. Add/extend tests proving the new manifest's floors are read
   correctly, exactly as `tests/unit/test_regression_manifest.py` and
   `tests/unit/test_regression_gate.py` do for `regression_manifest_v1.json`.
