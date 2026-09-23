# Agent Reliability Lab

Independent reliability and evaluation harness for a synthetic enterprise audit agent. This is a portfolio project, not a reproduction of any employer's proprietary system.

*(This README currently documents only the dataset and known limitations. A full README — architecture, install/run instructions, metrics, roadmap — is still pending; the CLI vertical slice itself is complete, but a comprehensive walkthrough has not yet been written.)*

## Dataset

`datasets/` contains a small, entirely **synthetic** control-testing dataset, organized into named versions via `datasets/versions.json`: `controls.json` (11 control definitions), `evidence.json` (72 evidence records), and `eval_cases.json` (30 evaluation cases, `AC-001`–`AC-030`). `synthetic-v1` is the original 10-case benchmark (`AC-001`–`AC-010`); `synthetic-v2` is the full 30-case set, adding 20 additional, harder cases (`AC-011`–`AC-030`) on top of the unmodified `synthetic-v1` cases. No real company data, employee data, or proprietary control language is used anywhere in this repository.

Each case pairs a neutral scenario description with a fixed pool of evidence (`EvaluationCase.evidence_pool`) — including, in most cases, at least one deliberate distractor (a wrong-period document, a wrong-employee record, or an unrelated document) so that a correct answer requires actually inspecting the evidence rather than receiving one pre-curated document.

**Evidence shown to the agent never contains the ground truth.** The correct answer lives exclusively in `EvaluationCase.expected_outcome` (an `ExpectedOutcome`: the expected finding, which evidence is genuinely required to reach it, whether abstention is correct, and a rationale) — a model the agent under test never sees. `Evidence.content` (prose) and `Evidence.structured_fields` (machine-readable facts) always describe the same underlying facts, so both a future LLM-based agent and the deterministic baseline agent can reason from the same evidence base without contradiction.

The original 10 `synthetic-v1` cases exercise: a timely baseline pass, a late-review exception, missing review evidence, an SLA violation and an SLA-compliant counterpart, an incomplete evidence artifact, a stale-vs-current policy conflict, a case requiring synthesis of two individually-insufficient documents, a wrong-reporting-period distractor, and an unavailable evidence/tool-access path. See `app/datasets/loader.py` for how the dataset is loaded, validated, and cross-referenced.

## Known limitations / technical debt

- **`Evidence.structured_fields` relies on an informal field-name vocabulary, not schema-enforced typed fact structures.** The deterministic baseline agent (`app/agent/deterministic_baseline.py`) reasons generically over field names like `trigger_timestamp`, `completion_timestamp`, `all_confirmed_appropriate`, `exceptions_found`, `policy_version`/`effective_start`/`effective_end`, and `employee_id`/`account_id` — but nothing in the `Evidence` Pydantic model ties an evidence author to that vocabulary. A future evidence record using a slightly different key name would be silently treated as "not applicable" by every evaluator rather than raising an error. This is an acceptable trade-off for a small, hand-authored synthetic dataset (72 records as of `synthetic-v2`), but it must be revisited — e.g. with typed per-control-category fact schemas — before the system supports heterogeneous or externally-ingested evidence at scale.
