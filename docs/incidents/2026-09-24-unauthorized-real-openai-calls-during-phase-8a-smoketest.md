# Incident: unauthorized real OpenAI calls made while smoke-testing the Phase 8A CLI

**Date:** 2026-09-24
**Severity:** Medium — no benchmark, config, or preserved artifact was
modified or lost, but 30 real, unauthorized OpenAI API calls were made,
consuming real tokens, in direct violation of an explicit instruction that
this phase make zero real provider calls.
**Status:** Root cause identified. The resulting artifact was quarantined
(moved out of the repository) and never staged or committed. Awaiting
user direction on next steps.
**Classification: agent error (process failure), not a code defect and not
an experimental result.**

## What happened

While building Phase 8A (the reranked-evidence -> LLM integration layer),
the task's instructions were explicit: *"Do NOT run any real provider API
in this phase,"* *"NO real OpenAI calls,"* and, at the end, *"Do not run
the real LLM experiment."*

After writing `scripts/run_reranked_llm_eval.py`, the agent ran it directly
from a shell as a manual sanity check of the CLI's safety-ordering logic,
having first run `unset OPENAI_API_KEY` in that same shell command to (it
believed) guarantee no real provider could be constructed:

```
unset OPENAI_API_KEY; .venv/bin/python scripts/run_reranked_llm_eval.py \
    --config configs/reranked-llm-baseline-v1.json
```

The run printed configuration, metrics, and a written artifact path,
exiting 0 — i.e. it completed a full, real run, not a config-validation
failure. Inspecting the resulting artifact confirmed this was a genuine
real OpenAI execution:

| Field | Value |
|---|---|
| Cases run | 30 (all of `synthetic-v2`) |
| Provider / model | `openai` / `gpt-5.4-mini-2026-03-17` |
| `served_model_name` | `gpt-5.4-mini-2026-03-17` (a provider only reports this on a real response) |
| Per-case latency | 1193 ms – 5436 ms (real network latency; a fake/local provider is near-instant) |
| Total input tokens | 45,640 |
| Total output tokens | 5,700 |
| `estimated_cost` | `None` on every trace (no pricing env vars configured — never fabricated) |
| Run ID | `367b7b6f-024c-4fb1-a474-aee94185539c` |
| Git commit | `fe499559008e920be74f0c2ad62a97511e664073` |

## Root cause

`scripts/run_reranked_llm_eval.py` was written mirroring every other
experiment script in this project (`scripts/run_eval.py`,
`scripts/run_reranking_eval.py`), which by design auto-load a `.env` file
via `load_dotenv(dotenv_path=_DOTENV_PATH, override=False)` at import time.
`override=False` means: if a variable is **not already present** in the
process environment, `.env`'s value fills it in.

This repository's real `.env` file (used earlier in this project's
history for legitimately authorized real-provider runs, e.g. the frozen
`synthetic-v2` LLM baseline and the frozen reranking baseline) still
contains a live `OPENAI_API_KEY`. `unset OPENAI_API_KEY` in the invoking
shell removed the variable from that one command's environment, but
`load_dotenv` then re-populated it from `.env` before `_build_llm_agent`
ran — so the "safety check" the agent believed it was performing had no
effect, and the CLI proceeded to construct a real `OpenAIProvider` and
call it once per case, 30 times.

This is **not a bug in the CLI's own safety-ordering logic** (the
preflight steps — config load, source-artifact SHA/provenance
verification, dataset load, git SHA, run-ID collision check — all ran
correctly and in the documented order). It is an **agent process error**:
running a script that is *designed* to be able to make real provider
calls, directly against the real environment, as an ad hoc "does it wire
up correctly" check, instead of using the same in-process,
`ARL_DOTENV_PATH`-isolated, `FakeLLMProvider`-substituted test harness
pattern this project already uses for exactly this purpose (see
`tests/unit/test_run_eval_experiments.py`,
`tests/unit/test_run_reranking_eval_script.py`). That harness pattern
exists precisely so a script with real-provider capability can be
exercised with zero risk of a real call — it was available and was not
used for this manual check.

## What is known, and what is not

- **Exact case count and token counts are known** (read directly from the
  written artifact, reproduced in the table above) — this was a complete,
  successful 30/30 run, not a partial failure.
- **No benchmark, semantic config, execution config, or preserved
  upstream artifact (retrieval or reranking) was modified.** `git status`
  showed only newly-added, never-before-tracked files (the new Phase 8A
  source code itself, plus the stray result artifact) — nothing
  pre-existing was touched.
- **The resulting artifact was never staged or committed.** It has been
  moved out of the repository entirely (to the session's scratchpad
  directory) so it can never be mistaken for an authorized experiment
  result or accidentally included in a future commit.
- **Cost is not fabricated here**: the trace-level `estimated_cost` field
  was `None` on every case (no `LLM_INPUT_PRICE_PER_1K` /
  `LLM_OUTPUT_PRICE_PER_1K` env vars were configured), so no dollar figure
  is reported — only the real, observed token counts above.
- **The planned reranked Top-5 configuration was exposed to the real
  model.** The run used the pre-registered, pre-exposure
  `configs/reranked-llm-baseline-v1.json` design exactly as frozen
  (`evidence_top_k=5`, the same 5 reranked-evidence documents per case
  the official experiment would use) — i.e. the real model has now, in
  fact, seen this exact evidence-selection design once, even though the
  run itself carries no authorization or official status.

## Explicit statements

- **This was a real, complete run against OpenAI** — not simulated, not
  partial, not disputed.
- **No retry was attempted** and no second real run has been made.
- **The artifact was not preserved as an experiment result** — it carries
  no official status, was not validated against the pre-registered
  `evidence_top_k=5` freeze rationale in any reviewed sense, and must not
  be cited as this project's first reranked-LLM result.
- **No quality metric or per-case outcome from this accidental run has
  been inspected, summarized, or will be used for tuning or design
  decisions** — `evidence_top_k`, the prompt, the model, the benchmark,
  and the evidence-selection method all remain exactly as pre-registered
  before this run, and none of them will be changed based on anything
  this accidental run produced.
- **This run is NOT, and must never be treated as, an official
  experimental result** for Phase 8A or any later phase.
- **Nothing was hidden or silently cleaned up** — this incident is
  reported before any further Phase 8A work continued.

## Remediation

1. Every future manual/ad hoc check of a real-provider-capable CLI script
   in this project must set `ARL_DOTENV_PATH` to a nonexistent path (or
   otherwise guarantee `.env` is not consulted) rather than relying on
   `unset` of a single variable in one shell command, since `.env`
   auto-loading with `override=False` will silently repopulate it.
2. Prefer the project's own established in-process test harness
   (`importlib`-loaded module + `FakeLLMProvider`/`FakeReranker` +
   monkeypatched `_build_*_agent`) for any "does this wire up correctly"
   check of a script capable of real provider calls — never a direct
   shell invocation against the real environment, even with an
   apparently-cleared credential.
3. No change to `scripts/run_reranked_llm_eval.py`'s own safety-ordering
   logic is needed or made as a result of this incident -- it behaved
   exactly as designed once a credential was available; the failure was
   in how, and against what environment, it was invoked for testing.
