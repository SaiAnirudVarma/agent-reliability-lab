# Development Checkpoint Policy

Adopted 2026-09-23, directly in response to
[`docs/incidents/2026-09-23-llm-baseline-artifact-deletion.md`](incidents/2026-09-23-llm-baseline-artifact-deletion.md).
The repository had **zero commits** at the time that incident occurred — the
only copy of the lost experiment result existed in the working directory,
with no version-control history to fall back on. This policy exists so that
sentence can never be true again.

## Checkpoint A — Component/phase implementation complete

When a component or phase's implementation is done and reviewed:

1. Run the full test suite.
2. Run a secret scan over what's about to be staged (see below).
3. Commit.

This is the routine "save your work" checkpoint — frequent, low-ceremony.

## Checkpoint B — before every real LLM experiment

Before spending any real API call:

1. Commit the frozen code + dataset + prompt that will produce the run —
   nothing about the configuration that determines the experiment's
   result should be uncommitted working-tree state at the moment the API
   call is made.
2. Tag that commit to mark it as an experiment's input state (e.g.
   `experiment-input-<short-description>`).

This guarantees that even if the *output* is later lost, the exact *inputs*
that would reproduce it (modulo the model's own stochasticity) are
permanently recoverable from git.

## Checkpoint C — after every real LLM experiment

Immediately after a real run completes:

1. Save the immutable, run-ID-named artifact (see `docs/ARTIFACT_POLICY.md`
   category A).
2. Validate it: `RunReport.model_validate_json(...)` against the saved
   file, before considering the run "done."
3. Commit the experiment artifact and/or its manifest metadata.
4. Tag the resulting commit as the result state (e.g.
   `experiment-result-<short-description>`).

Checkpoints B and C together mean a real experiment's full lifecycle —
exact inputs, exact output — is captured in git, not left to survive only
in a working directory or a conversation transcript.

## Checkpoint D — before major architecture changes

Before a change that touches core contracts, the evaluation pipeline, or
dataset structure (e.g. the Phase 6 dataset-versioning work): commit the
current working state first, so the change is a reviewable diff against a
known-good point, not an unrecoverable in-place mutation.

## Checkpoint E — application-ready release

When the project reaches a coherent, demonstrable state: tag it with a
version (e.g. `v0.1.0`) per normal semantic-versioning practice.

## What this prevents

> "The only copy existed in the working directory."

Every checkpoint above exists to make that sentence false at every stage of
this project's life — not just for real experiment outputs (Checkpoints B/C),
but for the code and configuration that produced them (Checkpoint A/D) and
for milestones worth being able to return to (Checkpoint E).

## Non-goals

This policy does not ask for commits on every file save, and it does not
ask for a clean, linear, curated history — see the git safety rules already
in force for this project (never force-push, never rewrite history to make
a timeline look cleaner than it honestly was, never amend a commit instead
of creating a new one). A checkpoint commit's job is to exist and be
truthful about what it contains, not to be pretty.
