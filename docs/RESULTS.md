# Results

This document walks through every experiment in the order it was run,
distinguishing **retrieval-quality metrics** (does the pipeline surface
the right evidence) from **end-to-end agent metrics** (does the agent
reach the right conclusion given whatever evidence it was handed).
Every number below is read from a preserved, hash-verifiable artifact
or the current git history — nothing here is invented or rounded from
memory.

All experiments use dataset version **`synthetic-v2`** (30 cases,
`AC-001`–`AC-030`) unless stated otherwise.

## 1. Deterministic baseline

A rule-based agent (`app/agent/deterministic_baseline.py`) that reasons
generically over `Evidence.structured_fields` — no LLM call, fully
reproducible on demand, hence not preserved as an immutable artifact
(regenerable via `python scripts/run_eval.py`).

| Dataset | Accuracy | Macro F1 |
|---|---|---|
| `synthetic-v1` (10 cases) | 1.000 | 1.000 |
| `synthetic-v2` (30 cases) | 0.667 | 0.673 |

`synthetic-v2`'s additional 20 cases were specifically authored to
require genuine reasoning (conflicting policy versions, synthesis of
two individually-insufficient documents, distractor evidence) that a
field-matching rule set cannot handle — the accuracy drop is expected,
not a defect.

## 2. Frozen LLM baseline, curated (oracle) evidence

The agent receives exactly `case.evidence_pool` — the benchmark's own
hand-curated evidence for each case — via `build_agent_input`.

- Run ID: `556d6703-35d6-4aed-af63-862def0d5194`
- Artifact: `results/experiments/synthetic-v2__llm-baseline-v1__556d6703-35d6-4aed-af63-862def0d5194.json`
- Artifact SHA-256: `fa815140dbf182397aa02703a378885b42e509678dc47f355cebc925202f2d91`
- Model: `gpt-5.4-mini-2026-03-17` · Prompt version: `llm-baseline-v1` · Temperature: `0.0`
- Freeze tag: `benchmark-v2-frozen` → `2ace808cdb9c72f435126f4480b420e27d0b42e0`
- Result tag: `benchmark-v2-llm-baseline-result` → `14818e917059e477215543aa3c7c337f21620429`

| Metric | Value |
|---|---|
| Accuracy | 1.000 |
| Macro F1 | 1.000 |
| Exception precision / recall | 1.000 / 1.000 |
| Citation validity | 1.000 |
| Required-evidence recall | 0.983 |
| Schema validity | 1.000 |
| Abstention precision / recall / F1 | 1.000 / 1.000 / 1.000 |
| Latency P50 / P95 | 1798.25 ms / 3419.40 ms |
| Input / output tokens | 35,873 / 5,611 |

## 3. Vector retrieval baseline (no LLM, no reranking)

Measures retrieval quality **in isolation**: does the pipeline surface
the right evidence at all, before any agent ever sees it. Corpus =
full `synthetic-v2` version corpus (72 evidence records) —
deliberately *not* any single case's `evidence_pool`.

- Config: `configs/retrieval-baseline-v1.json` (`config_id = retrieval-baseline-v1`)
- Embedding: OpenAI `text-embedding-3-small`, 1536-dim, cosine similarity
- Run ID: `5b363b87-f452-4d1b-90e1-1f2cd1c2ef36`
- Artifact: `results/retrieval_experiments/synthetic-v2__retrieval-baseline-v1__5b363b87-f452-4d1b-90e1-1f2cd1c2ef36.json`
- Artifact SHA-256: `e2e234a2cc8339e4a3d15410f04fcd63a74a15b1317e428defc32fce4a737b05`
- Freeze tag: `retrieval-baseline-v1-frozen` → `e6c609548e9aa24f8f27862f5852cc55e7eafe68`
- Result tag: `retrieval-baseline-v1-result` → `2dcd290191f8441c9e3096831c3a5b9ca03ee92d`

| K | Recall@K |
|---|---|
| 1 | 0.239 |
| 3 | 0.500 |
| 5 | 0.800 |
| 10 | 0.967 |
| MRR | 0.557 |

Two cases (`AC-019`, `AC-021`) have a required evidence item that
never entered the Top-10 candidate set at all — a genuine
candidate-generation limitation, confirmed structurally distinct from
any later reranking or LLM-stage failure (see Sections 4 and 5).

## 4. Reranking experiment (Cohere `rerank-v4.0-pro`)

Consumes the **preserved** retrieval artifact from Section 3 (no
re-embedding); reorders its Top-10 candidates.

- Config: `configs/reranker-baseline-v1.json` (`config_id = cohere-rerank-v4-pro-baseline-v1`)
- Execution policy: `configs/reranker-execution-cohere-trial-v1.json` (7.0s minimum call interval — see `docs/incidents/2026-09-24-cohere-reranking-rate-limit.md`)
- Run ID: `81ac369a-9041-4d1c-af83-1c6d8ed22dec`
- Artifact: `results/reranking_experiments/synthetic-v2__cohere-rerank-v4-pro-baseline-v1__81ac369a-9041-4d1c-af83-1c6d8ed22dec.json`
- Artifact SHA-256: `b96422866e8cdbc8cdb7eb88323d2c9cdb24bd47b57e750c98b3256e2cf47a31`
- Freeze tag: `reranker-baseline-v1-frozen` → `efe0360451b970fea58993fa38c56b69a35fac28`
- Result tag: `reranker-baseline-v1-result` → `fe499559008e920be74f0c2ad62a97511e664073`

| K | Recall@K, before → after |
|---|---|
| 1 | 0.239 → 0.578 |
| 3 | 0.500 → 0.822 |
| 5 | 0.800 → 0.967 |
| 10 | 0.967 → 0.967 |
| MRR | 0.557 → 0.873 |

Recall@10 is unchanged by design — reranking within an unchanged
Top-10 candidate set cannot recover evidence that candidate generation
never supplied. `AC-019`/`AC-021`'s missing items remain missing after
reranking, exactly as expected.

## 5. Reranked Top-5 → LLM experiment (Phase 8)

The end-to-end question: given the reranked pipeline's Top-5 evidence
per case (not the curated oracle pool), does the LLM agent still reach
the right conclusion? `evidence_top_k = 5` was frozen *before* this
experiment ran, because Section 4 showed Recall@5 (0.967) already
equals Recall@10 — Top-10 would double evidence volume for zero
additional recall.

- Config: `configs/reranked-llm-baseline-v1.json` (`config_id = reranked-llm-baseline-v1`)
- Model: `gpt-5.4-mini-2026-03-17` · Prompt version: `llm-baseline-v1` (unchanged from Section 2) · Temperature: `0.0`
- Run ID: `12457851-fd75-4f45-8e58-79d95c3e13a1`
- Artifact: `results/reranked_llm_experiments/synthetic-v2__reranked-llm-baseline-v1__12457851-fd75-4f45-8e58-79d95c3e13a1.json`
- Artifact SHA-256: `e8ff346c2c2a28078fc143c77bfa472a8875179602e97486984e73af9020bdc1`
- Freeze tag: `reranked-llm-baseline-v1-frozen` → `a94002b0b50a1bc2a43030da32551e72c11e6114`
- Result tag: `reranked-llm-baseline-v1-result` → `87b0abb419f3a276058816543e751242de46c82f`

| Metric | Curated (oracle) evidence | Reranked Top-5 | Delta |
|---|---|---|---|
| Accuracy | 1.000 | 0.833 | −0.167 |
| Macro F1 | 1.000 | 0.836 | −0.164 |
| Exception precision / recall | 1.000 / 1.000 | 0.800 / 0.727 | −0.200 / −0.273 |
| Citation validity | 1.000 | 1.000 | 0.000 |
| Required-evidence recall | 0.983 | 0.967 | −0.017 |
| Schema validity | 1.000 | 1.000 | 0.000 |
| Abstention precision / recall / F1 | 1.000 | 0.889 | −0.111 |
| Latency P50 / P95 | 1798.25 ms / 3419.40 ms | 2394.95 ms / 5516.54 ms | +596.70 ms / +2097.14 ms |
| Input / output tokens | 35,873 / 5,611 | 45,640 / 5,695 | +9,767 / +84 |

**The central finding**: retrieval + reranking surfaced nearly all
required evidence (required-evidence recall only dropped 0.017), and
citation/schema validity stayed perfect — yet end-to-end decision
accuracy dropped 16.7 points. Retrieval-quality metrics alone would
have suggested this pipeline was nearly as good as the oracle; only
end-to-end evaluation revealed it wasn't.

25/30 findings were correct. Precisely:

- **5 incorrect final findings**: `AC-004`, `AC-013`, `AC-019`,
  `AC-028`, `AC-030`.
- **1 correct final finding with degraded evidence grounding**:
  `AC-016` — `required_evidence_recall` dropped to 0.5 despite both
  required items being present in its Top-5.

Layer-specific attribution (`app/integration/failure_attribution.py`,
using only already-produced artifact fields — never hidden model
reasoning):

| Case | Layer | Note |
|---|---|---|
| `AC-004` | LLM decision/synthesis | Fully cited evidence; wrong conclusion (PASS vs. expected EXCEPTION) |
| `AC-013` | LLM decision/synthesis | Fully cited evidence; wrong conclusion (EXCEPTION vs. expected INSUFFICIENT_EVIDENCE) |
| `AC-019` | Candidate-generation miss + LLM decision | One required item (`EV-1901`) never entered the Top-10 candidate set at all (pre-existing, see Section 3); the model abstained rather than reaching EXCEPTION on the remaining, fully-cited item |
| `AC-021` | Candidate-generation miss only (no downstream error) | Same class of missing candidate as `AC-019`, but the model still reached the correct finding |
| `AC-028` | LLM decision/synthesis | Fully cited evidence; wrong conclusion (PASS vs. expected EXCEPTION) |
| `AC-030` | Citation/grounding + wrong finding | Both required items presented in Top-5, but only one cited; wrong finding (EXCEPTION vs. expected PASS) |
| `AC-016` | Citation/grounding only (finding correct) | Both required items presented in Top-5, but only one cited |

No case in this run showed a malformed/schema-invalid output — every
one of the 30 findings was schema-valid.

## 6. Regression-gate extraction (Phase 9)

The five wrong-finding cases and `AC-016`'s grounding degradation were
turned into a versioned regression manifest
(`regression/regression_manifest_v1.json`, `manifest_version =
regression-manifest-v1`), whose floors were read directly from the
Section 5 artifact — never fabricated. `scripts/run_regression_gate.py`
checks any future evaluation artifact against those floors; CI
(`.github/workflows/ci.yml`) runs it against the preserved Section 5
artifact as a self-consistency check. See `docs/REGRESSION_GATE.md`
for the full design and explicitly what this gate does **not** prove.

## Provenance summary

| Phase | Config ID | Run ID | Freeze tag | Result tag |
|---|---|---|---|---|
| LLM baseline (oracle) | `llm-baseline-v1` | `556d6703-35d6-4aed-af63-862def0d5194` | `benchmark-v2-frozen` | `benchmark-v2-llm-baseline-result` |
| Retrieval | `retrieval-baseline-v1` | `5b363b87-f452-4d1b-90e1-1f2cd1c2ef36` | `retrieval-baseline-v1-frozen` | `retrieval-baseline-v1-result` |
| Reranking | `cohere-rerank-v4-pro-baseline-v1` | `81ac369a-9041-4d1c-af83-1c6d8ed22dec` | `reranker-baseline-v1-frozen` | `reranker-baseline-v1-result` |
| Reranked-LLM | `reranked-llm-baseline-v1` | `12457851-fd75-4f45-8e58-79d95c3e13a1` | `reranked-llm-baseline-v1-frozen` | `reranked-llm-baseline-v1-result` |

Every artifact above is tracked in git, immutable once written, and
independently re-verifiable by re-computing its SHA-256 and comparing
against the value recorded in the corresponding config/report.
