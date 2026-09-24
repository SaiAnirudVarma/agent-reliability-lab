#!/usr/bin/env python3
"""Run a reranking-only evaluation using a frozen reranker baseline config
and a preserved retrieval artifact as the candidate source.

Usage:
    python scripts/run_reranking_eval.py --config configs/reranker-baseline-v1.json

No embedding calls, no LLM calls, no re-retrieval: the candidate source is
an already-preserved retrieval artifact (see app.reranking.candidate_source).

Safety ordering, ALWAYS in this order, before making the first real
reranker provider call (``reranker.rerank(...)``):

    1. loads the frozen reranker config
    2. locates the preserved retrieval artifact (path reconstructed from
       the config's own dataset_version/retriever_config_id/run_id --
       never a second, independently-guessed path)
    3. SHA-256 + provenance verifies that artifact against the config's
       recorded expectations (dataset/corpus fingerprint, retriever
       config ID, run ID, git provenance, exactly 30 results, every case
       has at least candidate_depth candidates) -- see
       app.reranking.candidate_source.load_and_validate_candidate_source
    4. constructs the reranker provider object (no network -- just
       __init__, mirroring every other provider adapter's lazy-no-network
       construction)
    5. resolves git_commit_sha (the SAME utility every other experiment
       script in this project uses)
    6. resolves run_id (a fresh UUID4, or ARL_RUN_ID for tests)
    7. computes the immutable, run-ID-qualified destination artifact path
       and checks it for a collision

Any failure at 1-7 aborts (exit 1) BEFORE step 8 (the real reranker calls)
is ever reached.

Exit codes:
    0  the run completed.
    1  the run could not complete: bad configuration, candidate-source
       validation failure, missing git provenance, a destination
       collision, or an infrastructure-level exception during reranking.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from app.datasets.loader import DatasetError, load_dataset
from app.observability.git_provenance import get_git_commit_sha
from app.reranking.artifact_io import (
    reranking_experiment_path,
    resolve_run_id,
    write_reranking_report_atomically,
)
from app.reranking.baseline_config import RerankerBaselineConfig, load_reranker_baseline_config
from app.reranking.artifact_loader import ArtifactVerificationError
from app.reranking.candidate_source import CandidateSourceValidationError, load_and_validate_candidate_source
from app.reranking.runner import RerankingReport, run_reranking_evaluation
from app.retrieval.artifact_io import retrieval_experiment_path
from app.retrieval.corpus import build_full_version_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "datasets"

# Where THIS run's own output is written -- mirrors every other
# experiment script's ARL_RESULTS_DIR convention.
RESULTS_DIR = Path(os.environ.get("ARL_RESULTS_DIR", str(REPO_ROOT / "results")))

# Where the candidate-source retrieval artifact is READ from. Deliberately
# a SEPARATE override from ARL_RESULTS_DIR: a test must be able to point
# this at a fabricated retrieval artifact without needing that fixture to
# also be this run's own output directory (and without ever touching the
# real repository's real preserved artifact).
SOURCE_RESULTS_DIR = Path(os.environ.get("ARL_SOURCE_RESULTS_DIR", str(REPO_ROOT / "results")))

_DOTENV_PATH = Path(os.environ.get("ARL_DOTENV_PATH", str(REPO_ROOT / ".env")))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a core dependency; defensive only
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(dotenv_path=_DOTENV_PATH, override=False)


class ConfigError(Exception):
    """A CLI-level configuration problem -- print and exit 1 before
    anything expensive runs, matching every other experiment script."""


def _build_reranker(model_name: str, evidence_by_id: dict, reranker_config_id: str):
    """Constructs (never calls) the reranker provider from explicit env
    configuration. Object construction only -- no network."""

    provider_raw = os.environ.get("RERANKER_PROVIDER")
    if not provider_raw:
        raise ConfigError("A real reranking run requires RERANKER_PROVIDER to be set (currently only 'cohere').")
    if provider_raw != "cohere":
        raise ConfigError(f"Unknown RERANKER_PROVIDER {provider_raw!r}; only 'cohere' is implemented.")

    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise ConfigError("RERANKER_PROVIDER=cohere requires COHERE_API_KEY to be set.")

    from app.reranking.cohere_reranker import CohereReranker, build_cohere_client

    client = build_cohere_client(api_key)
    return CohereReranker(client, model_name, evidence_by_id, reranker_config_id=reranker_config_id)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to a frozen RerankerBaselineConfig JSON file (e.g. configs/reranker-baseline-v1.json).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    print("Agent Reliability Lab -- Reranking Evaluation")
    print("===============================================")
    print()

    try:
        config: RerankerBaselineConfig = load_reranker_baseline_config(args.config)
    except OSError as exc:
        print(f"ERROR: could not read config file {args.config}: {exc}", file=sys.stderr)
        return 1
    except ValidationError as exc:
        print(f"ERROR: {args.config} is not a valid RerankerBaselineConfig: {exc}", file=sys.stderr)
        return 1

    # The source artifact's path is fully reconstructed from the config's
    # own fields, via the SAME naming convention the retrieval script
    # used to write it -- never a second, independently-guessed path.
    source_artifact_path = retrieval_experiment_path(
        SOURCE_RESULTS_DIR, config.dataset_version, config.candidate_source_retriever_config_id,
        config.candidate_source_run_id,
    )

    try:
        retrieval_report = load_and_validate_candidate_source(config, source_artifact_path)
    except (ArtifactVerificationError, CandidateSourceValidationError) as exc:
        print(f"ERROR: candidate source validation failed: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"ERROR: could not read candidate source artifact {source_artifact_path}: {exc}", file=sys.stderr)
        return 1

    try:
        dataset = load_dataset(DATASET_DIR, version=config.dataset_version)
    except DatasetError as exc:
        print(f"ERROR: failed to load dataset: {exc}", file=sys.stderr)
        return 1
    corpus = build_full_version_corpus(dataset)
    evidence_by_id = {evidence.evidence_id: evidence for evidence in corpus.evidence}
    required_evidence_ids_by_case = {
        case.case_id: case.expected_outcome.required_evidence_ids for case in dataset.cases
    }

    try:
        reranker = _build_reranker(config.requested_model, evidence_by_id, config.config_id)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Captured once, before any provider call -- reuses the exact same
    # utility every other experiment script in this project uses.
    git_commit_sha = get_git_commit_sha(REPO_ROOT)
    if git_commit_sha is None:
        print(
            "ERROR: could not determine the current git commit SHA. A real reranking experiment "
            "must always be traceable back to the exact commit that produced it -- refusing to run "
            "rather than record a real result with no code provenance.",
            file=sys.stderr,
        )
        return 1

    run_id = resolve_run_id()
    output_path = reranking_experiment_path(RESULTS_DIR, config.dataset_version, config.config_id, run_id)
    if output_path.exists():
        print(
            f"ERROR: {output_path} already exists. A real reranking experiment artifact is immutable "
            "and can never be overwritten. This should only happen if ARL_RUN_ID was reused "
            "deliberately; choose a different run ID, or remove/rename the existing file yourself first.",
            file=sys.stderr,
        )
        return 1

    # --- Everything above this line is provenance/collision safety.
    #     Only NOW may the real reranker provider be invoked. ---
    print("Configuration:")
    print(f"  Reranker config:  {config.config_id}")
    print(f"  Candidate source: {config.candidate_source_run_id} ({config.candidate_source_artifact_sha256[:16]}...)")
    print(f"  Dataset:          {config.dataset_version}")
    print(f"  Candidate depth:  {config.candidate_depth}  Output depth: {config.reranker_output_depth}")
    print(f"  Git commit:       {git_commit_sha}")
    print()

    try:
        report: RerankingReport = run_reranking_evaluation(
            retrieval_report,
            reranker,
            candidate_depth=config.candidate_depth,
            output_depth=config.reranker_output_depth,
            k_values=tuple(config.evaluation_k_values),
            required_evidence_ids_by_case=required_evidence_ids_by_case,
            source_retrieval_artifact_sha256=config.candidate_source_artifact_sha256,
            reranker_config_id=config.config_id,
            reranker_provider=config.provider,
            requested_reranker_model=config.requested_model,
            served_reranker_model=getattr(reranker, "last_served_model_name", None),
            git_commit_sha=git_commit_sha,
            run_id=run_id,
        )
    except Exception as exc:  # an infrastructure crash, not a poor reranking score
        print(f"ERROR: reranking evaluation run failed: {exc}", file=sys.stderr)
        return 1

    print("Aggregate Recall@K (before -> after)")
    print("-------------------------------------")
    for k in report.evaluation_k_values:
        print(f"  Recall@{k}: {report.aggregate_recall_at_k_before[k]:.3f} -> {report.aggregate_recall_at_k_after[k]:.3f}")
    print(f"  MRR:      {report.aggregate_mrr_before:.3f} -> {report.aggregate_mrr_after:.3f}")
    print()

    try:
        write_reranking_report_atomically(output_path, report)
    except OSError as exc:
        print(f"ERROR: failed to write results: {exc}", file=sys.stderr)
        return 1

    print(f"Run ID: {report.run_id}")
    print("Results written to:")
    try:
        displayed_path = output_path.relative_to(REPO_ROOT)
    except ValueError:
        displayed_path = output_path
    print(f"  {displayed_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
