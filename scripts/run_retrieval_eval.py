#!/usr/bin/env python3
"""Run a retrieval-only evaluation against a frozen dataset version.

Usage:
    python scripts/run_retrieval_eval.py                          # LexicalBaselineRetriever, synthetic-v1
    python scripts/run_retrieval_eval.py --retriever vector --dataset-version synthetic-v2
    python scripts/run_retrieval_eval.py --config configs/retrieval-baseline-v1.json   # official, frozen run

No AgentRunner, no LLM, is ever invoked from this script -- this measures
retrieval quality in isolation (see docs/RETRIEVAL_DESIGN.md).

``--config`` (a frozen ``RetrievalBaselineConfig`` JSON file, see
``app.retrieval.baseline_config``) is how an official experiment run uses a
reviewed, tracked, non-secret parameter set instead of reconstructing
dataset version / retriever kind / embedding model / k values independently
from ``--dataset-version``/``--retriever``/environment each time. It NEVER
carries credentials -- those still come only from the environment.

Safety ordering for a real-embedding-provider run (``--retriever vector``
or, equivalently, a ``--config`` whose ``retrieval_method`` is ``"vector"``):
this script ALWAYS, in this order, before making its first real provider
call (``build_corpus_embedding_index``'s ``embed_documents``):

    1. loads the frozen config, if ``--config`` was given (a local file
       read only -- no network, no dataset access yet)
    2. loads the dataset/version/fingerprint and builds the corpus
    3. if a config was given: verifies the ACTUAL current
       ``dataset.fingerprint``/``corpus.fingerprint`` against the config's
       recorded values -- a mismatch means the frozen config has drifted
       from the current code/data and the run refuses to proceed
    4. constructs the embedding provider object (no network -- this is
       just __init__, mirroring OpenAIProvider's own lazy-no-network
       construction) and, if a config was given, verifies its
       ``model_name``/``embedding_dimension`` against the config's
       recorded values
    5. resolves git_commit_sha (app.observability.git_provenance --
       the SAME utility scripts/run_eval.py uses)
    6. resolves run_id (a fresh UUID4, or ARL_RUN_ID for tests)
    7. computes the immutable, run-ID-qualified destination artifact path
    8. checks that path for a collision

Any failure at 1, 3-8 aborts (exit 1) BEFORE step 9 (the real embedding
call) is ever reached -- see
docs/incidents/2026-09-23-llm-baseline-artifact-deletion.md for why this
project treats "provenance/collision checks before any real provider
call" as non-negotiable.

Exit codes:
    0  the run completed (even a poor retrieval score is a result, not an
       infrastructure failure).
    1  the run could not complete: bad configuration, dataset load
       failure, missing git provenance for a real run, a destination
       collision, or an infrastructure-level exception during retrieval.
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
from app.retrieval.artifact_io import (
    resolve_run_id,
    retrieval_experiment_path,
    write_retrieval_report_atomically,
)
from app.retrieval.baseline_config import RetrievalBaselineConfig, load_retrieval_baseline_config
from app.retrieval.corpus import build_full_version_corpus
from app.retrieval.corpus_index import build_corpus_embedding_index
from app.retrieval.interface import Retriever
from app.retrieval.lexical_baseline import RETRIEVER_CONFIG_ID as LEXICAL_RETRIEVER_CONFIG_ID
from app.retrieval.lexical_baseline import LexicalBaselineRetriever
from app.retrieval.runner import DEFAULT_K_VALUES, RetrievalReport, run_retrieval_evaluation
from app.retrieval.vector_retriever import VectorRetriever

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "datasets"

# Mirrors scripts/run_eval.py's RESULTS_DIR/ARL_RESULTS_DIR convention
# exactly, so tests can redirect every read/write here into a pytest
# tmp_path the same way, and never touch this repository's real results/.
RESULTS_DIR = Path(os.environ.get("ARL_RESULTS_DIR", str(REPO_ROOT / "results")))

_DOTENV_PATH = Path(os.environ.get("ARL_DOTENV_PATH", str(REPO_ROOT / ".env")))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a core dependency; defensive only
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(dotenv_path=_DOTENV_PATH, override=False)


class ConfigError(Exception):
    """A CLI-level configuration problem -- handled like scripts/run_eval.py's
    own ConfigError: print and exit 1 before anything expensive runs."""


def _build_vector_embedding_provider():
    """Constructs (never calls) an OpenAIEmbeddingProvider from explicit
    env configuration. Object construction only -- no network -- mirroring
    OpenAIProvider's own lazy-no-network-at-construction design.
    """

    provider_raw = os.environ.get("EMBEDDING_PROVIDER")
    model_name = os.environ.get("EMBEDDING_MODEL_NAME")
    dimension_raw = os.environ.get("EMBEDDING_DIMENSION")
    if not provider_raw or not model_name or not dimension_raw:
        raise ConfigError(
            "--retriever vector requires EMBEDDING_PROVIDER, EMBEDDING_MODEL_NAME, and "
            "EMBEDDING_DIMENSION to all be set explicitly."
        )
    if provider_raw != "openai":
        raise ConfigError(f"Unknown EMBEDDING_PROVIDER {provider_raw!r}; only 'openai' is implemented.")
    try:
        embedding_dimension = int(dimension_raw)
    except ValueError:
        raise ConfigError(f"EMBEDDING_DIMENSION must be an integer, got {dimension_raw!r}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ConfigError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY to be set.")

    from app.retrieval.openai_embedding_provider import OpenAIEmbeddingProvider

    return OpenAIEmbeddingProvider(api_key=api_key, model_name=model_name, embedding_dimension=embedding_dimension)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--retriever", choices=["lexical", "vector"], default="lexical",
        help="Which retriever to run. Default: lexical (LexicalBaselineRetriever, no embedding provider, "
        "never touches the network). 'vector' constructs a real OpenAIEmbeddingProvider from env "
        "configuration -- see EMBEDDING_PROVIDER/EMBEDDING_MODEL_NAME/EMBEDDING_DIMENSION/OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--dataset-version", choices=["synthetic-v1", "synthetic-v2"], default="synthetic-v1",
        help="Which named dataset version to run. Default: synthetic-v1. Ignored if --config is given "
        "(the config's own dataset_version is authoritative).",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to a frozen RetrievalBaselineConfig JSON file (see app.retrieval.baseline_config, "
        "e.g. configs/retrieval-baseline-v1.json). When given, --dataset-version and --retriever are "
        "ignored -- dataset version, retriever kind, and k_values all come from the config instead, "
        "and the config's recorded dataset_fingerprint/corpus_fingerprint are verified against the "
        "actual current dataset/corpus before any provider call. Never carries credentials.",
    )
    # Deliberately NO --force-overwrite: an immutable retrieval experiment
    # artifact can never be overwritten by any flag -- see
    # docs/ARTIFACT_POLICY.md's Category A discipline, applied identically
    # here.
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    print("Agent Reliability Lab -- Retrieval Evaluation")
    print("==============================================")
    print()

    config: Optional[RetrievalBaselineConfig] = None
    if args.config is not None:
        try:
            config = load_retrieval_baseline_config(args.config)
        except OSError as exc:
            print(f"ERROR: could not read config file {args.config}: {exc}", file=sys.stderr)
            return 1
        except ValidationError as exc:
            print(f"ERROR: {args.config} is not a valid RetrievalBaselineConfig: {exc}", file=sys.stderr)
            return 1

    dataset_version = config.dataset_version if config is not None else args.dataset_version
    retriever_kind = config.retrieval_method if config is not None else args.retriever
    k_values = tuple(config.k_values) if config is not None else DEFAULT_K_VALUES

    try:
        dataset = load_dataset(DATASET_DIR, version=dataset_version)
    except DatasetError as exc:
        print(f"ERROR: failed to load dataset: {exc}", file=sys.stderr)
        return 1
    corpus = build_full_version_corpus(dataset)

    if config is not None:
        if dataset.fingerprint != config.dataset_fingerprint:
            print(
                f"ERROR: current dataset_fingerprint {dataset.fingerprint!r} does not match "
                f"{args.config}'s frozen dataset_fingerprint {config.dataset_fingerprint!r}. The "
                "frozen config has drifted from the current dataset -- refusing to run rather than "
                "record a result against the wrong benchmark content.",
                file=sys.stderr,
            )
            return 1
        if corpus.fingerprint != config.corpus_fingerprint:
            print(
                f"ERROR: current corpus_fingerprint {corpus.fingerprint!r} does not match "
                f"{args.config}'s frozen corpus_fingerprint {config.corpus_fingerprint!r}. The frozen "
                "config has drifted from the current corpus -- refusing to run rather than record a "
                "result against the wrong evidence content.",
                file=sys.stderr,
            )
            return 1

    embedding_provider = None
    if retriever_kind == "vector":
        try:
            embedding_provider = _build_vector_embedding_provider()
        except ConfigError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if config is not None:
            if embedding_provider.model_name != config.embedding_model:
                print(
                    f"ERROR: configured EMBEDDING_MODEL_NAME {embedding_provider.model_name!r} does not "
                    f"match {args.config}'s frozen embedding_model {config.embedding_model!r}.",
                    file=sys.stderr,
                )
                return 1
            if embedding_provider.embedding_dimension != config.embedding_dimension:
                print(
                    f"ERROR: configured EMBEDDING_DIMENSION {embedding_provider.embedding_dimension} does "
                    f"not match {args.config}'s frozen embedding_dimension {config.embedding_dimension}.",
                    file=sys.stderr,
                )
                return 1

    # Captured once, before any provider call -- reuses the exact same
    # utility scripts/run_eval.py uses, never a competing implementation.
    git_commit_sha = get_git_commit_sha(REPO_ROOT)
    if retriever_kind == "vector" and git_commit_sha is None:
        print(
            "ERROR: could not determine the current git commit SHA. A real retrieval experiment "
            "must always be traceable back to the exact commit that produced it -- refusing to run "
            "rather than record a real result with no code provenance.",
            file=sys.stderr,
        )
        return 1

    run_id = resolve_run_id()
    if config is not None:
        retriever_config_id = config.config_id
    else:
        retriever_config_id = (
            LEXICAL_RETRIEVER_CONFIG_ID if retriever_kind == "lexical" else f"vector-{embedding_provider.model_name}"
        )
    results_path = retrieval_experiment_path(RESULTS_DIR, dataset.version, retriever_config_id, run_id)

    if results_path.exists():
        print(
            f"ERROR: {results_path} already exists. A real retrieval experiment artifact is immutable "
            "and can never be overwritten. This should only happen if ARL_RUN_ID was reused "
            "deliberately; choose a different run ID, or remove/rename the existing file yourself first.",
            file=sys.stderr,
        )
        return 1

    # --- Everything above this line is provenance/collision safety.
    #     Only NOW may a real provider be invoked. ---
    embedding_index = None
    corpus_embedding_latency_ms = None
    if retriever_kind == "vector":
        started = time.perf_counter()
        embedding_index = build_corpus_embedding_index(corpus, embedding_provider)
        corpus_embedding_latency_ms = (time.perf_counter() - started) * 1000.0
        retriever: Retriever = VectorRetriever(embedding_provider, embedding_index, retriever_config_id=retriever_config_id)
    else:
        retriever = LexicalBaselineRetriever()

    print("Configuration:")
    if config is not None:
        print(f"  Config:     {config.config_id} ({args.config})")
    print(f"  Retriever:  {retriever_config_id}")
    print(f"  Dataset:    {dataset.version} ({len(dataset.cases)} cases)")
    print(f"  Corpus:     {len(corpus)} evidence records")
    print(f"  Git commit: {git_commit_sha if git_commit_sha is not None else 'unknown'}")
    print()

    try:
        report = run_retrieval_evaluation(
            dataset, retriever, corpus, k_values=k_values, embedding_index=embedding_index,
            git_commit_sha=git_commit_sha, run_id=run_id, corpus_embedding_latency_ms=corpus_embedding_latency_ms,
        )
    except Exception as exc:  # an infrastructure crash, not a poor retrieval score
        print(f"ERROR: retrieval evaluation run failed: {exc}", file=sys.stderr)
        return 1

    _print_report_summary(report)

    try:
        write_retrieval_report_atomically(results_path, report)
    except OSError as exc:
        print(f"ERROR: failed to write results: {exc}", file=sys.stderr)
        return 1

    print(f"Run ID: {report.run_id}")
    print("Results written to:")
    try:
        displayed_path = results_path.relative_to(REPO_ROOT)
    except ValueError:
        displayed_path = results_path
    print(f"  {displayed_path}")

    return 0


def _print_report_summary(report: RetrievalReport) -> None:
    print("Aggregate Recall@K")
    print("-------------------")
    for k in report.top_k_values:
        print(f"  Recall@{k}: {report.aggregate_recall_at_k[k]:.3f}")
    print(f"  MRR:      {report.aggregate_mrr:.3f}")
    print()


if __name__ == "__main__":
    sys.exit(main())
