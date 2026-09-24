#!/usr/bin/env python3
"""Run a reranked-evidence LLM evaluation using a frozen reranked-LLM
baseline config and a preserved reranking artifact as the evidence source.

Usage:
    python scripts/run_reranked_llm_eval.py --config configs/reranked-llm-baseline-v1.json

No embedding calls, no Cohere calls, no re-retrieval, no re-reranking: the
evidence source is an already-preserved reranking artifact (see
app.integration.source).

Safety ordering, ALWAYS in this order, before making the first real LLM
provider call (``agent.run(...)``):

    1. loads the frozen reranked-LLM config
    2. locates the preserved reranking artifact (path reconstructed from
       the config's own dataset_version/source_reranking_config_id/
       source_reranking_run_id -- never a second, independently-guessed
       path)
    3. SHA-256 + provenance verifies that artifact against the config's
       recorded expectations (dataset/corpus fingerprint, reranker config
       ID, run ID, git provenance, exactly 30 results, every case has at
       least evidence_top_k candidates, evidence_top_k is one of the
       artifact's own evaluated K values) -- see
       app.integration.source.load_and_validate_reranking_source
    4. checks explicit real-API-call authorization (ARL_ALLOW_REAL_API_CALLS
       must be exactly "1" -- see app.observability.real_api_gate) BEFORE
       constructing the LLM agent. A real credential (OPENAI_API_KEY),
       however it was set -- exported directly or loaded from .env --
       never authorizes a real call by itself; this check is completely
       independent of credential presence.
    5. constructs the LLM agent (no network -- just __init__, mirroring
       every other provider adapter's lazy-no-network construction)
    6. resolves git_commit_sha (the SAME utility every other experiment
       script in this project uses)
    7. resolves run_id (a fresh UUID4, or ARL_RUN_ID for tests)
    8. computes the immutable, run-ID-qualified destination artifact path
       and checks it for a collision

Any failure at 1-8 aborts (exit 1) BEFORE step 9 (the real LLM calls) is
ever reached.

Exit codes:
    0  the run completed.
    1  the run could not complete: bad configuration, evidence-source
       validation failure, missing git provenance, a destination
       collision, or an infrastructure-level exception during evaluation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from app.datasets.loader import DatasetError, load_dataset
from app.integration.artifact_io import (
    reranked_llm_experiment_path,
    resolve_run_id,
    write_reranked_llm_report_atomically,
)
from app.integration.config import RerankedLLMBaselineConfig, load_reranked_llm_baseline_config
from app.integration.runner import RerankedLLMReport, run_reranked_llm_evaluation
from app.integration.source import (
    ArtifactVerificationError,
    RerankedEvidenceSourceValidationError,
    load_and_validate_reranking_source,
)
from app.models.contracts import LLMConfig, ModelProvider
from app.observability.git_provenance import get_git_commit_sha
from app.observability.real_api_gate import RealApiCallsNotAuthorizedError, require_real_api_authorization
from app.reranking.artifact_io import reranking_experiment_path
from app.retrieval.corpus import build_full_version_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "datasets"

# Where THIS run's own output is written -- mirrors every other
# experiment script's ARL_RESULTS_DIR convention.
RESULTS_DIR = Path(os.environ.get("ARL_RESULTS_DIR", str(REPO_ROOT / "results")))

# Where the evidence-source reranking artifact is READ from. Deliberately
# a SEPARATE override from ARL_RESULTS_DIR -- see
# scripts/run_reranking_eval.py's own SOURCE_RESULTS_DIR for the identical
# reasoning: a test must be able to point this at a fabricated reranking
# artifact without needing that fixture to also be this run's own output
# directory.
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


def _build_llm_agent(config: RerankedLLMBaselineConfig):
    """Constructs (never calls) the LLM agent from explicit env
    configuration, using this config's own frozen prompt_version/
    temperature/max_output_tokens. Object construction only -- no
    network."""

    try:
        provider_enum = ModelProvider(config.llm_provider)
    except ValueError:
        valid = ", ".join(p.value for p in ModelProvider)
        raise ConfigError(f"Unknown llm_provider {config.llm_provider!r} in config; expected one of: {valid}")

    if provider_enum is not ModelProvider.OPENAI:
        raise ConfigError(f"Provider {provider_enum.value!r} is not implemented yet. Only 'openai' is available.")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ConfigError("A real reranked-LLM run requires OPENAI_API_KEY to be set.")

    # Every configuration/credential-presence check above this line can
    # run freely -- none of them construct a real provider client. This
    # is the LAST check before that construction, independent of
    # everything above.
    require_real_api_authorization("OpenAI LLM agent construction (scripts/run_reranked_llm_eval.py)")

    from app.agent.llm_agent import LLMAgent
    from app.agent.pricing import load_pricing_config
    from app.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(
        api_key=api_key, model_name=config.requested_model,
        temperature=config.temperature, max_output_tokens=config.max_output_tokens,
    )
    pricing = load_pricing_config(config.requested_model)
    agent = LLMAgent(provider=provider, pricing=pricing)
    llm_config = LLMConfig(
        prompt_version=config.prompt_version, temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
    )
    return agent, provider_enum, config.requested_model, llm_config


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to a frozen RerankedLLMBaselineConfig JSON file (e.g. "
        "configs/reranked-llm-baseline-v1.json).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    print("Agent Reliability Lab -- Reranked-Evidence LLM Evaluation")
    print("===========================================================")
    print()

    try:
        config: RerankedLLMBaselineConfig = load_reranked_llm_baseline_config(args.config)
    except OSError as exc:
        print(f"ERROR: could not read config file {args.config}: {exc}", file=sys.stderr)
        return 1
    except ValidationError as exc:
        print(f"ERROR: {args.config} is not a valid RerankedLLMBaselineConfig: {exc}", file=sys.stderr)
        return 1

    # The source artifact's path is fully reconstructed from the config's
    # own fields, via the SAME naming convention the reranking script used
    # to write it -- never a second, independently-guessed path.
    source_artifact_path = reranking_experiment_path(
        SOURCE_RESULTS_DIR, config.dataset_version, config.source_reranking_config_id,
        config.source_reranking_run_id,
    )

    try:
        reranking_report = load_and_validate_reranking_source(config, source_artifact_path)
    except (ArtifactVerificationError, RerankedEvidenceSourceValidationError) as exc:
        print(f"ERROR: evidence source validation failed: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"ERROR: could not read evidence source artifact {source_artifact_path}: {exc}", file=sys.stderr)
        return 1

    try:
        dataset = load_dataset(DATASET_DIR, version=config.dataset_version)
    except DatasetError as exc:
        print(f"ERROR: failed to load dataset: {exc}", file=sys.stderr)
        return 1
    corpus = build_full_version_corpus(dataset)

    # Captured once, before any provider call -- reuses the exact same
    # utility every other experiment script in this project uses.
    git_commit_sha = get_git_commit_sha(REPO_ROOT)
    if git_commit_sha is None:
        print(
            "ERROR: could not determine the current git commit SHA. A real reranked-LLM experiment "
            "must always be traceable back to the exact commit that produced it -- refusing to run "
            "rather than record a real result with no code provenance.",
            file=sys.stderr,
        )
        return 1

    run_id = resolve_run_id()
    output_path = reranked_llm_experiment_path(RESULTS_DIR, config.dataset_version, config.config_id, run_id)
    if output_path.exists():
        print(
            f"ERROR: {output_path} already exists. A real reranked-LLM experiment artifact is "
            "immutable and can never be overwritten. This should only happen if ARL_RUN_ID was "
            "reused deliberately; choose a different run ID, or remove/rename the existing file "
            "yourself first.",
            file=sys.stderr,
        )
        return 1

    # --- Everything above this line is config/evidence-source/
    #     provenance/collision safety. Only NOW may the real LLM
    #     provider be constructed (and, moments later, invoked) --
    #     provider construction never precedes any check above,
    #     including the explicit real-API authorization gate inside
    #     _build_llm_agent itself. ---
    try:
        agent, provider_enum, model_name, llm_config = _build_llm_agent(config)
    except (ConfigError, RealApiCallsNotAuthorizedError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Configuration:")
    print(f"  Config:            {config.config_id}")
    print(f"  Evidence source:   {config.source_reranking_run_id} "
          f"({config.source_reranking_artifact_sha256[:16]}...)")
    print(f"  Dataset:           {config.dataset_version}")
    print(f"  Evidence top-K:    {config.evidence_top_k}")
    print(f"  Provider:          {provider_enum.value}  Model: {model_name}")
    print(f"  Prompt version:    {llm_config.prompt_version}")
    print(f"  Git commit:        {git_commit_sha}")
    print(f"  RUN_ID={run_id}")
    try:
        displayed_output_path = output_path.relative_to(REPO_ROOT)
    except ValueError:
        displayed_output_path = output_path
    print(f"  ARTIFACT_PATH={displayed_output_path}")
    print()

    try:
        report: RerankedLLMReport = run_reranked_llm_evaluation(
            reranking_report, dataset, corpus, agent,
            evidence_top_k=config.evidence_top_k,
            provider=provider_enum, model_name=model_name, llm_config=llm_config,
            source_reranking_artifact_sha256=config.source_reranking_artifact_sha256,
            source_reranking_config_id=config.source_reranking_config_id,
            git_commit_sha=git_commit_sha,
            run_id=run_id,
        )
    except Exception as exc:  # an infrastructure crash, not a wrong/malformed prediction
        print(f"ERROR: reranked-LLM evaluation run failed: {exc}", file=sys.stderr)
        print("FAILURE SUMMARY:", file=sys.stderr)
        print(f"  run_id: {run_id}", file=sys.stderr)
        print(f"  future_artifact_path: {displayed_output_path}", file=sys.stderr)
        print(f"  exception_category: {type(exc).__name__}", file=sys.stderr)
        print("  artifact_written: false", file=sys.stderr)
        return 1

    print("Metrics")
    print("-------")
    print(f"  Finding accuracy:       {report.metrics.finding_accuracy:.3f}")
    print(f"  Macro F1:               {report.metrics.macro_f1:.3f}")
    print(f"  Citation validity:      {report.metrics.citation_validity:.3f}")
    recall_display = (
        f"{report.metrics.required_evidence_recall:.3f}"
        if report.metrics.required_evidence_recall is not None else "N/A"
    )
    print(f"  Required-evidence recall: {recall_display}")
    print(f"  Schema validity:        {report.metrics.schema_validity:.3f}")
    print()

    try:
        write_reranked_llm_report_atomically(output_path, report)
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
