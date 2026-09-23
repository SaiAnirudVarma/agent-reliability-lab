#!/usr/bin/env python3
"""Run the Phase 1 evaluation suite.

Usage:
    python scripts/run_eval.py                  # DeterministicBaselineAgent -> results/baseline.json
    python scripts/run_eval.py --agent llm       # LLMAgent -> results/llm-baseline.json
    python scripts/run_eval.py --compare         # print a comparison of the two result files

The default (no arguments) always runs DeterministicBaselineAgent -- LLM
mode is never entered implicitly, regardless of which API keys happen to be
set in the environment.

Exit codes:
    0  the run completed. This is returned even if the agent got every case
       wrong, or every case came back schema-invalid -- an incorrect or
       malformed prediction is an experiment result, not an infrastructure
       failure.
    1  the run could not complete: bad configuration (missing/invalid
       AGENT_MODE, MODEL_PROVIDER, MODEL_NAME, or API key), the dataset
       failed to load/validate, an infrastructure-level exception during
       execution (auth/network/etc.), or the results file could not be
       written.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
from uuid import uuid4

from app.agent.deterministic_baseline import DeterministicBaselineAgent
from app.agent.interface import AgentRunner
from app.agent.llm_agent import MAX_OUTPUT_TOKENS, TEMPERATURE, build_llm_config
from app.agent.pricing import load_pricing_config
from app.datasets.loader import DatasetError, load_dataset
from app.evaluation.metrics import percentile
from app.evaluation.runner import run_evaluation
from app.models.contracts import AgentMode, LLMConfig, ModelProvider, RunReport
from app.observability.git_provenance import get_git_commit_sha

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "datasets"

# The directory results are read from and written to. Overridable via
# ARL_RESULTS_DIR (mirroring ARL_DOTENV_PATH below) so tests can redirect
# every result read/write into a pytest tmp_path and never touch this
# repository's real results/ directory. See
# docs/incidents/2026-09-23-llm-baseline-artifact-deletion.md for why this
# matters: a test that manipulated the hardcoded production path here is
# what destroyed the original results/llm-baseline.json.
RESULTS_DIR = Path(os.environ.get("ARL_RESULTS_DIR", str(REPO_ROOT / "results")))
BASELINE_RESULTS_PATH = RESULTS_DIR / "baseline.json"
LLM_RESULTS_PATH = RESULTS_DIR / "llm-baseline.json"

# Every real LLM run (any dataset version, from this point forward) is
# written here instead of to a fixed filename -- see
# docs/ARTIFACT_POLICY.md's "Future real-experiment naming" section, which
# this implements. LLM_RESULTS_PATH above is no longer written to by any
# code path; it stays defined only because it is still READ by --compare
# and still names the one frozen, historic synthetic-v1 LLM baseline that
# predates this scheme and must never be touched.
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"

# The path .env is loaded from. Overridable via ARL_DOTENV_PATH so tests can
# point it at a nonexistent path and get fully deterministic, real-.env-
# independent behavior without touching this repo's actual .env file.
# Defaults to <repo_root>/.env -- an explicit, predictable location rather
# than relying on python-dotenv's own directory-walking auto-discovery.
_DOTENV_PATH = Path(os.environ.get("ARL_DOTENV_PATH", str(REPO_ROOT / ".env")))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a core dependency; defensive only
    load_dotenv = None

if load_dotenv is not None:
    # override=False (the default, made explicit here): a value already
    # present in the real shell environment always wins over .env -- .env
    # only fills in what isn't already exported. Silently does nothing if
    # _DOTENV_PATH doesn't exist (never raises).
    load_dotenv(dotenv_path=_DOTENV_PATH, override=False)


class ConfigError(Exception):
    """A CLI-level configuration problem (missing env var, bad provider
    name, missing API key) -- distinct from a dataset error or an
    in-run infrastructure failure, but handled the same way: print and
    exit 1 before anything is run."""


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--agent",
        choices=["deterministic", "llm"],
        default="deterministic",
        help="Which agent to run. Default: deterministic (never changes based on env vars alone).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print a comparison of results/baseline.json vs results/llm-baseline.json and exit "
        "(does not run any agent).",
    )
    parser.add_argument(
        "--dataset-version",
        choices=["synthetic-v1", "synthetic-v2"],
        default="synthetic-v1",
        help="Which named dataset version to run. Default: synthetic-v1 (the original 10-case "
        "benchmark) -- never changes unless explicitly requested, so existing scripts/results stay "
        "reproducible.",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="No longer has any effect on --agent llm (kept only for backward CLI compatibility): "
        "every real LLM run now writes a unique, run-ID-qualified file under "
        "results/experiments/, so it can never collide with a prior real result, and that "
        "immutable-artifact protection cannot be bypassed by any flag -- see "
        "docs/ARTIFACT_POLICY.md. Deterministic-mode output is always freely overwritten (it is "
        "reproducible on demand), so this flag has no effect there either.",
    )
    return parser.parse_args(argv)


def _build_llm_agent() -> tuple[AgentRunner, ModelProvider, str, LLMConfig]:
    """Construct LLMAgent from explicit env configuration. Never infers a
    provider merely because an API key happens to exist -- MODEL_PROVIDER
    and MODEL_NAME must both be set.

    The returned LLMConfig comes from build_llm_config() (the single source
    of truth for the current, untouched prompt/temperature/max-tokens
    configuration) -- it is not independently re-typed here.
    """

    provider_raw = os.environ.get("MODEL_PROVIDER")
    model_name = os.environ.get("MODEL_NAME")
    if not provider_raw or not model_name:
        raise ConfigError(
            "AGENT_MODE=llm requires both MODEL_PROVIDER and MODEL_NAME to be set explicitly "
            "(e.g. MODEL_PROVIDER=openai MODEL_NAME=gpt-4o-2024-08-06). See .env.example."
        )
    try:
        provider_enum = ModelProvider(provider_raw)
    except ValueError:
        valid = ", ".join(p.value for p in ModelProvider)
        raise ConfigError(f"Unknown MODEL_PROVIDER {provider_raw!r}; expected one of: {valid}")

    if provider_enum is not ModelProvider.OPENAI:
        raise ConfigError(
            f"Provider {provider_enum.value!r} is not implemented yet. Only 'openai' is available."
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ConfigError("MODEL_PROVIDER=openai requires OPENAI_API_KEY to be set.")

    from app.agent.llm_agent import LLMAgent
    from app.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(
        api_key=api_key, model_name=model_name, temperature=TEMPERATURE, max_output_tokens=MAX_OUTPUT_TOKENS
    )
    pricing = load_pricing_config(model_name)
    agent = LLMAgent(provider=provider, pricing=pricing)
    return agent, provider_enum, model_name, build_llm_config()


def _print_case_results(report: RunReport) -> None:
    # "PASS"/"FAIL" here would be ambiguous with the audit finding class
    # named PASS, so results are shown as expected-vs-actual with a
    # match/mismatch mark instead.
    for result in report.results:
        expected = result.expected.expected_finding.value
        if result.actual is not None:
            actual = result.actual.finding.value
        else:
            actual = "SCHEMA_INVALID"
        mark = "✓" if result.correct_finding else "✗"
        print(f"{result.case_id}  expected={expected:<22} actual={actual:<22} {mark}")


def _print_metrics(report: RunReport) -> None:
    m = report.metrics
    print()
    print("Metrics")
    print("-------")
    print(f"Finding accuracy:       {m.finding_accuracy:.3f}")
    print(f"Macro F1:               {m.macro_f1:.3f}")
    print(f"Exception precision:    {m.exception_precision:.3f}")
    print(f"Exception recall:       {m.exception_recall:.3f}")
    print(f"Citation validity:      {m.citation_validity:.3f}")
    recall_display = f"{m.required_evidence_recall:.3f}" if m.required_evidence_recall is not None else "N/A"
    print(f"Required-evidence recall: {recall_display}")
    print(f"Schema validity:        {m.schema_validity:.3f}")
    print(f"Abstention precision:   {m.abstention_precision:.3f}")
    print(f"Abstention recall:      {m.abstention_recall:.3f}")
    print(f"Abstention F1:          {m.abstention_f1:.3f}")
    print()


def _results_path(agent_choice: str, dataset_version: str) -> Path:
    """The default output path for one (agent, dataset_version) combination.

    synthetic-v1 keeps the exact original paths (results/baseline.json,
    results/llm-baseline.json) -- including the official, frozen
    synthetic-v1 LLM baseline, which this function never points at for any
    other combination. Any other dataset version gets its own
    version-qualified filename, so it can never collide with or overwrite a
    v1 artifact.
    """

    if dataset_version == "synthetic-v1":
        return BASELINE_RESULTS_PATH if agent_choice == "deterministic" else LLM_RESULTS_PATH
    version_suffix = dataset_version.removeprefix("synthetic-")
    prefix = "deterministic" if agent_choice == "deterministic" else "llm"
    return RESULTS_DIR / f"{prefix}-{version_suffix}.json"


def _experiment_results_path(dataset_version: str, agent_config_id: str, run_id: str) -> Path:
    """The immutable, collision-proof destination for one real LLM run.

    Every component is stable, known-before-the-run metadata except
    ``run_id`` -- a fresh UUID4 (or an explicit ``ARL_RUN_ID`` override, see
    ``main``) generated by the CLI *before* calling ``run_evaluation``, so
    this exact path can be computed and checked for a collision before any
    API call is made, not only after. See docs/ARTIFACT_POLICY.md's "Future
    real-experiment naming" section, which this implements.
    """

    return EXPERIMENTS_DIR / f"{dataset_version}__{agent_config_id}__{run_id}.json"


def _mean_or_na(values: list[Optional[float]]) -> str:
    if not values or any(v is None for v in values):
        return "N/A"
    return f"{sum(values) / len(values):.2f}"


def _print_comparison() -> int:
    if not BASELINE_RESULTS_PATH.exists() or not LLM_RESULTS_PATH.exists():
        print(
            "ERROR: both results/baseline.json and results/llm-baseline.json must exist to "
            "compare. Run both `python scripts/run_eval.py` and "
            "`python scripts/run_eval.py --agent llm` first.",
            file=sys.stderr,
        )
        return 1

    baseline = RunReport.model_validate_json(BASELINE_RESULTS_PATH.read_text())
    llm = RunReport.model_validate_json(LLM_RESULTS_PATH.read_text())
    bm, lm = baseline.metrics, llm.metrics

    def row(label: str, det_val: str, llm_val: str) -> None:
        print(f"{label:<26} {det_val:<16} {llm_val}")

    print(f"{'Metric':<26} {'Deterministic':<16} LLM Baseline")
    print("-" * 62)
    row("Finding accuracy", f"{bm.finding_accuracy:.3f}", f"{lm.finding_accuracy:.3f}")
    row("Macro F1", f"{bm.macro_f1:.3f}", f"{lm.macro_f1:.3f}")
    row("Exception precision", f"{bm.exception_precision:.3f}", f"{lm.exception_precision:.3f}")
    row("Exception recall", f"{bm.exception_recall:.3f}", f"{lm.exception_recall:.3f}")
    row("Citation validity", f"{bm.citation_validity:.3f}", f"{lm.citation_validity:.3f}")
    recall_row = lambda m: f"{m.required_evidence_recall:.3f}" if m.required_evidence_recall is not None else "N/A"
    row("Required-evidence recall", recall_row(bm), recall_row(lm))
    row("Schema validity", f"{bm.schema_validity:.3f}", f"{lm.schema_validity:.3f}")
    row("Abstention precision", f"{bm.abstention_precision:.3f}", f"{lm.abstention_precision:.3f}")
    row("Abstention recall", f"{bm.abstention_recall:.3f}", f"{lm.abstention_recall:.3f}")
    row("Abstention F1", f"{bm.abstention_f1:.3f}", f"{lm.abstention_f1:.3f}")

    det_latencies = [t.latency_ms for t in baseline.traces]
    llm_latencies = [t.latency_ms for t in llm.traces]
    row("P50 latency (ms)", f"{percentile(det_latencies, 0.5):.3f}", f"{percentile(llm_latencies, 0.5):.1f}")
    row("P95 latency (ms)", f"{percentile(det_latencies, 0.95):.3f}", f"{percentile(llm_latencies, 0.95):.1f}")

    row("Input tokens/run", "N/A", _mean_or_na([t.input_tokens for t in llm.traces]))
    row("Output tokens/run", "N/A", _mean_or_na([t.output_tokens for t in llm.traces]))
    row("Estimated cost/run", "N/A", _mean_or_na([t.estimated_cost for t in llm.traces]))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    if args.compare:
        return _print_comparison()

    print("Agent Reliability Lab")
    print("=====================")
    print()

    try:
        dataset = load_dataset(DATASET_DIR, version=args.dataset_version)
    except DatasetError as exc:
        print(f"ERROR: failed to load dataset: {exc}", file=sys.stderr)
        return 1

    try:
        if args.agent == "deterministic":
            agent: AgentRunner = DeterministicBaselineAgent()
            agent_mode, provider_enum, model_name, llm_config = AgentMode.MOCK, None, None, None
        else:
            agent, provider_enum, model_name, llm_config = _build_llm_agent()
            agent_mode = AgentMode.LLM
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Captured exactly once per run, regardless of agent mode -- never
    # shelled out to per case. ARL_GIT_COMMIT_SHA lets tests inject a fixed
    # value without depending on this machine's real git state.
    git_commit_sha = get_git_commit_sha(REPO_ROOT)

    if agent_mode == AgentMode.LLM:
        if git_commit_sha is None:
            print(
                "ERROR: could not determine the current git commit SHA (not a git repository, "
                "git is not installed, or `git rev-parse HEAD` failed). A real LLM experiment "
                "must always be traceable back to the exact commit that produced it -- refusing "
                "to run rather than record a real result with no code provenance. See "
                "docs/ARTIFACT_POLICY.md.",
                file=sys.stderr,
            )
            return 1

        # run_id is decided BEFORE run_evaluation() is ever called (an
        # explicit ARL_RUN_ID override for tests, otherwise a fresh UUID4),
        # so the immutable, run-ID-qualified destination path is fully known
        # -- and can be checked for a collision -- before any real API call
        # is made. This check is NOT overridable by --force-overwrite: a
        # run-ID-qualified path is unique by construction, and if it somehow
        # already exists anyway, that is exactly the "immutable experiment
        # artifact" case ARTIFACT_POLICY.md Category A says must never be
        # overwritten, by any flag, under any circumstance.
        run_id = os.environ.get("ARL_RUN_ID") or str(uuid4())
        results_path = _experiment_results_path(dataset.version, agent.agent_config_id, run_id)
        if results_path.exists():
            print(
                f"ERROR: {results_path} already exists. A real LLM experiment artifact is "
                "immutable and can never be overwritten -- not even with --force-overwrite. This "
                "should only happen if ARL_RUN_ID was reused deliberately; choose a different run "
                "ID, or remove/rename the existing file yourself first.",
                file=sys.stderr,
            )
            return 1
    else:
        run_id = None
        results_path = _results_path(args.agent, args.dataset_version)

    print("Configuration:")
    print(f"  Agent:    {agent.agent_config_id}")
    print(f"  Mode:     {agent_mode.value}")
    if provider_enum is not None:
        print(f"  Provider: {provider_enum.value}")
        print(f"  Model:    {model_name}")
    if llm_config is not None:
        print(f"  Prompt version:     {llm_config.prompt_version}")
        print(f"  Temperature:        {llm_config.temperature}")
        print(f"  Max output tokens:  {llm_config.max_output_tokens}")
    print(f"  Dataset:  {dataset.version} ({len(dataset.cases)} cases)")
    print(f"  Dataset fingerprint: {dataset.fingerprint[:16]}...")
    print(f"  Git commit: {git_commit_sha if git_commit_sha is not None else 'unknown'}")
    print()
    print(f"Running {len(dataset.cases)} cases...")
    print()

    try:
        report = run_evaluation(
            dataset, agent, agent_mode=agent_mode, provider=provider_enum,
            model_name=model_name, llm_config=llm_config, run_id=run_id,
            git_commit_sha=git_commit_sha,
        )
    except Exception as exc:  # an infrastructure crash (auth/network/bug), not a wrong/malformed prediction
        print(f"ERROR: evaluation run failed: {exc}", file=sys.stderr)
        return 1

    _print_case_results(report)
    _print_metrics(report)

    try:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(report.model_dump_json(indent=2) + "\n")
    except OSError as exc:
        print(f"ERROR: failed to write results: {exc}", file=sys.stderr)
        return 1

    print(f"Run ID: {report.run_id}")
    print("Results written to:")
    try:
        displayed_path = results_path.relative_to(REPO_ROOT)
    except ValueError:
        # results_path is outside REPO_ROOT -- only happens when
        # ARL_RESULTS_DIR redirects output elsewhere (e.g. a test's
        # tmp_path). Show the absolute path in that case.
        displayed_path = results_path
    print(f"  {displayed_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
