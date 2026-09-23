"""Orchestrates one complete evaluation run: dataset -> traced agent
execution -> evaluation -> aggregate metrics -> RunReport.

The dependency direction this project insists on — the agent may never see
ground truth, and the evaluator only sees a finding after it has already
been produced — is enforced here by the literal order of statements in the
loop body, not merely by a comment: ``build_agent_input`` (ground-truth-free)
runs first, then ``run_traced`` executes the agent against only that safe
input, and only on the line after that does ``evaluate_finding`` receive
``case`` (and therefore ``case.expected_outcome``) at all.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.agent.interface import AgentRunner, build_agent_input
from app.datasets.loader import Dataset
from app.evaluation.evaluator import evaluate_finding
from app.evaluation.metrics import compute_metric_summary
from app.models.contracts import AgentMode, LLMConfig, ModelProvider, RunReport
from app.observability.tracing import run_traced


def run_evaluation(
    dataset: Dataset,
    agent: AgentRunner,
    agent_mode: AgentMode,
    provider: ModelProvider | None = None,
    model_name: str | None = None,
    llm_config: LLMConfig | None = None,
    run_id: str | None = None,
    git_commit_sha: str | None = None,
) -> RunReport:
    """Run ``agent`` against every case in ``dataset`` and return a RunReport.

    Cases are executed in ``dataset.cases`` order, which is the order the
    dataset's JSON file was written in (the loader does not sort or
    reorder), so results/traces are always in stable, reproducible order.

    ``provider``/``model_name``/``llm_config``/``git_commit_sha`` are
    declared by the caller (e.g. the CLI, reading explicit configuration or
    capturing git state once) for the whole run and threaded into the report
    -- never inferred from ``agent`` itself, and never re-derived per case.

    ``run_id`` defaults to a fresh UUID4 when not supplied, preserving this
    function's original self-contained behavior for direct callers (e.g.
    tests, or deterministic-mode use). A caller that needs to know the run
    ID *before* the run completes -- e.g. to precompute a run-ID-qualified
    output path and check for a collision before spending a real API call --
    supplies one explicitly, and the returned ``RunReport.run_id`` is
    guaranteed to be exactly that value (see ``scripts/run_eval.py``).
    """

    if run_id is None:
        run_id = str(uuid4())
    results = []
    traces = []

    for case in dataset.cases:
        control = dataset.controls[case.control_id]
        evidence = dataset.evidence_for_case(case)

        # --- Ground truth is not yet visible past this point. ---
        task = build_agent_input(case, control, evidence)
        trace = run_traced(agent, task, run_id, agent_mode, provider, model_name, llm_config)
        # --- Ground truth becomes visible only now, via `case`, after the
        #     agent has already produced `trace.output`. ---
        result = evaluate_finding(case, evidence, trace.output, run_id)

        traces.append(trace)
        results.append(result)

    metrics = compute_metric_summary(results, run_id)

    return RunReport(
        run_id=run_id,
        agent_mode=agent_mode,
        provider=provider,
        model_name=model_name,
        llm_config=llm_config,
        dataset_version=dataset.version,
        dataset_fingerprint=dataset.fingerprint,
        git_commit_sha=git_commit_sha,
        results=results,
        traces=traces,
        metrics=metrics,
        generated_at=datetime.now(timezone.utc),
    )
