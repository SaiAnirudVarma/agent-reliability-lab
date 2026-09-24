"""Reranked-evidence LLM evaluation: for each case, take its preserved
Top-K reranked candidate set from a validated reranking artifact, build
an AgentInput from it (never from ``case.evidence_pool``), run the
existing LLM agent, and evaluate the result with the existing evaluator.
No embedding calls, no Cohere calls, no re-retrieval, no re-reranking --
those stages already happened and produced the artifact this consumes
(see ``app.integration.source``).

``evidence_top_k`` is decided ONCE, before any LLM exposure, and frozen
in ``configs/reranked-llm-baseline-v1.json`` -- see that file and
``app.integration.config`` for the reasoning (Recall@5 == Recall@10 on
the frozen reranking baseline, so Top-10 buys no required-evidence
recall over Top-5 while doubling evidence volume). This module never
changes, tunes, or re-derives that value; it only consumes whatever a
caller's config supplies.

This experiment's evidence source is deliberately distinct from, and
never a replacement for, the existing oracle LLM baseline
(``app.evaluation.runner.run_evaluation``, which uses
``case.evidence_pool`` via ``build_agent_input``): here, the agent sees
evidence selected by the retrieval + reranking pipeline instead. Both
remain independently runnable and comparable by ``case_id`` -- neither
overwrites or reinterprets the other.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.agent.interface import AgentRunner
from app.datasets.loader import Dataset
from app.evaluation.evaluator import evaluate_finding
from app.evaluation.metrics import compute_metric_summary
from app.integration.agent_input import build_agent_input_from_reranking
from app.integration.failure_attribution import FailureLayer, attribute_case_failure_layers
from app.integration.source import build_top_k_candidates_by_case
from app.models.contracts import AgentMode, EvaluationResult, ExecutionTrace, LLMConfig, MetricSummary, ModelProvider
from app.observability.tracing import run_traced
from app.reranking.runner import RerankingReport
from app.retrieval.corpus import EvidenceCorpus


class RerankedLLMCaseReport(BaseModel):
    """Per-case record letting a later analysis compare this experiment's
    outcome against retrieval-only and reranked evidence availability
    (comparison capabilities #2/#3) without recomputing anything -- both
    recall values are read directly from the already-preserved
    ``RerankingReport`` this run consumed.

    ``required_evidence_ids`` is ground truth -- safe here because this
    is an EVALUATOR-side report, produced only after the agent has
    already run (see ``run_reranked_llm_evaluation``); it is never fed
    back into any ``AgentInput``.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    presented_evidence_ids: list[str]
    required_evidence_ids: list[str]
    retrieval_recall_at_top_k: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reranked_recall_at_top_k: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    failure_layers: list[FailureLayer]


class RerankedLLMReport(BaseModel):
    """The full, persistable outcome of one reranked-evidence LLM
    evaluation run.

    ``evidence_source`` is always the literal ``"reranked"`` -- an
    explicit, structural marker (not just a docstring claim) that this
    report's agent input came from the retrieval+reranking pipeline, not
    from ``case.evidence_pool`` (the existing oracle LLM baseline) and not
    from unranked retrieval output alone. This is the "preserve that
    distinction explicitly in provenance" requirement.

    ``provider``/``model_name``/``llm_config`` are always present (never
    ``None``): unlike ``app.models.contracts.RunReport``, which also
    supports a mock/deterministic agent, this report type exists ONLY for
    the reranked-evidence LLM experiment.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    git_commit_sha: Optional[str] = None
    generated_at: datetime

    dataset_version: str = Field(min_length=1)
    dataset_fingerprint: str = Field(min_length=1)
    corpus_fingerprint: str = Field(min_length=1)

    source_reranking_run_id: str = Field(min_length=1)
    source_reranking_artifact_sha256: str = Field(min_length=1)
    source_reranking_config_id: str = Field(min_length=1)

    evidence_top_k: int = Field(ge=1)
    evidence_source: str = Field(default="reranked", pattern=r"^reranked$")

    agent_config_id: str = Field(min_length=1)
    provider: ModelProvider
    model_name: str = Field(min_length=1)
    served_model_name: Optional[str] = None
    llm_config: LLMConfig

    case_reports: list[RerankedLLMCaseReport]
    results: list[EvaluationResult]
    traces: list[ExecutionTrace]
    metrics: MetricSummary


def run_reranked_llm_evaluation(
    reranking_report: RerankingReport,
    dataset: Dataset,
    corpus: EvidenceCorpus,
    agent: AgentRunner,
    evidence_top_k: int,
    provider: ModelProvider,
    model_name: str,
    llm_config: LLMConfig,
    source_reranking_artifact_sha256: str,
    source_reranking_config_id: str,
    git_commit_sha: Optional[str] = None,
    run_id: Optional[str] = None,
) -> RerankedLLMReport:
    """Runs ``agent`` against every case in ``dataset``, using exactly
    that case's preserved Top-``evidence_top_k`` reranked candidates
    (hydrated to full ``Evidence`` via ``corpus``) as the presented
    evidence.

    Mirrors ``app.evaluation.runner.run_evaluation``'s dependency
    ordering exactly: ``build_agent_input_from_reranking`` (ground-truth
    -free) runs first, ``run_traced`` executes the agent against only
    that safe input, and only on the line after that does
    ``evaluate_finding`` receive ``case`` (and therefore
    ``case.expected_outcome``) at all.

    ``run_id`` defaults to a fresh UUID4 when not supplied; a real
    experiment caller supplies one resolved BEFORE this function runs
    (see ``scripts/run_reranked_llm_eval.py``), so a precomputed output
    path can be collision-checked before any provider call.
    """

    if run_id is None:
        run_id = str(uuid4())

    top_k_candidates_by_case = build_top_k_candidates_by_case(reranking_report, evidence_top_k)
    full_candidates_by_case = {result.case_id: result.candidates for result in reranking_report.rerank_results}
    source_case_report_by_id = {cr.case_id: cr for cr in reranking_report.case_reports}

    results: list[EvaluationResult] = []
    traces: list[ExecutionTrace] = []
    case_reports: list[RerankedLLMCaseReport] = []

    for case in dataset.cases:
        control = dataset.controls[case.control_id]
        top_k_candidates = top_k_candidates_by_case[case.case_id]

        # --- Ground truth is not yet visible past this point. ---
        task = build_agent_input_from_reranking(case, control, top_k_candidates, corpus)
        trace = run_traced(agent, task, run_id, AgentMode.LLM, provider, model_name, llm_config)
        # --- Ground truth becomes visible only now, via `case`, after the
        #     agent has already produced `trace.output`. ---
        result = evaluate_finding(case, task.evidence, trace.output, run_id)

        required_evidence_ids = case.expected_outcome.required_evidence_ids
        failure_layers = attribute_case_failure_layers(
            required_evidence_ids, full_candidates_by_case[case.case_id], evidence_top_k, result
        )

        source_case_report = source_case_report_by_id[case.case_id]
        case_reports.append(
            RerankedLLMCaseReport(
                case_id=case.case_id,
                presented_evidence_ids=[evidence.evidence_id for evidence in task.evidence],
                required_evidence_ids=required_evidence_ids,
                retrieval_recall_at_top_k=source_case_report.recall_at_k_before.get(evidence_top_k),
                reranked_recall_at_top_k=source_case_report.recall_at_k_after.get(evidence_top_k),
                failure_layers=failure_layers,
            )
        )
        results.append(result)
        traces.append(trace)

    metrics = compute_metric_summary(results, run_id)

    return RerankedLLMReport(
        run_id=run_id,
        git_commit_sha=git_commit_sha,
        generated_at=datetime.now(timezone.utc),
        dataset_version=dataset.version,
        dataset_fingerprint=dataset.fingerprint,
        corpus_fingerprint=corpus.fingerprint,
        source_reranking_run_id=reranking_report.run_id,
        source_reranking_artifact_sha256=source_reranking_artifact_sha256,
        source_reranking_config_id=source_reranking_config_id,
        evidence_top_k=evidence_top_k,
        agent_config_id=agent.agent_config_id,
        provider=provider,
        model_name=model_name,
        served_model_name=getattr(agent, "last_served_model_name", None),
        llm_config=llm_config,
        case_reports=case_reports,
        results=results,
        traces=traces,
        metrics=metrics,
    )
