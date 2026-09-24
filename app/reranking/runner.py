"""Reranking-only evaluation: for each case, take its preserved,
already-generated candidate set from a validated retrieval artifact,
rerank it, and score BEFORE (original retrieval order) vs AFTER
(reranked order). No embedding calls, no LLM calls, no re-retrieval --
the retrieval stage already happened and produced the artifact this
consumes (see app.reranking.candidate_source).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

import app.retrieval.metrics as retrieval_metrics
from app.reranking.candidate_source import build_candidate_sets
from app.reranking.contracts import RerankInput, RerankResult
from app.reranking.interface import Reranker
from app.reranking.metrics import RequiredEvidenceRankMovement, compute_delta, compute_required_evidence_rank_movement
from app.reranking.metrics import mrr as reranked_mrr
from app.reranking.metrics import recall_at_k as reranked_recall_at_k
from app.reranking.pacing import CallPacer
from app.retrieval.runner import RetrievalReport

# (index, total, case_id) -> None. Operational telemetry only -- never
# passed evidence content, provider payloads, or ground truth (see
# scripts/run_reranking_eval.py's own callbacks, which print only these
# three values).
ProgressCallback = Callable[[int, int, str], None]


class RerankingCaseReport(BaseModel):
    """Per-case before/after retrieval-quality comparison -- no agent
    finding, no citation_validity; purely a reranking-quality record."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    recall_at_k_before: dict[int, Optional[float]]
    recall_at_k_after: dict[int, Optional[float]]
    recall_at_k_delta: dict[int, Optional[float]]
    mrr_before: Optional[float]
    mrr_after: Optional[float]
    mrr_delta: Optional[float]
    required_evidence_rank_movement: list[RequiredEvidenceRankMovement]
    reranking_latency_ms: float = Field(ge=0.0)


class RerankingReport(BaseModel):
    """The full, persistable outcome of one reranking-only evaluation run.

    Provenance answers both "what produced this reranking run" (``run_id``/
    ``git_commit_sha``) and "what candidate set did it evaluate" (the
    ``source_retrieval_*`` fields) -- the same discipline
    ``app.retrieval.runner.RetrievalReport`` applies to its own upstream
    dataset/corpus provenance.

    ``reranker_provider``/``requested_reranker_model``/``served_reranker_model``
    are all ``None`` for a pass-through or other local/deterministic
    reranker -- never fabricated for one that cannot legitimately report
    them (see ``app.reranking.provenance.RerankExperimentProvenance``,
    whose fields this report's own provenance section mirrors).
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    git_commit_sha: Optional[str] = None
    generated_at: datetime

    dataset_version: str = Field(min_length=1)
    dataset_fingerprint: str = Field(min_length=1)
    corpus_fingerprint: str = Field(min_length=1)

    source_retrieval_run_id: str = Field(min_length=1)
    source_retrieval_artifact_sha256: str = Field(min_length=1)
    source_retriever_config_id: str = Field(min_length=1)

    candidate_depth: int = Field(ge=1)
    output_depth: int = Field(ge=1)

    reranker_config_id: str = Field(min_length=1)
    reranker_provider: Optional[str] = None
    requested_reranker_model: Optional[str] = None
    served_reranker_model: Optional[str] = None

    execution_policy_id: Optional[str] = None

    evaluation_k_values: list[int]

    case_reports: list[RerankingCaseReport]
    rerank_results: list[RerankResult]

    aggregate_recall_at_k_before: dict[int, float]
    aggregate_recall_at_k_after: dict[int, float]
    aggregate_recall_at_k_delta: dict[int, float]
    aggregate_mrr_before: float
    aggregate_mrr_after: float
    aggregate_mrr_delta: float

    total_reranking_latency_ms: float = Field(ge=0.0)


def run_reranking_evaluation(
    report: RetrievalReport,
    reranker: Reranker,
    candidate_depth: int,
    output_depth: int,
    k_values: tuple[int, ...],
    required_evidence_ids_by_case: dict[str, list[str]],
    source_retrieval_artifact_sha256: str,
    reranker_config_id: str,
    reranker_provider: Optional[str] = None,
    requested_reranker_model: Optional[str] = None,
    served_reranker_model: Optional[str] = None,
    execution_policy_id: Optional[str] = None,
    pacer: Optional[CallPacer] = None,
    on_case_start: Optional[ProgressCallback] = None,
    on_case_complete: Optional[ProgressCallback] = None,
    git_commit_sha: Optional[str] = None,
    run_id: Optional[str] = None,
) -> RerankingReport:
    """Reranks every case's preserved candidate set from ``report`` (via
    ``app.reranking.candidate_source.build_candidate_sets`` -- a pure
    truncation to ``candidate_depth``, never a re-retrieval) and scores
    each one before vs. after.

    ``required_evidence_ids_by_case`` supplies ground truth explicitly, by
    the caller -- this function never resolves it itself (mirrors
    ``app.evaluation.evaluator``'s own "ground truth arrives only after
    the finding already exists" discipline: here, only after the rerank
    already happened).

    ``run_id`` defaults to a fresh UUID4 when not supplied; a real
    experiment caller supplies one resolved BEFORE this function runs (see
    ``scripts/run_reranking_eval.py``), so a precomputed output path can be
    collision-checked before any provider call.

    ``pacer``, when supplied, has ``wait_for_next_call()`` invoked
    immediately before each case's ``reranker.rerank()`` call -- never
    after the loop's final case, never before the very first (the pacer's
    own logic already skips sleeping on its first call). Pacing changes
    only WHEN a call happens, never its semantic content.

    ``on_case_start``/``on_case_complete`` are operational telemetry only
    (see ``scripts/run_reranking_eval.py``, which uses them to print
    ``case N/30: starting``/``completed`` and nothing else) -- if this
    function raises partway through, whatever the caller's own callbacks
    have already recorded (e.g. attempted/completed counts, the in-flight
    case ID) is the caller's only record of progress; this function itself
    persists nothing partial.
    """

    if run_id is None:
        run_id = str(uuid4())

    query_by_case = {result.case_id: result.query for result in report.results}
    candidate_sets = build_candidate_sets(report, candidate_depth)
    total_cases = len(candidate_sets)

    case_reports: list[RerankingCaseReport] = []
    rerank_results: list[RerankResult] = []
    recall_before_by_k: dict[int, list[Optional[float]]] = {k: [] for k in k_values}
    recall_after_by_k: dict[int, list[Optional[float]]] = {k: [] for k in k_values}
    mrr_before_values: list[Optional[float]] = []
    mrr_after_values: list[Optional[float]] = []

    total_started = time.perf_counter()
    for index, (case_id, candidates) in enumerate(candidate_sets.items(), start=1):
        if on_case_start is not None:
            on_case_start(index, total_cases, case_id)

        query = query_by_case[case_id]
        rerank_input = RerankInput(
            case_id=case_id, query=query, candidates=candidates, candidate_depth=len(candidates)
        )
        required_ids = required_evidence_ids_by_case.get(case_id, [])

        if pacer is not None:
            pacer.wait_for_next_call()

        case_started = time.perf_counter()
        rerank_result = reranker.rerank(rerank_input, final_k=output_depth)
        latency_ms = (time.perf_counter() - case_started) * 1000.0

        recall_before = {k: retrieval_metrics.recall_at_k(required_ids, candidates, k) for k in k_values}
        recall_after = {k: reranked_recall_at_k(required_ids, rerank_result.candidates, k) for k in k_values}
        recall_delta = {k: compute_delta(recall_before[k], recall_after[k]) for k in k_values}

        mrr_before = retrieval_metrics.mrr(required_ids, candidates)
        mrr_after = reranked_mrr(required_ids, rerank_result.candidates)
        mrr_delta = compute_delta(mrr_before, mrr_after)

        movement = compute_required_evidence_rank_movement(required_ids, candidates, rerank_result.candidates)

        for k in k_values:
            recall_before_by_k[k].append(recall_before[k])
            recall_after_by_k[k].append(recall_after[k])
        mrr_before_values.append(mrr_before)
        mrr_after_values.append(mrr_after)

        rerank_results.append(rerank_result)
        case_reports.append(
            RerankingCaseReport(
                case_id=case_id,
                recall_at_k_before=recall_before,
                recall_at_k_after=recall_after,
                recall_at_k_delta=recall_delta,
                mrr_before=mrr_before,
                mrr_after=mrr_after,
                mrr_delta=mrr_delta,
                required_evidence_rank_movement=movement,
                reranking_latency_ms=latency_ms,
            )
        )
        if on_case_complete is not None:
            on_case_complete(index, total_cases, case_id)
    total_latency_ms = (time.perf_counter() - total_started) * 1000.0

    aggregate_recall_before = {
        k: retrieval_metrics.aggregate_recall_at_k(v) for k, v in recall_before_by_k.items()
    }
    aggregate_recall_after = {
        k: retrieval_metrics.aggregate_recall_at_k(v) for k, v in recall_after_by_k.items()
    }
    aggregate_mrr_before = retrieval_metrics.aggregate_mrr(mrr_before_values)
    aggregate_mrr_after = retrieval_metrics.aggregate_mrr(mrr_after_values)

    return RerankingReport(
        run_id=run_id,
        git_commit_sha=git_commit_sha,
        generated_at=datetime.now(timezone.utc),
        dataset_version=report.dataset_version,
        dataset_fingerprint=report.dataset_fingerprint,
        corpus_fingerprint=report.corpus_fingerprint,
        source_retrieval_run_id=report.run_id,
        source_retrieval_artifact_sha256=source_retrieval_artifact_sha256,
        source_retriever_config_id=report.retriever_config_id,
        candidate_depth=candidate_depth,
        output_depth=output_depth,
        reranker_config_id=reranker_config_id,
        reranker_provider=reranker_provider,
        requested_reranker_model=requested_reranker_model,
        served_reranker_model=served_reranker_model,
        execution_policy_id=execution_policy_id,
        evaluation_k_values=list(k_values),
        case_reports=case_reports,
        rerank_results=rerank_results,
        aggregate_recall_at_k_before=aggregate_recall_before,
        aggregate_recall_at_k_after=aggregate_recall_after,
        aggregate_recall_at_k_delta={
            # aggregate_recall_at_k always returns a real float (never
            # None), so compute_delta here is never None either -- both
            # aggregate_recall_before/after are populated for every k.
            k: compute_delta(aggregate_recall_before[k], aggregate_recall_after[k])
            for k in k_values
        },
        aggregate_mrr_before=aggregate_mrr_before,
        aggregate_mrr_after=aggregate_mrr_after,
        aggregate_mrr_delta=compute_delta(aggregate_mrr_before, aggregate_mrr_after),
        total_reranking_latency_ms=total_latency_ms,
    )
