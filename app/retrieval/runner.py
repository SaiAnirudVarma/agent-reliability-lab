"""Retrieval-only evaluation: for each EvaluationCase, build a
RetrievalQuery, retrieve top-K, score against ground truth (Recall@K,
MRR). No AgentRunner, no LLM invocation, anywhere in this module -- this
measures retrieval quality in isolation, exactly as Phase 7 requires.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.datasets.loader import Dataset
from app.models.contracts import ModelProvider
from app.retrieval.contracts import RetrievalResult, build_retrieval_query
from app.retrieval.corpus import EvidenceCorpus
from app.retrieval.corpus_index import CorpusEmbeddingIndex
from app.retrieval.interface import Retriever
from app.retrieval.metrics import aggregate_mrr, aggregate_recall_at_k, mrr, recall_at_k

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)


class RetrievalCaseMetrics(BaseModel):
    """Per-case retrieval-only metrics -- no agent finding, no
    citation_validity, nothing from ``app.evaluation.metrics``.

    ``latency_ms`` measures ONLY this case's ``retriever.retrieve()`` call
    (query embedding + similarity scoring for a ``VectorRetriever``, or
    pure lexical scoring for ``LexicalBaselineRetriever``). It deliberately
    EXCLUDES corpus embedding, which is a one-time cost incurred once
    before the per-case loop even starts -- see
    ``RetrievalReport.corpus_embedding_latency_ms``. Mixing the two would
    misattribute a fixed setup cost onto whichever case happened to run
    first.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    recall_at_k: dict[int, Optional[float]]
    mrr: Optional[float]
    latency_ms: float = Field(ge=0.0)


class RetrievalReport(BaseModel):
    """The full, persistable outcome of one retrieval-only evaluation run.

    Provenance is held to the same standard as ``RunReport``
    (``app.models.contracts``): ``git_commit_sha`` is captured by the
    caller via ``app.observability.git_provenance.get_git_commit_sha`` --
    the SAME utility ``scripts/run_eval.py`` uses, not a competing
    implementation -- and threaded in here, never re-derived. ``run_id``
    is likewise decided by the caller BEFORE any retrieval happens (see
    ``scripts/run_retrieval_eval.py``), so a real experiment's destination
    artifact path can be computed and collision-checked before any
    provider call, exactly mirroring the LLM experiment pipeline's own
    safety ordering.

    Embedding provenance (``embedding_provider``, ``requested_embedding_model``,
    ``served_embedding_model``, ``embedding_dimension``) is ``None`` for a
    non-embedding retriever (e.g. ``LexicalBaselineRetriever``) and is
    NEVER fabricated for one that cannot legitimately report a served
    model -- see ``app.retrieval.embedding_provider.EmbeddedVector``.
    ``served_embedding_model`` reflects the CORPUS embedding batch's
    reported served model (``CorpusEmbeddingIndex.served_model_name``);
    per-query served model is not separately tracked, since a query and
    the corpus it is compared against always share one provider/model
    configuration in this design.

    ``corpus_fingerprint`` is content identity for exactly what was
    searched (see ``app.retrieval.corpus.compute_corpus_fingerprint`),
    independent of and stronger than ``dataset_fingerprint``/``dataset_version``
    alone.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    git_commit_sha: Optional[str] = None
    dataset_version: str = Field(min_length=1)
    dataset_fingerprint: str = Field(min_length=1)
    corpus_fingerprint: str = Field(min_length=1)
    retriever_config_id: str = Field(min_length=1)
    embedding_provider: Optional[ModelProvider] = None
    requested_embedding_model: Optional[str] = None
    served_embedding_model: Optional[str] = None
    embedding_dimension: Optional[int] = None
    top_k_values: list[int]
    case_metrics: list[RetrievalCaseMetrics]
    results: list[RetrievalResult]
    aggregate_recall_at_k: dict[int, float]
    aggregate_mrr: float
    corpus_embedding_latency_ms: Optional[float] = Field(default=None, ge=0.0)
    total_retrieval_latency_ms: float = Field(ge=0.0)
    generated_at: datetime


def run_retrieval_evaluation(
    dataset: Dataset,
    retriever: Retriever,
    corpus: EvidenceCorpus,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    embedding_index: Optional[CorpusEmbeddingIndex] = None,
    git_commit_sha: Optional[str] = None,
    run_id: Optional[str] = None,
    corpus_embedding_latency_ms: Optional[float] = None,
) -> RetrievalReport:
    """Runs ``retriever`` against every case in ``dataset``, once each
    (a single ``retrieve(top_k=max(k_values))`` call per case -- Recall@1,
    @3, @5, @10 are all computed from that one result's candidates, not
    four separate retrieve() calls).

    Cases are executed in ``dataset.cases`` order, matching
    ``run_evaluation``'s own stability guarantee.

    ``embedding_index``, when the retriever is embedding-backed (e.g. a
    ``VectorRetriever``), supplies this report's embedding-provenance
    fields directly -- it is never re-derived or guessed. Left ``None``
    for a non-embedding retriever.

    ``git_commit_sha`` and ``run_id`` are accepted, not computed here:
    both must be resolved by the CALLER before this function is invoked at
    all for a real experiment, so a provider call is never made before
    provenance and collision safety are already settled (see
    ``scripts/run_retrieval_eval.py``). ``run_id`` defaults to a fresh
    UUID4 when not supplied, for direct/test callers that don't need that
    pre-resolution guarantee.
    """

    max_k = max(k_values)
    if run_id is None:
        run_id = str(uuid4())

    case_metrics: list[RetrievalCaseMetrics] = []
    results: list[RetrievalResult] = []
    recall_values_by_k: dict[int, list[Optional[float]]] = {k: [] for k in k_values}
    mrr_values: list[Optional[float]] = []

    loop_started = time.perf_counter()
    for case in dataset.cases:
        control = dataset.controls[case.control_id]
        query = build_retrieval_query(case, control)

        case_started = time.perf_counter()
        result = retriever.retrieve(query, corpus, top_k=max_k)
        latency_ms = (time.perf_counter() - case_started) * 1000.0

        required_ids = case.expected_outcome.required_evidence_ids
        per_k = {k: recall_at_k(required_ids, result.candidates, k) for k in k_values}
        case_mrr = mrr(required_ids, result.candidates)

        for k in k_values:
            recall_values_by_k[k].append(per_k[k])
        mrr_values.append(case_mrr)

        results.append(result)
        case_metrics.append(
            RetrievalCaseMetrics(case_id=case.case_id, recall_at_k=per_k, mrr=case_mrr, latency_ms=latency_ms)
        )
    total_retrieval_latency_ms = (time.perf_counter() - loop_started) * 1000.0

    return RetrievalReport(
        run_id=run_id,
        git_commit_sha=git_commit_sha,
        dataset_version=dataset.version,
        dataset_fingerprint=dataset.fingerprint,
        corpus_fingerprint=corpus.fingerprint,
        retriever_config_id=retriever.retriever_config_id,
        embedding_provider=embedding_index.provider if embedding_index is not None else None,
        requested_embedding_model=embedding_index.model_name if embedding_index is not None else None,
        served_embedding_model=embedding_index.served_model_name if embedding_index is not None else None,
        embedding_dimension=embedding_index.embedding_dimension if embedding_index is not None else None,
        top_k_values=list(k_values),
        case_metrics=case_metrics,
        results=results,
        aggregate_recall_at_k={k: aggregate_recall_at_k(v) for k, v in recall_values_by_k.items()},
        aggregate_mrr=aggregate_mrr(mrr_values),
        corpus_embedding_latency_ms=corpus_embedding_latency_ms,
        total_retrieval_latency_ms=total_retrieval_latency_ms,
        generated_at=datetime.now(timezone.utc),
    )
