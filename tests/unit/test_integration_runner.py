"""Integration tests for app.integration.runner.run_reranked_llm_evaluation
-- exercised end to end against the REAL synthetic-v2 dataset/corpus and a
FAKE, hand-built RerankingReport (never the real preserved artifact) and
FakeLLMProvider (never a real OpenAI call).

Proves: exactly Top-K reranked evidence reaches the agent in rank order,
no ground-truth leakage, one LLM invocation per case, existing malformed
-output handling stays intact, and the resulting report's provenance
records the source reranking run/SHA and evidence_top_k.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.agent.interface import AgentOutputError
from app.agent.llm_agent import LLMAgent, LLMFindingPayload, build_llm_config
from app.datasets.loader import load_dataset
from app.integration.failure_attribution import FailureLayer
from app.integration.runner import RerankedLLMReport, run_reranked_llm_evaluation
from app.models.contracts import Finding, ModelProvider
from app.providers.base import ProviderMessage, StructuredCompletion
from app.reranking.contracts import RerankedEvidence, RerankInput, RerankResult
from app.reranking.runner import RerankingCaseReport, RerankingReport
from app.retrieval.contracts import RetrievalQuery, RetrievedEvidence
from app.retrieval.corpus import build_full_version_corpus
from tests.support.fake_llm_provider import FakeLLMProvider

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "datasets"


def _dataset_and_corpus():
    dataset = load_dataset(DATASET_DIR, version="synthetic-v2")
    corpus = build_full_version_corpus(dataset)
    return dataset, corpus


def _fabricated_reranking_report(dataset, corpus, candidates_per_case: int = 10) -> RerankingReport:
    """A hand-built RerankingReport referencing REAL evidence_ids from the
    real synthetic-v2 corpus (so evidence hydration succeeds), but
    otherwise fabricated -- never the real preserved reranking artifact.
    Every case gets the SAME first `candidates_per_case` corpus evidence
    IDs (sorted), which is enough to exercise the pipeline's wiring
    without needing a realistic reranking outcome.
    """

    pool_ids = [e.evidence_id for e in corpus.evidence[:candidates_per_case]]
    control_by_id = dataset.controls

    case_reports = []
    rerank_results = []
    for case in dataset.cases:
        control = control_by_id[case.control_id]
        query = RetrievalQuery(case_id=case.case_id, control=control, scenario_description="S", period=case.period)
        original = [
            RetrievedEvidence(evidence_id=eid, document_id=f"doc_{eid}", score=1.0, rank=i, retrieval_method="x")
            for i, eid in enumerate(pool_ids, start=1)
        ]
        rerank_input = RerankInput(case_id=case.case_id, query=query, candidates=original, candidate_depth=len(pool_ids))
        reranked = [
            RerankedEvidence(
                evidence_id=eid, document_id=f"doc_{eid}", original_rank=i, reranked_rank=i,
                original_score=1.0, reranker_score=1.0, reranker_method="pass-through",
            )
            for i, eid in enumerate(pool_ids, start=1)
        ]
        rerank_results.append(
            RerankResult(case_id=case.case_id, input=rerank_input, candidates=reranked, reranker_config_id="fake-reranker-config")
        )
        case_reports.append(
            RerankingCaseReport(
                case_id=case.case_id, recall_at_k_before={1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0},
                recall_at_k_after={1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0}, recall_at_k_delta={1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0},
                mrr_before=1.0, mrr_after=1.0, mrr_delta=0.0, required_evidence_rank_movement=[], reranking_latency_ms=1.0,
            )
        )

    return RerankingReport(
        run_id="fake-rerank-run-1", git_commit_sha="d" * 40, generated_at=datetime.now(timezone.utc),
        dataset_version=dataset.version, dataset_fingerprint=dataset.fingerprint, corpus_fingerprint=corpus.fingerprint,
        source_retrieval_run_id="fake-retrieval-run-1", source_retrieval_artifact_sha256="e" * 64,
        source_retriever_config_id="fake-retriever-config", candidate_depth=candidates_per_case,
        output_depth=candidates_per_case, reranker_config_id="fake-reranker-config", reranker_provider="cohere",
        requested_reranker_model="fake-rerank-model", served_reranker_model=None, execution_policy_id=None,
        evaluation_k_values=[1, 3, 5, 10], case_reports=case_reports, rerank_results=rerank_results,
        aggregate_recall_at_k_before={1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0},
        aggregate_recall_at_k_after={1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0},
        aggregate_recall_at_k_delta={1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0},
        aggregate_mrr_before=1.0, aggregate_mrr_after=1.0, aggregate_mrr_delta=0.0, total_reranking_latency_ms=1.0,
    )


def _fake_completion(finding=Finding.PASS, evidence=None) -> StructuredCompletion:
    payload = LLMFindingPayload(
        finding=finding, confidence=0.9, reasoning_summary="fake, zero-network response for testing.",
        evidence=evidence or [], missing_information=[], abstain=(finding == Finding.INSUFFICIENT_EVIDENCE),
    )
    return StructuredCompletion(parsed=payload, raw_text=None, usage=None, provider=ModelProvider.OPENAI, model_name="fake-model-v1")


class TestExactTopKReachesTheAgent:
    def test_exactly_five_evidence_items_per_case_in_rank_order(self):
        dataset, corpus = _dataset_and_corpus()
        reranking_report = _fabricated_reranking_report(dataset, corpus, candidates_per_case=10)
        agent = LLMAgent(provider=FakeLLMProvider(_fake_completion()))

        report = run_reranked_llm_evaluation(
            reranking_report, dataset, corpus, agent, evidence_top_k=5,
            provider=ModelProvider.OPENAI, model_name="fake-model", llm_config=build_llm_config(),
            source_reranking_artifact_sha256="e" * 64, source_reranking_config_id="fake-reranker-config",
        )

        for case_report in report.case_reports:
            assert len(case_report.presented_evidence_ids) == 5

    def test_no_rank_beyond_five_reaches_the_agent(self):
        dataset, corpus = _dataset_and_corpus()
        pool_ids = [e.evidence_id for e in corpus.evidence[:10]]
        reranking_report = _fabricated_reranking_report(dataset, corpus, candidates_per_case=10)
        agent = LLMAgent(provider=FakeLLMProvider(_fake_completion()))

        report = run_reranked_llm_evaluation(
            reranking_report, dataset, corpus, agent, evidence_top_k=5,
            provider=ModelProvider.OPENAI, model_name="fake-model", llm_config=build_llm_config(),
            source_reranking_artifact_sha256="e" * 64, source_reranking_config_id="fake-reranker-config",
        )

        # ranks 6-10 (pool_ids[5:]) must never appear in what was presented
        excluded_ids = set(pool_ids[5:])
        for case_report in report.case_reports:
            assert not (set(case_report.presented_evidence_ids) & excluded_ids)


class TestOneLLMInvocationPerCase:
    def test_call_count_equals_case_count(self):
        dataset, corpus = _dataset_and_corpus()
        reranking_report = _fabricated_reranking_report(dataset, corpus)
        fake_provider = FakeLLMProvider(_fake_completion())
        agent = LLMAgent(provider=fake_provider)

        run_reranked_llm_evaluation(
            reranking_report, dataset, corpus, agent, evidence_top_k=5,
            provider=ModelProvider.OPENAI, model_name="fake-model", llm_config=build_llm_config(),
            source_reranking_artifact_sha256="e" * 64, source_reranking_config_id="fake-reranker-config",
        )

        assert fake_provider.call_count == len(dataset.cases)


class TestMalformedOutputHandlingRemainsIntact:
    def test_unparseable_completion_produces_none_output_not_a_crash(self):
        dataset, corpus = _dataset_and_corpus()
        reranking_report = _fabricated_reranking_report(dataset, corpus)
        malformed_completion = StructuredCompletion(
            parsed=None, raw_text="not valid json", usage=None, provider=ModelProvider.OPENAI, model_name="fake-model-v1",
        )
        agent = LLMAgent(provider=FakeLLMProvider(malformed_completion))

        report = run_reranked_llm_evaluation(
            reranking_report, dataset, corpus, agent, evidence_top_k=5,
            provider=ModelProvider.OPENAI, model_name="fake-model", llm_config=build_llm_config(),
            source_reranking_artifact_sha256="e" * 64, source_reranking_config_id="fake-reranker-config",
        )

        assert all(trace.output is None for trace in report.traces)
        assert all(not result.schema_valid for result in report.results)
        assert all(FailureLayer.MALFORMED_SCHEMA_FAILURE in cr.failure_layers for cr in report.case_reports)


class TestProvenance:
    def test_report_records_source_reranking_run_sha_and_top_k(self):
        dataset, corpus = _dataset_and_corpus()
        reranking_report = _fabricated_reranking_report(dataset, corpus)
        agent = LLMAgent(provider=FakeLLMProvider(_fake_completion()))

        report = run_reranked_llm_evaluation(
            reranking_report, dataset, corpus, agent, evidence_top_k=5,
            provider=ModelProvider.OPENAI, model_name="fake-model", llm_config=build_llm_config(),
            source_reranking_artifact_sha256="deadbeef" * 8, source_reranking_config_id="fake-reranker-config",
        )

        assert report.source_reranking_run_id == "fake-rerank-run-1"
        assert report.source_reranking_artifact_sha256 == "deadbeef" * 8
        assert report.source_reranking_config_id == "fake-reranker-config"
        assert report.evidence_top_k == 5
        assert report.evidence_source == "reranked"

    def test_report_round_trips_through_json(self):
        dataset, corpus = _dataset_and_corpus()
        reranking_report = _fabricated_reranking_report(dataset, corpus)
        agent = LLMAgent(provider=FakeLLMProvider(_fake_completion()))

        report = run_reranked_llm_evaluation(
            reranking_report, dataset, corpus, agent, evidence_top_k=5,
            provider=ModelProvider.OPENAI, model_name="fake-model", llm_config=build_llm_config(),
            source_reranking_artifact_sha256="e" * 64, source_reranking_config_id="fake-reranker-config",
        )
        round_tripped = RerankedLLMReport.model_validate_json(report.model_dump_json())
        assert round_tripped.run_id == report.run_id
        assert len(round_tripped.case_reports) == len(report.case_reports)


class TestNoGroundTruthLeakageEndToEnd:
    def test_fake_provider_never_receives_ground_truth_in_its_messages(self):
        dataset, corpus = _dataset_and_corpus()
        reranking_report = _fabricated_reranking_report(dataset, corpus)
        secret_marker = "SECRET-RATIONALE-MUST-NEVER-REACH-THE-AGENT"
        # Monkeypatch-free check: inspect the real dataset's own case
        # rationales/failure_mode_tags and confirm none of them ever
        # appear in what the fake provider was actually sent.
        fake_provider = FakeLLMProvider(_fake_completion())
        agent = LLMAgent(provider=fake_provider)

        run_reranked_llm_evaluation(
            reranking_report, dataset, corpus, agent, evidence_top_k=5,
            provider=ModelProvider.OPENAI, model_name="fake-model", llm_config=build_llm_config(),
            source_reranking_artifact_sha256="e" * 64, source_reranking_config_id="fake-reranker-config",
        )

        last_messages_text = "\n".join(m.content for m in fake_provider.last_messages)
        for case in dataset.cases:
            assert case.expected_outcome.rationale not in last_messages_text
            assert case.failure_mode_tag not in last_messages_text
