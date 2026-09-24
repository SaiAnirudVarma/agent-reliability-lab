"""Unit tests for app.integration.source: validating a preserved
reranking artifact against a RerankedLLMBaselineConfig BEFORE any LLM
provider call, and truncating to the configured evidence_top_k.

All fixtures here are hand-built, fabricated RerankingReports written to
tmp_path -- never the real preserved artifact -- so these tests cannot
accidentally mutate or depend on the real repository's results/.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.integration.config import RerankedLLMBaselineConfig
from app.integration.source import (
    ArtifactVerificationError,
    RerankedEvidenceSourceValidationError,
    build_top_k_candidates_by_case,
    load_and_validate_reranking_source,
)
from app.reranking.contracts import RerankedEvidence, RerankInput, RerankResult
from app.reranking.runner import RerankingCaseReport, RerankingReport
from app.models.contracts import Control
from app.retrieval.contracts import RetrievalQuery, RetrievedEvidence

DATASET_FINGERPRINT = "b" * 64
CORPUS_FINGERPRINT = "c" * 64
GIT_COMMIT_SHA = "d" * 40


def _control() -> Control:
    return Control(control_id="CTRL-X", name="N", description="D", requirement_text="R")


def _query(case_id: str) -> RetrievalQuery:
    return RetrievalQuery(case_id=case_id, control=_control(), scenario_description="S", period="2025-Q1")


def _rerank_result(case_id: str, num_candidates: int) -> RerankResult:
    original = [
        RetrievedEvidence(evidence_id=f"EV-{case_id}-{i}", document_id=f"doc_{case_id}_{i}", score=1.0, rank=i, retrieval_method="x")
        for i in range(1, num_candidates + 1)
    ]
    rerank_input = RerankInput(case_id=case_id, query=_query(case_id), candidates=original, candidate_depth=num_candidates)
    reranked = [
        RerankedEvidence(
            evidence_id=f"EV-{case_id}-{i}", document_id=f"doc_{case_id}_{i}", original_rank=i, reranked_rank=i,
            original_score=1.0, reranker_score=1.0, reranker_method="pass-through",
        )
        for i in range(1, num_candidates + 1)
    ]
    return RerankResult(case_id=case_id, input=rerank_input, candidates=reranked, reranker_config_id="test-reranker-config")


def _case_report(case_id: str) -> RerankingCaseReport:
    return RerankingCaseReport(
        case_id=case_id, recall_at_k_before={1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0},
        recall_at_k_after={1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0}, recall_at_k_delta={1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0},
        mrr_before=1.0, mrr_after=1.0, mrr_delta=0.0, required_evidence_rank_movement=[], reranking_latency_ms=1.0,
    )


def _fabricated_report(
    num_cases: int = 30, candidates_per_case: int = 10, run_id="rerank-run-1", git_commit_sha=GIT_COMMIT_SHA,
    evaluation_k_values=(1, 3, 5, 10),
) -> RerankingReport:
    case_ids = [f"AC-{i:03d}" for i in range(1, num_cases + 1)]
    return RerankingReport(
        run_id=run_id, git_commit_sha=git_commit_sha, generated_at=datetime.now(timezone.utc),
        dataset_version="synthetic-v2", dataset_fingerprint=DATASET_FINGERPRINT, corpus_fingerprint=CORPUS_FINGERPRINT,
        source_retrieval_run_id="retrieval-run-1", source_retrieval_artifact_sha256="e" * 64,
        source_retriever_config_id="retrieval-baseline-v1", candidate_depth=candidates_per_case,
        output_depth=candidates_per_case, reranker_config_id="test-reranker-config", reranker_provider="cohere",
        requested_reranker_model="rerank-v4.0-pro", served_reranker_model=None, execution_policy_id=None,
        evaluation_k_values=list(evaluation_k_values),
        case_reports=[_case_report(cid) for cid in case_ids],
        rerank_results=[_rerank_result(cid, candidates_per_case) for cid in case_ids],
        aggregate_recall_at_k_before={1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0},
        aggregate_recall_at_k_after={1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0},
        aggregate_recall_at_k_delta={1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0},
        aggregate_mrr_before=1.0, aggregate_mrr_after=1.0, aggregate_mrr_delta=0.0,
        total_reranking_latency_ms=1.0,
    )


def _write_report(tmp_path: Path, report: RerankingReport, filename="artifact.json") -> tuple[Path, str]:
    path = tmp_path / filename
    content = report.model_dump_json(indent=2) + "\n"
    path.write_text(content)
    sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return path, sha256


def _config_for(sha256: str, **overrides) -> RerankedLLMBaselineConfig:
    defaults = dict(
        config_id="test-reranked-llm-config",
        dataset_version="synthetic-v2", dataset_fingerprint=DATASET_FINGERPRINT, corpus_fingerprint=CORPUS_FINGERPRINT,
        source_reranking_run_id="rerank-run-1", source_reranking_artifact_sha256=sha256,
        source_reranking_config_id="test-reranker-config", evidence_top_k=5,
        llm_provider="openai", requested_model="fake-model", prompt_version="llm-baseline-v1",
        temperature=0.0, max_output_tokens=800,
    )
    defaults.update(overrides)
    return RerankedLLMBaselineConfig(**defaults)


class TestLoadAndValidateReankingSourceSuccess:
    def test_valid_artifact_loads(self, tmp_path):
        report = _fabricated_report()
        path, sha256 = _write_report(tmp_path, report)
        config = _config_for(sha256)
        loaded = load_and_validate_reranking_source(config, path)
        assert loaded.run_id == "rerank-run-1"
        assert len(loaded.rerank_results) == 30

    def test_build_top_k_candidates_truncates_and_sorts_by_reranked_rank(self, tmp_path):
        report = _fabricated_report(candidates_per_case=10)
        path, sha256 = _write_report(tmp_path, report)
        config = _config_for(sha256, evidence_top_k=5)
        loaded = load_and_validate_reranking_source(config, path)
        by_case = build_top_k_candidates_by_case(loaded, config.evidence_top_k)
        assert all(len(candidates) == 5 for candidates in by_case.values())
        assert [c.reranked_rank for c in by_case["AC-001"]] == [1, 2, 3, 4, 5]


class TestReankingSourceFailsClosed:
    def test_wrong_sha256_raises(self, tmp_path):
        report = _fabricated_report()
        path, _real_sha256 = _write_report(tmp_path, report)
        config = _config_for("0" * 64)
        with pytest.raises(ArtifactVerificationError):
            load_and_validate_reranking_source(config, path)

    def test_wrong_run_id_raises(self, tmp_path):
        report = _fabricated_report(run_id="actual-run-id")
        path, sha256 = _write_report(tmp_path, report)
        config = _config_for(sha256, source_reranking_run_id="different-run-id")
        with pytest.raises(RerankedEvidenceSourceValidationError, match="run_id"):
            load_and_validate_reranking_source(config, path)

    def test_wrong_dataset_fingerprint_raises(self, tmp_path):
        report = _fabricated_report()
        path, sha256 = _write_report(tmp_path, report)
        config = _config_for(sha256, dataset_fingerprint="9" * 64)
        with pytest.raises(ArtifactVerificationError, match="dataset_fingerprint"):
            load_and_validate_reranking_source(config, path)

    def test_wrong_corpus_fingerprint_raises(self, tmp_path):
        report = _fabricated_report()
        path, sha256 = _write_report(tmp_path, report)
        config = _config_for(sha256, corpus_fingerprint="9" * 64)
        with pytest.raises(ArtifactVerificationError, match="corpus_fingerprint"):
            load_and_validate_reranking_source(config, path)

    def test_wrong_reranker_config_id_raises(self, tmp_path):
        report = _fabricated_report()
        path, sha256 = _write_report(tmp_path, report)
        config = _config_for(sha256, source_reranking_config_id="not-the-real-one")
        with pytest.raises(ArtifactVerificationError, match="reranker_config_id"):
            load_and_validate_reranking_source(config, path)

    def test_missing_git_commit_sha_raises(self, tmp_path):
        report = _fabricated_report(git_commit_sha=None)
        path, sha256 = _write_report(tmp_path, report)
        config = _config_for(sha256)
        with pytest.raises(RerankedEvidenceSourceValidationError, match="git_commit_sha"):
            load_and_validate_reranking_source(config, path)

    def test_wrong_number_of_results_raises(self, tmp_path):
        report = _fabricated_report(num_cases=29)
        path, sha256 = _write_report(tmp_path, report)
        config = _config_for(sha256)
        with pytest.raises(RerankedEvidenceSourceValidationError, match="expected exactly 30"):
            load_and_validate_reranking_source(config, path)

    def test_insufficient_evidence_top_k_raises(self, tmp_path):
        report = _fabricated_report(candidates_per_case=3)
        path, sha256 = _write_report(tmp_path, report)
        config = _config_for(sha256, evidence_top_k=5)
        with pytest.raises(RerankedEvidenceSourceValidationError, match="evidence_top_k"):
            load_and_validate_reranking_source(config, path)

    def test_evidence_top_k_not_among_evaluated_k_values_raises(self, tmp_path):
        report = _fabricated_report(candidates_per_case=10, evaluation_k_values=(1, 3, 10))
        path, sha256 = _write_report(tmp_path, report)
        config = _config_for(sha256, evidence_top_k=5)
        with pytest.raises(RerankedEvidenceSourceValidationError, match="evaluation_k_values"):
            load_and_validate_reranking_source(config, path)

    def test_missing_artifact_file_raises_oserror(self, tmp_path):
        config = _config_for("0" * 64)
        with pytest.raises(OSError):
            load_and_validate_reranking_source(config, tmp_path / "does-not-exist.json")
