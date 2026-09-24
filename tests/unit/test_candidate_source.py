"""Unit tests for app.reranking.candidate_source: validating a preserved
retrieval artifact against a RerankerBaselineConfig BEFORE any reranker
provider call, and truncating to the configured candidate_depth.

All fixtures here are hand-built, fabricated RetrievalReports written to
tmp_path -- never the real preserved artifact -- so these tests cannot
accidentally mutate or depend on the real repository's results/.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.reranking.artifact_loader import ArtifactVerificationError
from app.reranking.baseline_config import RerankerBaselineConfig
from app.reranking.candidate_source import (
    CandidateSourceValidationError,
    build_candidate_sets,
    load_and_validate_candidate_source,
)
from app.models.contracts import Control
from app.retrieval.contracts import RetrievalQuery, RetrievalResult, RetrievedEvidence
from app.retrieval.runner import RetrievalReport

DATASET_FINGERPRINT = "b" * 64
CORPUS_FINGERPRINT = "c" * 64
GIT_COMMIT_SHA = "d" * 40


def _control() -> Control:
    return Control(control_id="CTRL-X", name="N", description="D", requirement_text="R")


def _query(case_id: str) -> RetrievalQuery:
    return RetrievalQuery(case_id=case_id, control=_control(), scenario_description="S", period="2025-Q1")


def _candidate(evidence_id: str, rank: int) -> RetrievedEvidence:
    return RetrievedEvidence(evidence_id=evidence_id, document_id=f"doc_{evidence_id}", score=1.0, rank=rank, retrieval_method="x")


def _result(case_id: str, num_candidates: int) -> RetrievalResult:
    candidates = [_candidate(f"EV-{case_id}-{i}", i) for i in range(1, num_candidates + 1)]
    return RetrievalResult(case_id=case_id, query=_query(case_id), candidates=candidates, top_k=num_candidates, retriever_config_id="retrieval-baseline-v1")


def _fabricated_report(num_cases: int = 30, candidates_per_case: int = 10, run_id="run-1", git_commit_sha=GIT_COMMIT_SHA) -> RetrievalReport:
    results = [_result(f"AC-{i:03d}", candidates_per_case) for i in range(1, num_cases + 1)]
    return RetrievalReport(
        run_id=run_id, git_commit_sha=git_commit_sha, dataset_version="synthetic-v2",
        dataset_fingerprint=DATASET_FINGERPRINT, corpus_fingerprint=CORPUS_FINGERPRINT,
        retriever_config_id="retrieval-baseline-v1", top_k_values=[1, 3, 5, 10],
        case_metrics=[], results=results, aggregate_recall_at_k={1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0},
        aggregate_mrr=1.0, total_retrieval_latency_ms=1.0, generated_at=datetime.now(timezone.utc),
    )


def _write_report(tmp_path: Path, report: RetrievalReport, filename="artifact.json") -> tuple[Path, str]:
    path = tmp_path / filename
    content = report.model_dump_json(indent=2) + "\n"
    path.write_text(content)
    sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return path, sha256


def _config_for(sha256: str, **overrides) -> RerankerBaselineConfig:
    defaults = dict(
        config_id="test-config", provider="cohere", requested_model="rerank-v4.0-pro", served_model=None,
        candidate_source_type="preserved_retrieval_artifact", candidate_source_run_id="run-1",
        candidate_source_artifact_sha256=sha256, candidate_source_retriever_config_id="retrieval-baseline-v1",
        candidate_depth=10, reranker_output_depth=10, evaluation_k_values=[1, 3, 5, 10],
        dataset_version="synthetic-v2", dataset_fingerprint=DATASET_FINGERPRINT, corpus_fingerprint=CORPUS_FINGERPRINT,
        query_serialization="x", document_serialization="x",
    )
    defaults.update(overrides)
    return RerankerBaselineConfig(**defaults)


class TestLoadAndValidateCandidateSourceSuccess:
    def test_valid_artifact_loads(self, tmp_path):
        report = _fabricated_report()
        path, sha256 = _write_report(tmp_path, report)
        config = _config_for(sha256)
        loaded = load_and_validate_candidate_source(config, path)
        assert loaded.run_id == "run-1"
        assert len(loaded.results) == 30

    def test_build_candidate_sets_truncates_to_depth(self, tmp_path):
        report = _fabricated_report(candidates_per_case=10)
        path, sha256 = _write_report(tmp_path, report)
        config = _config_for(sha256, candidate_depth=3, reranker_output_depth=3, evaluation_k_values=[1, 3])
        loaded = load_and_validate_candidate_source(config, path)
        sets = build_candidate_sets(loaded, config.candidate_depth)
        assert all(len(candidates) == 3 for candidates in sets.values())
        assert [c.rank for c in sets["AC-001"]] == [1, 2, 3]


class TestCandidateSourceFailsClosed:
    def test_wrong_sha256_raises(self, tmp_path):
        report = _fabricated_report()
        path, _real_sha256 = _write_report(tmp_path, report)
        config = _config_for("0" * 64)
        with pytest.raises(ArtifactVerificationError):
            load_and_validate_candidate_source(config, path)

    def test_wrong_run_id_raises(self, tmp_path):
        report = _fabricated_report(run_id="actual-run-id")
        path, sha256 = _write_report(tmp_path, report)
        config = _config_for(sha256, candidate_source_run_id="different-run-id")
        with pytest.raises(CandidateSourceValidationError, match="run_id"):
            load_and_validate_candidate_source(config, path)

    def test_wrong_dataset_fingerprint_raises(self, tmp_path):
        report = _fabricated_report()
        path, sha256 = _write_report(tmp_path, report)
        config = _config_for(sha256, dataset_fingerprint="9" * 64)
        with pytest.raises(ArtifactVerificationError, match="dataset_fingerprint"):
            load_and_validate_candidate_source(config, path)

    def test_wrong_corpus_fingerprint_raises(self, tmp_path):
        report = _fabricated_report()
        path, sha256 = _write_report(tmp_path, report)
        config = _config_for(sha256, corpus_fingerprint="9" * 64)
        with pytest.raises(ArtifactVerificationError, match="corpus_fingerprint"):
            load_and_validate_candidate_source(config, path)

    def test_wrong_retriever_config_id_raises(self, tmp_path):
        report = _fabricated_report()
        path, sha256 = _write_report(tmp_path, report)
        config = _config_for(sha256, candidate_source_retriever_config_id="not-the-real-one")
        with pytest.raises(ArtifactVerificationError, match="retriever_config_id"):
            load_and_validate_candidate_source(config, path)

    def test_missing_git_commit_sha_raises(self, tmp_path):
        report = _fabricated_report(git_commit_sha=None)
        path, sha256 = _write_report(tmp_path, report)
        config = _config_for(sha256)
        with pytest.raises(CandidateSourceValidationError, match="git_commit_sha"):
            load_and_validate_candidate_source(config, path)

    def test_wrong_number_of_results_raises(self, tmp_path):
        report = _fabricated_report(num_cases=29)
        path, sha256 = _write_report(tmp_path, report)
        config = _config_for(sha256)
        with pytest.raises(CandidateSourceValidationError, match="expected exactly 30"):
            load_and_validate_candidate_source(config, path)

    def test_insufficient_candidate_depth_raises(self, tmp_path):
        report = _fabricated_report(candidates_per_case=5)
        path, sha256 = _write_report(tmp_path, report)
        config = _config_for(sha256, candidate_depth=10, reranker_output_depth=10)
        with pytest.raises(CandidateSourceValidationError, match="candidate_depth"):
            load_and_validate_candidate_source(config, path)

    def test_missing_artifact_file_raises_oserror(self, tmp_path):
        config = _config_for("0" * 64)
        with pytest.raises(OSError):
            load_and_validate_candidate_source(config, tmp_path / "does-not-exist.json")
