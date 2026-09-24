"""Unit tests for app.retrieval.artifact_io: immutable retrieval-experiment
artifact naming, run-ID resolution, and the atomic writer. Everything here
uses tmp_path only -- no test may touch this repository's real results/.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.retrieval.artifact_io import (
    resolve_run_id,
    retrieval_experiment_path,
    retrieval_experiments_dir,
    write_retrieval_report_atomically,
)
from app.retrieval.runner import RetrievalReport


def _report(run_id: str = "run-1") -> RetrievalReport:
    return RetrievalReport(
        run_id=run_id,
        dataset_version="synthetic-v2",
        dataset_fingerprint="f" * 64,
        corpus_fingerprint="c" * 64,
        retriever_config_id="lexical-baseline-v1",
        top_k_values=[1, 3, 5, 10],
        case_metrics=[],
        results=[],
        aggregate_recall_at_k={1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0},
        aggregate_mrr=0.0,
        total_retrieval_latency_ms=1.0,
        generated_at=datetime.now(timezone.utc),
    )


class TestRetrievalExperimentPath:
    def test_naming_convention(self, tmp_path):
        path = retrieval_experiment_path(tmp_path, "synthetic-v2", "lexical-baseline-v1", "abc-123")
        assert path == tmp_path / "retrieval_experiments" / "synthetic-v2__lexical-baseline-v1__abc-123.json"

    def test_lives_under_retrieval_experiments_not_experiments(self, tmp_path):
        """Must never collide with, or be confused for, the LLM experiment
        path (results/experiments/)."""
        path = retrieval_experiment_path(tmp_path, "synthetic-v2", "vector-fake", "run-1")
        assert "retrieval_experiments" in path.parts
        assert path.parent != tmp_path / "experiments"

    def test_experiments_dir_helper(self, tmp_path):
        assert retrieval_experiments_dir(tmp_path) == tmp_path / "retrieval_experiments"


class TestResolveRunId:
    def test_generates_a_fresh_uuid_when_no_override(self, monkeypatch):
        monkeypatch.delenv("ARL_RUN_ID", raising=False)
        run_id_a = resolve_run_id()
        run_id_b = resolve_run_id()
        assert run_id_a != run_id_b  # two fresh UUIDs, never identical by chance

    def test_uses_arl_run_id_override(self, monkeypatch):
        monkeypatch.setenv("ARL_RUN_ID", "fixed-run-id-for-test")
        assert resolve_run_id() == "fixed-run-id-for-test"

    def test_empty_override_falls_through_to_fresh_uuid(self, monkeypatch):
        monkeypatch.setenv("ARL_RUN_ID", "")
        assert resolve_run_id() != ""


class TestWriteRetrievalReportAtomically:
    def test_writes_a_valid_json_file(self, tmp_path):
        path = tmp_path / "retrieval_experiments" / "synthetic-v2__lexical-baseline-v1__run-1.json"
        write_retrieval_report_atomically(path, _report())
        assert path.exists()
        loaded = RetrievalReport.model_validate_json(path.read_text())
        assert loaded.run_id == "run-1"

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "report.json"
        write_retrieval_report_atomically(path, _report())
        assert path.exists()

    def test_no_leftover_temp_file(self, tmp_path):
        path = tmp_path / "retrieval_experiments" / "report.json"
        write_retrieval_report_atomically(path, _report())
        remaining = list(path.parent.iterdir())
        assert remaining == [path]  # only the final file, no .tmp-* sibling

    def test_overwrites_when_called_directly(self, tmp_path):
        """This function is the WRITE mechanism, not the immutability
        policy -- collision refusal is the CALLER's job (see
        scripts/run_retrieval_eval.py). Documented here so the two
        responsibilities aren't confused."""
        path = tmp_path / "report.json"
        write_retrieval_report_atomically(path, _report(run_id="first"))
        write_retrieval_report_atomically(path, _report(run_id="second"))
        assert RetrievalReport.model_validate_json(path.read_text()).run_id == "second"
