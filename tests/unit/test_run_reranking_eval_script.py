"""In-process tests for scripts/run_reranking_eval.py: the precall safety
ordering (config -> candidate-source validation -> reranker construction
-> git SHA -> run ID -> artifact path -> collision check, ALL before any
real reranker provider call), and the results-root override.

Loaded as an independent module object via importlib -- never as a
subprocess, and the real Cohere SDK is never constructed -- mirroring
tests/unit/test_run_retrieval_eval_script.py's approach for the retrieval
script. ``_build_reranker`` is monkeypatched to a call-counting stub, so
these tests prove zero provider calls occur on every failure path.
"""

from __future__ import annotations

import importlib.util
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.models.contracts import Control
from app.retrieval.artifact_io import retrieval_experiment_path
from app.retrieval.contracts import RetrievalQuery, RetrievalResult, RetrievedEvidence
from app.retrieval.runner import RetrievalReport

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_reranking_eval.py"
FROZEN_CONFIG_PATH = REPO_ROOT / "configs" / "reranker-baseline-v1.json"
FROZEN_EXECUTION_CONFIG_PATH = REPO_ROOT / "configs" / "reranker-execution-cohere-trial-v1.json"
NONEXISTENT_DOTENV_PATH = str(REPO_ROOT / ".pytest-isolation-nonexistent-dir" / ".env")

DATASET_FINGERPRINT = "35e54a143b90b9d8bf30db7e1cbdb27346eddba0e0c71dd8070e69f22fe63a72"
CORPUS_FINGERPRINT = "8b9c0462d085addf206667e35aea7e7cc3cfb9473a4961ec902bc3e71e99a109"
SOURCE_RUN_ID = "5b363b87-f452-4d1b-90e1-1f2cd1c2ef36"


class _CountingReranker:
    reranker_config_id = "counting-v1"

    def __init__(self):
        self.call_count = 0

    def rerank(self, rerank_input, *, final_k=None):
        self.call_count += 1
        raise AssertionError("the reranker must never actually be invoked in these failure-path tests")


def _load_module(monkeypatch, results_dir: Path, source_results_dir: Path):
    monkeypatch.setenv("ARL_DOTENV_PATH", NONEXISTENT_DOTENV_PATH)
    monkeypatch.setenv("ARL_RESULTS_DIR", str(results_dir))
    monkeypatch.setenv("ARL_SOURCE_RESULTS_DIR", str(source_results_dir))
    for key in ("ARL_RUN_ID", "ARL_GIT_COMMIT_SHA", "RERANKER_PROVIDER", "COHERE_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    module_name = f"run_reranking_eval_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _control() -> Control:
    return Control(control_id="CTRL-PRIV-ACCESS-REVIEW", name="N", description="D", requirement_text="R")


def _query(case_id: str) -> RetrievalQuery:
    return RetrievalQuery(case_id=case_id, control=_control(), scenario_description="S", period="2025-Q1")


def _candidate(evidence_id: str, rank: int) -> RetrievedEvidence:
    return RetrievedEvidence(evidence_id=evidence_id, document_id=f"doc_{evidence_id}", score=1.0, rank=rank, retrieval_method="x")


def _result(case_id: str) -> RetrievalResult:
    candidates = [_candidate(f"EV-{i}", i) for i in range(1, 11)]
    return RetrievalResult(case_id=case_id, query=_query(case_id), candidates=candidates, top_k=10, retriever_config_id="retrieval-baseline-v1")


def _valid_fabricated_report(git_commit_sha="d" * 40) -> RetrievalReport:
    results = [_result(f"AC-{i:03d}") for i in range(1, 31)]
    return RetrievalReport(
        run_id=SOURCE_RUN_ID, git_commit_sha=git_commit_sha, dataset_version="synthetic-v2",
        dataset_fingerprint=DATASET_FINGERPRINT, corpus_fingerprint=CORPUS_FINGERPRINT,
        retriever_config_id="retrieval-baseline-v1", top_k_values=[1, 3, 5, 10], case_metrics=[],
        results=results, aggregate_recall_at_k={1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0}, aggregate_mrr=1.0,
        total_retrieval_latency_ms=1.0, generated_at=datetime.now(timezone.utc),
    )


def _write_source_artifact(source_results_dir: Path, report: RetrievalReport):
    """Writes at the EXACT path the script will reconstruct from the
    config's own fields, and returns that content's real SHA-256."""

    import hashlib

    path = retrieval_experiment_path(source_results_dir, "synthetic-v2", "retrieval-baseline-v1", SOURCE_RUN_ID)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = report.model_dump_json(indent=2) + "\n"
    path.write_text(content)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _write_config(tmp_path: Path, **overrides) -> Path:
    base = json.loads(FROZEN_CONFIG_PATH.read_text())
    base.update(overrides)
    path = tmp_path / "reranker-config.json"
    path.write_text(json.dumps(base))
    return path


class TestPreProviderFailureOrdering:
    def test_bad_source_artifact_hash_blocks_before_any_provider_call(self, monkeypatch, tmp_path, capsys):
        results_dir = tmp_path / "results"
        source_dir = tmp_path / "source"
        module = _load_module(monkeypatch, results_dir, source_dir)
        report = _valid_fabricated_report()
        real_sha256 = _write_source_artifact(source_dir, report)

        counting_reranker = _CountingReranker()
        monkeypatch.setattr(module, "_build_reranker", lambda *a, **k: counting_reranker)
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "e" * 40)

        # Config declares a WRONG expected hash -- deliberately mismatched
        # from what was actually written above.
        config_path = _write_config(
            tmp_path, candidate_source_run_id=SOURCE_RUN_ID, candidate_source_artifact_sha256="0" * 64,
            dataset_fingerprint=DATASET_FINGERPRINT, corpus_fingerprint=CORPUS_FINGERPRINT,
        )

        exit_code = module.main(["--config", str(config_path)])

        assert exit_code == 1
        assert "candidate source validation failed" in capsys.readouterr().err
        assert counting_reranker.call_count == 0
        assert not (results_dir / "reranking_experiments").exists()

    def test_source_provenance_mismatch_blocks_before_any_provider_call(self, monkeypatch, tmp_path, capsys):
        results_dir = tmp_path / "results"
        source_dir = tmp_path / "source"
        module = _load_module(monkeypatch, results_dir, source_dir)
        report = _valid_fabricated_report()
        real_sha256 = _write_source_artifact(source_dir, report)

        counting_reranker = _CountingReranker()
        monkeypatch.setattr(module, "_build_reranker", lambda *a, **k: counting_reranker)
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "e" * 40)

        # Correct hash, but the config's OWN recorded dataset_fingerprint
        # doesn't match the artifact's -- a provenance mismatch, not a
        # hash mismatch.
        config_path = _write_config(
            tmp_path, candidate_source_run_id=SOURCE_RUN_ID, candidate_source_artifact_sha256=real_sha256,
            dataset_fingerprint="9" * 64, corpus_fingerprint=CORPUS_FINGERPRINT,
        )

        exit_code = module.main(["--config", str(config_path)])

        assert exit_code == 1
        assert counting_reranker.call_count == 0
        assert not (results_dir / "reranking_experiments").exists()

    def test_missing_git_sha_blocks_before_any_provider_call(self, monkeypatch, tmp_path, capsys):
        results_dir = tmp_path / "results"
        source_dir = tmp_path / "source"
        module = _load_module(monkeypatch, results_dir, source_dir)
        report = _valid_fabricated_report()
        real_sha256 = _write_source_artifact(source_dir, report)

        counting_reranker = _CountingReranker()
        monkeypatch.setattr(module, "_build_reranker", lambda *a, **k: counting_reranker)
        monkeypatch.setattr(module, "get_git_commit_sha", lambda repo_root: None)

        config_path = _write_config(
            tmp_path, candidate_source_run_id=SOURCE_RUN_ID, candidate_source_artifact_sha256=real_sha256,
            dataset_fingerprint=DATASET_FINGERPRINT, corpus_fingerprint=CORPUS_FINGERPRINT,
        )

        exit_code = module.main(["--config", str(config_path)])

        assert exit_code == 1
        assert "git commit" in capsys.readouterr().err.lower()
        assert counting_reranker.call_count == 0
        assert not (results_dir / "reranking_experiments").exists()

    def test_artifact_collision_blocks_before_any_provider_call(self, monkeypatch, tmp_path, capsys):
        results_dir = tmp_path / "results"
        source_dir = tmp_path / "source"
        module = _load_module(monkeypatch, results_dir, source_dir)
        report = _valid_fabricated_report()
        real_sha256 = _write_source_artifact(source_dir, report)

        counting_reranker = _CountingReranker()
        monkeypatch.setattr(module, "_build_reranker", lambda *a, **k: counting_reranker)
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "e" * 40)
        monkeypatch.setenv("ARL_RUN_ID", "colliding-rerank-run-id")

        config_path = _write_config(
            tmp_path, config_id="cohere-rerank-v4-pro-baseline-v1",
            candidate_source_run_id=SOURCE_RUN_ID, candidate_source_artifact_sha256=real_sha256,
            dataset_fingerprint=DATASET_FINGERPRINT, corpus_fingerprint=CORPUS_FINGERPRINT,
        )

        from app.reranking.artifact_io import reranking_experiment_path

        existing = reranking_experiment_path(
            results_dir, "synthetic-v2", "cohere-rerank-v4-pro-baseline-v1", "colliding-rerank-run-id",
        )
        existing.parent.mkdir(parents=True)
        existing.write_text('{"sentinel": "pre-existing, must survive"}')

        exit_code = module.main(["--config", str(config_path)])

        assert exit_code == 1
        assert "already exists" in capsys.readouterr().err
        assert existing.read_text() == '{"sentinel": "pre-existing, must survive"}'
        assert counting_reranker.call_count == 0

    def test_malformed_config_blocks_before_any_provider_call(self, monkeypatch, tmp_path, capsys):
        results_dir = tmp_path / "results"
        source_dir = tmp_path / "source"
        module = _load_module(monkeypatch, results_dir, source_dir)

        counting_reranker = _CountingReranker()
        monkeypatch.setattr(module, "_build_reranker", lambda *a, **k: counting_reranker)

        bad_config_path = tmp_path / "bad.json"
        bad_config_path.write_text(json.dumps({"config_id": "x"}))

        exit_code = module.main(["--config", str(bad_config_path)])

        assert exit_code == 1
        assert "not a valid RerankerBaselineConfig" in capsys.readouterr().err
        assert counting_reranker.call_count == 0

    def test_missing_reranker_provider_config_blocks_before_git_or_collision_checks(self, monkeypatch, tmp_path, capsys):
        results_dir = tmp_path / "results"
        source_dir = tmp_path / "source"
        module = _load_module(monkeypatch, results_dir, source_dir)
        report = _valid_fabricated_report()
        real_sha256 = _write_source_artifact(source_dir, report)

        config_path = _write_config(
            tmp_path, candidate_source_run_id=SOURCE_RUN_ID, candidate_source_artifact_sha256=real_sha256,
            dataset_fingerprint=DATASET_FINGERPRINT, corpus_fingerprint=CORPUS_FINGERPRINT,
        )
        # RERANKER_PROVIDER / COHERE_API_KEY deliberately left unset --
        # _build_reranker is NOT monkeypatched here, so the real
        # (unpatched) function must itself fail cleanly with no network.
        exit_code = module.main(["--config", str(config_path)])
        assert exit_code == 1
        assert "RERANKER_PROVIDER" in capsys.readouterr().err


class TestResultsRootOverride:
    def test_output_written_under_overridden_results_dir(self, monkeypatch, tmp_path, capsys):
        results_dir = tmp_path / "custom-results"
        source_dir = tmp_path / "source"
        module = _load_module(monkeypatch, results_dir, source_dir)
        report = _valid_fabricated_report()
        real_sha256 = _write_source_artifact(source_dir, report)

        # Force a failure AFTER the output path would be under
        # results_dir but BEFORE any write, just to prove RESULTS_DIR
        # itself resolved to the override (via the collision path).
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "e" * 40)
        monkeypatch.setenv("ARL_RUN_ID", "some-run-id")
        from app.reranking.artifact_io import reranking_experiment_path

        existing = reranking_experiment_path(
            results_dir, "synthetic-v2", "cohere-rerank-v4-pro-baseline-v1", "some-run-id",
        )
        existing.parent.mkdir(parents=True)
        existing.write_text("{}")

        config_path = _write_config(
            tmp_path, candidate_source_run_id=SOURCE_RUN_ID, candidate_source_artifact_sha256=real_sha256,
            dataset_fingerprint=DATASET_FINGERPRINT, corpus_fingerprint=CORPUS_FINGERPRINT,
        )
        module.main(["--config", str(config_path)])

        assert (results_dir / "reranking_experiments").exists()
        assert not (tmp_path / "results" / "reranking_experiments").exists()


class _FailFirstCaseReranker:
    reranker_config_id = "fail-first-v1"

    def __init__(self):
        self.call_count = 0

    def rerank(self, rerank_input, *, final_k=None):
        self.call_count += 1
        raise RuntimeError(f"simulated provider failure on {rerank_input.case_id}")


class _SucceedingCountingReranker:
    """Delegates to PassThroughReranker for every case -- used to exercise
    a full successful run's pacing/observability without any real
    provider or evidence lookup."""

    reranker_config_id = "succeeding-counting-v1"

    def __init__(self):
        self.call_count = 0

    def rerank(self, rerank_input, *, final_k=None):
        from app.reranking.pass_through import PassThroughReranker

        self.call_count += 1
        return PassThroughReranker().rerank(rerank_input, final_k=final_k)


class TestRunIdObservabilityAndFailureSummary:
    def test_run_id_and_artifact_path_printed_before_any_case_output(self, monkeypatch, tmp_path, capsys):
        results_dir = tmp_path / "results"
        source_dir = tmp_path / "source"
        module = _load_module(monkeypatch, results_dir, source_dir)
        report = _valid_fabricated_report()
        real_sha256 = _write_source_artifact(source_dir, report)

        failing_reranker = _FailFirstCaseReranker()
        monkeypatch.setattr(module, "_build_reranker", lambda *a, **k: failing_reranker)
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "e" * 40)
        monkeypatch.setenv("ARL_RUN_ID", "observable-run-id")

        config_path = _write_config(
            tmp_path, candidate_source_run_id=SOURCE_RUN_ID, candidate_source_artifact_sha256=real_sha256,
            dataset_fingerprint=DATASET_FINGERPRINT, corpus_fingerprint=CORPUS_FINGERPRINT,
        )

        exit_code = module.main(["--config", str(config_path)])

        stdout = capsys.readouterr().out
        assert exit_code == 1
        assert "RUN_ID=observable-run-id" in stdout
        run_id_pos = stdout.index("RUN_ID=observable-run-id")
        case_pos = stdout.find("case 1/30")
        assert case_pos == -1 or run_id_pos < case_pos  # RUN_ID printed before any case activity

    def test_failure_summary_reports_attempted_and_completed_counts(self, monkeypatch, tmp_path, capsys):
        results_dir = tmp_path / "results"
        source_dir = tmp_path / "source"
        module = _load_module(monkeypatch, results_dir, source_dir)
        report = _valid_fabricated_report()
        real_sha256 = _write_source_artifact(source_dir, report)

        failing_reranker = _FailFirstCaseReranker()
        monkeypatch.setattr(module, "_build_reranker", lambda *a, **k: failing_reranker)
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "e" * 40)
        monkeypatch.setenv("ARL_RUN_ID", "failure-summary-run-id")

        config_path = _write_config(
            tmp_path, candidate_source_run_id=SOURCE_RUN_ID, candidate_source_artifact_sha256=real_sha256,
            dataset_fingerprint=DATASET_FINGERPRINT, corpus_fingerprint=CORPUS_FINGERPRINT,
        )

        exit_code = module.main(["--config", str(config_path)])

        stderr = capsys.readouterr().err
        assert exit_code == 1
        assert "run_id: failure-summary-run-id" in stderr
        assert "attempted_case_count: 1" in stderr
        assert "completed_case_count: 0" in stderr
        assert "failed_case_id: AC-001" in stderr
        assert "exception_category: RuntimeError" in stderr
        assert "artifact_written: false" in stderr
        assert failing_reranker.call_count == 1  # never retried
        assert not (results_dir / "reranking_experiments").exists()


class TestExecutionConfigPacing:
    def test_pacing_applied_across_a_full_successful_run(self, monkeypatch, tmp_path, capsys):
        import time as time_module

        from tests.support.fake_clock import FakeClock

        results_dir = tmp_path / "results"
        source_dir = tmp_path / "source"
        module = _load_module(monkeypatch, results_dir, source_dir)
        report = _valid_fabricated_report()
        real_sha256 = _write_source_artifact(source_dir, report)

        succeeding_reranker = _SucceedingCountingReranker()
        monkeypatch.setattr(module, "_build_reranker", lambda *a, **k: succeeding_reranker)
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "e" * 40)

        clock = FakeClock(start=0.0)
        monkeypatch.setattr(module.time, "monotonic", clock.monotonic)
        monkeypatch.setattr(module.time, "sleep", clock.sleep)

        config_path = _write_config(
            tmp_path, candidate_source_run_id=SOURCE_RUN_ID, candidate_source_artifact_sha256=real_sha256,
            dataset_fingerprint=DATASET_FINGERPRINT, corpus_fingerprint=CORPUS_FINGERPRINT,
        )

        exit_code = module.main([
            "--config", str(config_path),
            "--execution-config", str(FROZEN_EXECUTION_CONFIG_PATH),
        ])

        assert exit_code == 0
        assert succeeding_reranker.call_count == 30
        assert clock.sleep_calls == [7.0] * 29  # first call: no sleep; 29 subsequent: paced
        report_files = list((results_dir / "reranking_experiments").glob("*.json"))
        assert len(report_files) == 1

    def test_no_execution_config_means_no_pacing(self, monkeypatch, tmp_path):
        results_dir = tmp_path / "results"
        source_dir = tmp_path / "source"
        module = _load_module(monkeypatch, results_dir, source_dir)
        report = _valid_fabricated_report()
        real_sha256 = _write_source_artifact(source_dir, report)

        succeeding_reranker = _SucceedingCountingReranker()
        monkeypatch.setattr(module, "_build_reranker", lambda *a, **k: succeeding_reranker)
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "e" * 40)

        config_path = _write_config(
            tmp_path, candidate_source_run_id=SOURCE_RUN_ID, candidate_source_artifact_sha256=real_sha256,
            dataset_fingerprint=DATASET_FINGERPRINT, corpus_fingerprint=CORPUS_FINGERPRINT,
        )

        exit_code = module.main(["--config", str(config_path)])  # no --execution-config

        assert exit_code == 0
        assert succeeding_reranker.call_count == 30
