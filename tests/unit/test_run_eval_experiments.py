"""In-process tests for scripts/run_eval.py's real-LLM-run infrastructure:
the immutable, run-ID-qualified experiment output path under
results/experiments/ (Phase 6 correction, item D) and the mandatory
git-commit-SHA provenance capture (item E).

scripts/run_eval.py is loaded as an independent module object via
importlib -- never as a subprocess, and the real OpenAI SDK is never
constructed -- so these tests can monkeypatch `_build_llm_agent` to return
an LLMAgent wired to FakeLLMProvider (the same zero-network test double
already used by tests/unit/test_llm_agent.py and
tests/unit/test_llm_provenance.py) and exercise the REAL run_evaluation()
pipeline, against the real synthetic-v1 dataset, with zero risk of any
network call and no OpenAI credentials of any kind, real or fake.
"""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

import pytest

from app.agent.llm_agent import LLMAgent, LLMFindingPayload, build_llm_config
from app.models.contracts import Finding, ModelProvider, RunReport
from app.providers.base import StructuredCompletion
from tests.support.fake_llm_provider import FakeLLMProvider

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_eval.py"
NONEXISTENT_DOTENV_PATH = str(REPO_ROOT / ".pytest-isolation-nonexistent-dir" / ".env")


def _load_run_eval_module(monkeypatch, results_dir: Path):
    """Load scripts/run_eval.py as a fresh module object, isolated from
    this repository's real .env and real results/ directory (the module
    reads ARL_DOTENV_PATH / ARL_RESULTS_DIR at import time). A unique
    module name per call means successive tests never share cached state.
    """

    monkeypatch.setenv("ARL_DOTENV_PATH", NONEXISTENT_DOTENV_PATH)
    monkeypatch.setenv("ARL_RESULTS_DIR", str(results_dir))
    for key in (
        "AGENT_MODE", "MODEL_PROVIDER", "MODEL_NAME",
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ARL_RUN_ID", "ARL_GIT_COMMIT_SHA",
    ):
        monkeypatch.delenv(key, raising=False)

    module_name = f"run_eval_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_llm_agent() -> LLMAgent:
    payload = LLMFindingPayload(
        finding=Finding.PASS,
        confidence=0.9,
        reasoning_summary="Fake, zero-network response for CLI infrastructure tests.",
        evidence=[],
        missing_information=[],
        abstain=False,
    )
    completion = StructuredCompletion(
        parsed=payload, raw_text=None, usage=None,
        provider=ModelProvider.OPENAI, model_name="fake-model-v1",
    )
    return LLMAgent(provider=FakeLLMProvider(completion))


def _patch_build_llm_agent(monkeypatch, module) -> None:
    monkeypatch.setattr(
        module, "_build_llm_agent",
        lambda: (_fake_llm_agent(), ModelProvider.OPENAI, "fake-model-v1", build_llm_config()),
    )


class TestExperimentResultsPathFormat:
    def test_format(self, monkeypatch, tmp_path):
        module = _load_run_eval_module(monkeypatch, tmp_path)
        path = module._experiment_results_path("synthetic-v1", "llm-baseline-v1", "abc-123")
        assert path == tmp_path / "experiments" / "synthetic-v1__llm-baseline-v1__abc-123.json"


class TestSuccessfulRunWritesImmutableExperimentArtifact:
    def test_writes_under_experiments_dir_with_run_id_in_filename(self, monkeypatch, tmp_path):
        module = _load_run_eval_module(monkeypatch, tmp_path)
        _patch_build_llm_agent(monkeypatch, module)
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "fake-sha-for-this-test")

        exit_code = module.main(["--agent", "llm"])
        assert exit_code == 0

        experiment_files = list((tmp_path / "experiments").glob("*.json"))
        assert len(experiment_files) == 1
        assert experiment_files[0].name.startswith("synthetic-v1__llm-baseline-v1__")

        report = RunReport.model_validate_json(experiment_files[0].read_text())
        assert report.dataset_version == "synthetic-v1"
        assert report.git_commit_sha == "fake-sha-for-this-test"
        assert len(report.results) == 10
        assert experiment_files[0].name == f"synthetic-v1__llm-baseline-v1__{report.run_id}.json"

    def test_legacy_fixed_llm_baseline_path_is_never_written(self, monkeypatch, tmp_path):
        module = _load_run_eval_module(monkeypatch, tmp_path)
        _patch_build_llm_agent(monkeypatch, module)
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "fake-sha-for-this-test")

        exit_code = module.main(["--agent", "llm"])
        assert exit_code == 0
        assert not (tmp_path / "llm-baseline.json").exists()

    def test_arl_run_id_override_controls_the_exact_filename(self, monkeypatch, tmp_path):
        module = _load_run_eval_module(monkeypatch, tmp_path)
        _patch_build_llm_agent(monkeypatch, module)
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "fake-sha-for-this-test")
        monkeypatch.setenv("ARL_RUN_ID", "fixed-run-id-for-this-test")

        exit_code = module.main(["--agent", "llm"])
        assert exit_code == 0

        expected = tmp_path / "experiments" / "synthetic-v1__llm-baseline-v1__fixed-run-id-for-this-test.json"
        assert expected.exists()
        report = RunReport.model_validate_json(expected.read_text())
        assert report.run_id == "fixed-run-id-for-this-test"


class TestExperimentImmutability:
    def test_refuses_to_overwrite_existing_experiment_artifact(self, monkeypatch, tmp_path, capsys):
        module = _load_run_eval_module(monkeypatch, tmp_path)
        _patch_build_llm_agent(monkeypatch, module)
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "fake-sha-for-this-test")
        monkeypatch.setenv("ARL_RUN_ID", "colliding-run-id")

        existing = tmp_path / "experiments" / "synthetic-v1__llm-baseline-v1__colliding-run-id.json"
        existing.parent.mkdir(parents=True)
        existing.write_text('{"sentinel": "pre-existing, must survive"}')

        exit_code = module.main(["--agent", "llm"])
        assert exit_code == 1
        assert existing.read_text() == '{"sentinel": "pre-existing, must survive"}'
        assert "already exists" in capsys.readouterr().err

    def test_force_overwrite_does_not_bypass_the_refusal(self, monkeypatch, tmp_path, capsys):
        module = _load_run_eval_module(monkeypatch, tmp_path)
        _patch_build_llm_agent(monkeypatch, module)
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "fake-sha-for-this-test")
        monkeypatch.setenv("ARL_RUN_ID", "colliding-run-id")

        existing = tmp_path / "experiments" / "synthetic-v1__llm-baseline-v1__colliding-run-id.json"
        existing.parent.mkdir(parents=True)
        existing.write_text('{"sentinel": "must survive even with --force-overwrite"}')

        exit_code = module.main(["--agent", "llm", "--force-overwrite"])
        assert exit_code == 1
        assert existing.read_text() == '{"sentinel": "must survive even with --force-overwrite"}'


class TestGitCommitShaProvenance:
    def test_llm_run_fails_before_any_write_when_sha_cannot_be_determined(self, monkeypatch, tmp_path, capsys):
        module = _load_run_eval_module(monkeypatch, tmp_path)
        _patch_build_llm_agent(monkeypatch, module)
        monkeypatch.setattr(module, "get_git_commit_sha", lambda repo_root: None)

        exit_code = module.main(["--agent", "llm"])
        assert exit_code == 1
        assert not (tmp_path / "experiments").exists()
        assert "git commit" in capsys.readouterr().err.lower()

    def test_llm_run_succeeds_and_stores_the_captured_sha(self, monkeypatch, tmp_path):
        module = _load_run_eval_module(monkeypatch, tmp_path)
        _patch_build_llm_agent(monkeypatch, module)
        monkeypatch.setattr(module, "get_git_commit_sha", lambda repo_root: "captured-sha-value")

        exit_code = module.main(["--agent", "llm"])
        assert exit_code == 0

        experiment_files = list((tmp_path / "experiments").glob("*.json"))
        report = RunReport.model_validate_json(experiment_files[0].read_text())
        assert report.git_commit_sha == "captured-sha-value"

    def test_git_sha_helper_is_called_exactly_once_per_run_not_per_case(self, monkeypatch, tmp_path):
        module = _load_run_eval_module(monkeypatch, tmp_path)
        _patch_build_llm_agent(monkeypatch, module)
        call_count = {"n": 0}

        def _counting_get_sha(repo_root):
            call_count["n"] += 1
            return "captured-sha-value"

        monkeypatch.setattr(module, "get_git_commit_sha", _counting_get_sha)
        exit_code = module.main(["--agent", "llm"])
        assert exit_code == 0
        assert call_count["n"] == 1  # not once per one of the 10 synthetic-v1 cases

    def test_deterministic_mode_captures_sha_best_effort_and_is_never_fatal(self, monkeypatch, tmp_path):
        """Deterministic runs are free/reproducible -- a missing git SHA
        must not block them, unlike a real LLM run."""
        module = _load_run_eval_module(monkeypatch, tmp_path)
        monkeypatch.setattr(module, "get_git_commit_sha", lambda repo_root: None)

        exit_code = module.main([])
        assert exit_code == 0
        report = RunReport.model_validate_json((tmp_path / "baseline.json").read_text())
        assert report.git_commit_sha is None
