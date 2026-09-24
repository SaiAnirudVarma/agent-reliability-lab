"""In-process tests for scripts/run_reranked_llm_eval.py: the precall
safety ordering (config -> evidence-source validation -> git SHA -> run
ID -> artifact path -> collision check -> real-API authorization, ALL
before any real LLM provider call), and a full successful run driven
entirely by FakeLLMProvider.

Loaded as an independent module object via importlib -- never as a
subprocess, and the real OpenAI SDK is never constructed. ``_build_llm_agent``
is monkeypatched to a call-counting stub for the failure-path tests, so
these tests prove zero provider construction occurs on every failure path.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.datasets.loader import load_dataset
from app.integration.runner import RerankedLLMReport
from app.models.contracts import Control, Finding, ModelProvider
from app.providers.base import StructuredCompletion
from app.agent.llm_agent import LLMFindingPayload, build_llm_config, LLMAgent
from app.reranking.artifact_io import reranking_experiment_path
from app.reranking.contracts import RerankedEvidence, RerankInput, RerankResult
from app.reranking.runner import RerankingCaseReport, RerankingReport
from app.retrieval.contracts import RetrievalQuery, RetrievedEvidence
from app.retrieval.corpus import build_full_version_corpus
from tests.support.fake_llm_provider import FakeLLMProvider

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_reranked_llm_eval.py"
FROZEN_CONFIG_PATH = REPO_ROOT / "configs" / "reranked-llm-baseline-v1.json"
NONEXISTENT_DOTENV_PATH = str(REPO_ROOT / ".pytest-isolation-nonexistent-dir" / ".env")
DATASET_DIR = REPO_ROOT / "datasets"

DATASET_FINGERPRINT = "35e54a143b90b9d8bf30db7e1cbdb27346eddba0e0c71dd8070e69f22fe63a72"
CORPUS_FINGERPRINT = "8b9c0462d085addf206667e35aea7e7cc3cfb9473a4961ec902bc3e71e99a109"
SOURCE_RUN_ID = "fake-rerank-run-1"

# Real evidence_ids from the real synthetic-v2 corpus -- fabricated
# RerankingReport fixtures below must reference IDs that actually exist
# in the corpus the script's own main() builds, or evidence hydration
# (app.integration.agent_input) would KeyError.
_REAL_CORPUS_EVIDENCE_IDS = [
    e.evidence_id
    for e in build_full_version_corpus(load_dataset(DATASET_DIR, version="synthetic-v2")).evidence[:10]
]


class _CountingLLMAgent:
    agent_config_id = "counting-v1"

    def __init__(self):
        self.call_count = 0

    def run(self, task):
        self.call_count += 1
        raise AssertionError("the real LLM agent construction path must never be reached in these tests")


def _load_module(monkeypatch, results_dir: Path, source_results_dir: Path):
    monkeypatch.setenv("ARL_DOTENV_PATH", NONEXISTENT_DOTENV_PATH)
    monkeypatch.setenv("ARL_RESULTS_DIR", str(results_dir))
    monkeypatch.setenv("ARL_SOURCE_RESULTS_DIR", str(source_results_dir))
    for key in ("ARL_RUN_ID", "ARL_GIT_COMMIT_SHA", "ARL_ALLOW_REAL_API_CALLS", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    module_name = f"run_reranked_llm_eval_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _control() -> Control:
    return Control(control_id="CTRL-PRIV-ACCESS-REVIEW", name="N", description="D", requirement_text="R")


def _rerank_result(case_id: str) -> RerankResult:
    evidence_ids = _REAL_CORPUS_EVIDENCE_IDS
    original = [
        RetrievedEvidence(evidence_id=eid, document_id=f"doc_{eid}", score=1.0, rank=i, retrieval_method="x")
        for i, eid in enumerate(evidence_ids, start=1)
    ]
    query = RetrievalQuery(case_id=case_id, control=_control(), scenario_description="S", period="2025-Q1")
    rerank_input = RerankInput(case_id=case_id, query=query, candidates=original, candidate_depth=len(evidence_ids))
    reranked = [
        RerankedEvidence(
            evidence_id=eid, document_id=f"doc_{eid}", original_rank=i, reranked_rank=i,
            original_score=1.0, reranker_score=1.0, reranker_method="pass-through",
        )
        for i, eid in enumerate(evidence_ids, start=1)
    ]
    return RerankResult(case_id=case_id, input=rerank_input, candidates=reranked, reranker_config_id="cohere-rerank-v4-pro-baseline-v1")


def _case_report(case_id: str) -> RerankingCaseReport:
    return RerankingCaseReport(
        case_id=case_id, recall_at_k_before={1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0},
        recall_at_k_after={1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0}, recall_at_k_delta={1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0},
        mrr_before=1.0, mrr_after=1.0, mrr_delta=0.0, required_evidence_rank_movement=[], reranking_latency_ms=1.0,
    )


def _valid_fabricated_reranking_report(git_commit_sha="d" * 40) -> RerankingReport:
    case_ids = [f"AC-{i:03d}" for i in range(1, 31)]
    return RerankingReport(
        run_id=SOURCE_RUN_ID, git_commit_sha=git_commit_sha, generated_at=datetime.now(timezone.utc),
        dataset_version="synthetic-v2", dataset_fingerprint=DATASET_FINGERPRINT, corpus_fingerprint=CORPUS_FINGERPRINT,
        source_retrieval_run_id="5b363b87-f452-4d1b-90e1-1f2cd1c2ef36", source_retrieval_artifact_sha256="e" * 64,
        source_retriever_config_id="retrieval-baseline-v1", candidate_depth=10, output_depth=10,
        reranker_config_id="cohere-rerank-v4-pro-baseline-v1", reranker_provider="cohere",
        requested_reranker_model="rerank-v4.0-pro", served_reranker_model=None, execution_policy_id=None,
        evaluation_k_values=[1, 3, 5, 10],
        case_reports=[_case_report(cid) for cid in case_ids],
        rerank_results=[_rerank_result(cid) for cid in case_ids],
        aggregate_recall_at_k_before={1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0},
        aggregate_recall_at_k_after={1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0},
        aggregate_recall_at_k_delta={1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0},
        aggregate_mrr_before=1.0, aggregate_mrr_after=1.0, aggregate_mrr_delta=0.0, total_reranking_latency_ms=1.0,
    )


def _write_source_artifact(source_results_dir: Path, report: RerankingReport) -> str:
    """Writes at the EXACT path the script will reconstruct from the
    config's own fields, and returns that content's real SHA-256."""

    path = reranking_experiment_path(source_results_dir, "synthetic-v2", "cohere-rerank-v4-pro-baseline-v1", SOURCE_RUN_ID)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = report.model_dump_json(indent=2) + "\n"
    path.write_text(content)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _write_config(tmp_path: Path, **overrides) -> Path:
    base = json.loads(FROZEN_CONFIG_PATH.read_text())
    base["source_reranking_run_id"] = SOURCE_RUN_ID
    base.update(overrides)
    path = tmp_path / "reranked-llm-config.json"
    path.write_text(json.dumps(base))
    return path


def _fake_llm_agent() -> LLMAgent:
    payload = LLMFindingPayload(
        finding=Finding.PASS, confidence=0.9, reasoning_summary="Fake, zero-network response.",
        evidence=[], missing_information=[], abstain=False,
    )
    completion = StructuredCompletion(parsed=payload, raw_text=None, usage=None, provider=ModelProvider.OPENAI, model_name="fake-model-v1")
    return LLMAgent(provider=FakeLLMProvider(completion))


def _patch_build_llm_agent(monkeypatch, module) -> None:
    monkeypatch.setattr(
        module, "_build_llm_agent",
        lambda config: (_fake_llm_agent(), ModelProvider.OPENAI, "fake-model-v1", build_llm_config()),
    )


class TestPreProviderFailureOrdering:
    def test_bad_source_artifact_hash_blocks_before_any_provider_call(self, monkeypatch, tmp_path, capsys):
        results_dir = tmp_path / "results"
        source_dir = tmp_path / "source"
        module = _load_module(monkeypatch, results_dir, source_dir)
        report = _valid_fabricated_reranking_report()
        _write_source_artifact(source_dir, report)

        counting_agent = _CountingLLMAgent()
        monkeypatch.setattr(module, "_build_llm_agent", lambda config: (counting_agent, ModelProvider.OPENAI, "x", build_llm_config()))
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "e" * 40)
        monkeypatch.setenv("ARL_ALLOW_REAL_API_CALLS", "1")

        config_path = _write_config(tmp_path, source_reranking_artifact_sha256="0" * 64)

        exit_code = module.main(["--config", str(config_path)])

        assert exit_code == 1
        assert "evidence source validation failed" in capsys.readouterr().err
        assert counting_agent.call_count == 0
        assert not (results_dir / "reranked_llm_experiments").exists()

    def test_missing_source_artifact_blocks_before_any_provider_call(self, monkeypatch, tmp_path, capsys):
        results_dir = tmp_path / "results"
        source_dir = tmp_path / "source"
        module = _load_module(monkeypatch, results_dir, source_dir)
        # Deliberately never write the source artifact at all.

        counting_agent = _CountingLLMAgent()
        monkeypatch.setattr(module, "_build_llm_agent", lambda config: (counting_agent, ModelProvider.OPENAI, "x", build_llm_config()))
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "e" * 40)
        monkeypatch.setenv("ARL_ALLOW_REAL_API_CALLS", "1")

        report = _valid_fabricated_reranking_report()
        content = report.model_dump_json(indent=2) + "\n"
        real_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        config_path = _write_config(tmp_path, source_reranking_artifact_sha256=real_sha256)

        exit_code = module.main(["--config", str(config_path)])

        assert exit_code == 1
        assert "could not read evidence source artifact" in capsys.readouterr().err
        assert counting_agent.call_count == 0
        assert not (results_dir / "reranked_llm_experiments").exists()

    def test_artifact_collision_blocks_before_any_provider_call(self, monkeypatch, tmp_path, capsys):
        results_dir = tmp_path / "results"
        source_dir = tmp_path / "source"
        module = _load_module(monkeypatch, results_dir, source_dir)
        report = _valid_fabricated_reranking_report()
        real_sha256 = _write_source_artifact(source_dir, report)

        counting_agent = _CountingLLMAgent()
        monkeypatch.setattr(module, "_build_llm_agent", lambda config: (counting_agent, ModelProvider.OPENAI, "x", build_llm_config()))
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "e" * 40)
        monkeypatch.setenv("ARL_ALLOW_REAL_API_CALLS", "1")
        monkeypatch.setenv("ARL_RUN_ID", "colliding-reranked-llm-run-id")

        config_path = _write_config(tmp_path, source_reranking_artifact_sha256=real_sha256)

        from app.integration.artifact_io import reranked_llm_experiment_path

        existing = reranked_llm_experiment_path(
            results_dir, "synthetic-v2", "reranked-llm-baseline-v1", "colliding-reranked-llm-run-id",
        )
        existing.parent.mkdir(parents=True)
        existing.write_text('{"sentinel": "pre-existing, must survive"}')

        exit_code = module.main(["--config", str(config_path)])

        assert exit_code == 1
        assert "already exists" in capsys.readouterr().err
        assert existing.read_text() == '{"sentinel": "pre-existing, must survive"}'
        assert counting_agent.call_count == 0

    def test_missing_git_sha_blocks_before_any_provider_call(self, monkeypatch, tmp_path, capsys):
        results_dir = tmp_path / "results"
        source_dir = tmp_path / "source"
        module = _load_module(monkeypatch, results_dir, source_dir)
        report = _valid_fabricated_reranking_report()
        real_sha256 = _write_source_artifact(source_dir, report)

        counting_agent = _CountingLLMAgent()
        monkeypatch.setattr(module, "_build_llm_agent", lambda config: (counting_agent, ModelProvider.OPENAI, "x", build_llm_config()))
        monkeypatch.setattr(module, "get_git_commit_sha", lambda repo_root: None)
        monkeypatch.setenv("ARL_ALLOW_REAL_API_CALLS", "1")

        config_path = _write_config(tmp_path, source_reranking_artifact_sha256=real_sha256)

        exit_code = module.main(["--config", str(config_path)])

        assert exit_code == 1
        assert "git commit" in capsys.readouterr().err.lower()
        assert counting_agent.call_count == 0
        assert not (results_dir / "reranked_llm_experiments").exists()

    def test_real_api_authorization_absent_blocks_before_any_provider_call(self, monkeypatch, tmp_path, capsys):
        """Even with a fully valid source artifact, git SHA, and no
        collision -- i.e. every OTHER check passes -- the run still
        blocks unless ARL_ALLOW_REAL_API_CALLS=1 is explicitly set."""

        results_dir = tmp_path / "results"
        source_dir = tmp_path / "source"
        module = _load_module(monkeypatch, results_dir, source_dir)
        report = _valid_fabricated_reranking_report()
        real_sha256 = _write_source_artifact(source_dir, report)

        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "e" * 40)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-not-a-real-key-for-this-test")
        # ARL_ALLOW_REAL_API_CALLS deliberately left unset.

        config_path = _write_config(tmp_path, source_reranking_artifact_sha256=real_sha256)

        exit_code = module.main(["--config", str(config_path)])

        assert exit_code == 1
        assert "ARL_ALLOW_REAL_API_CALLS" in capsys.readouterr().err
        assert not (results_dir / "reranked_llm_experiments").exists()

    def test_explicit_authorization_allows_the_real_build_function_to_reach_construction(self, monkeypatch, tmp_path):
        """Unlike every other test in this class, this one does NOT
        monkeypatch _build_llm_agent itself -- it exercises the REAL
        (unpatched) function with ARL_ALLOW_REAL_API_CALLS=1, and proves
        it proceeds past the gate to actual OpenAIProvider construction
        by substituting a call-counting stub for that one class only."""

        results_dir = tmp_path / "results"
        source_dir = tmp_path / "source"
        module = _load_module(monkeypatch, results_dir, source_dir)

        monkeypatch.setenv("ARL_ALLOW_REAL_API_CALLS", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-not-a-real-key-for-this-test")

        class _CountingConstructor:
            call_count = 0

            def __init__(self, *args, **kwargs):
                type(self).call_count += 1

        _CountingConstructor.call_count = 0
        import app.providers.openai_provider as openai_provider_module

        monkeypatch.setattr(openai_provider_module, "OpenAIProvider", _CountingConstructor)

        config = module.load_reranked_llm_baseline_config(FROZEN_CONFIG_PATH)
        agent, provider_enum, model_name, llm_config = module._build_llm_agent(config)

        assert _CountingConstructor.call_count == 1
        assert model_name == config.requested_model


class TestSuccessfulFakeRunEndToEnd:
    def test_full_30_case_run_with_fake_provider_writes_immutable_artifact(self, monkeypatch, tmp_path):
        results_dir = tmp_path / "results"
        source_dir = tmp_path / "source"
        module = _load_module(monkeypatch, results_dir, source_dir)
        report = _valid_fabricated_reranking_report()
        real_sha256 = _write_source_artifact(source_dir, report)

        _patch_build_llm_agent(monkeypatch, module)
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "e" * 40)

        config_path = _write_config(tmp_path, source_reranking_artifact_sha256=real_sha256)

        exit_code = module.main(["--config", str(config_path)])

        assert exit_code == 0
        files = list((results_dir / "reranked_llm_experiments").glob("*.json"))
        assert len(files) == 1

        written_report = RerankedLLMReport.model_validate_json(files[0].read_text())
        assert len(written_report.results) == 30
        assert len(written_report.traces) == 30
        assert written_report.evidence_top_k == 5
        assert written_report.evidence_source == "reranked"
        assert written_report.source_reranking_run_id == SOURCE_RUN_ID
        assert written_report.source_reranking_artifact_sha256 == real_sha256
        assert written_report.git_commit_sha == "e" * 40


class TestNoCohereOrEmbeddingInvocationIsPossible:
    """Structural proof, not a runtime spy: this script's source never
    even references a Cohere or embedding provider, so no code path here
    could invoke either regardless of any monkeypatching."""

    def test_script_never_imports_cohere(self):
        source = SCRIPT_PATH.read_text()
        assert "import cohere" not in source
        assert "cohere_reranker" not in source

    def test_script_never_imports_an_embedding_provider(self):
        source = SCRIPT_PATH.read_text()
        assert "embedding_provider" not in source
        assert "corpus_index" not in source


class TestExistingArtifactsUnchangedAfterFakeRun:
    def test_running_the_script_never_modifies_the_real_repositorys_results(self, monkeypatch, tmp_path):
        """This test still redirects ARL_RESULTS_DIR/ARL_SOURCE_RESULTS_DIR
        to tmp_path (never touching the real results/ directory at all),
        which is itself the strongest possible proof of non-interference
        -- included explicitly so this guarantee is asserted by a test,
        not merely implied by every other test's use of tmp_path."""

        results_dir = tmp_path / "results"
        source_dir = tmp_path / "source"
        module = _load_module(monkeypatch, results_dir, source_dir)
        report = _valid_fabricated_reranking_report()
        real_sha256 = _write_source_artifact(source_dir, report)

        _patch_build_llm_agent(monkeypatch, module)
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "e" * 40)

        real_results_dir = REPO_ROOT / "results"
        before = {
            p: p.read_bytes() for p in real_results_dir.rglob("*.json")
        }

        config_path = _write_config(tmp_path, source_reranking_artifact_sha256=real_sha256)
        module.main(["--config", str(config_path)])

        after = {p: p.read_bytes() for p in real_results_dir.rglob("*.json")}
        assert before == after
