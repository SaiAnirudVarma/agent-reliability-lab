"""In-process tests for scripts/run_retrieval_eval.py's --config handling:
the frozen-config-driven "official run" path, and its fingerprint/model/
dimension drift checks -- ALL of which must fail before any real provider
call. Mirrors tests/unit/test_run_retrieval_eval_script.py's module-loading
approach; zero network, zero real credentials.
"""

from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path

from app.datasets.loader import load_dataset
from app.retrieval.baseline_config import load_retrieval_baseline_config
from app.retrieval.contracts import build_retrieval_query
from app.retrieval.corpus import build_full_version_corpus
from app.retrieval.embedding_provider import FakeEmbeddingProvider
from app.retrieval.runner import RetrievalReport
from app.retrieval.serialization import evidence_embedding_text, retrieval_query_embedding_text

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_retrieval_eval.py"
FROZEN_CONFIG_PATH = REPO_ROOT / "configs" / "retrieval-baseline-v1.json"
NONEXISTENT_DOTENV_PATH = str(REPO_ROOT / ".pytest-isolation-nonexistent-dir" / ".env")
DATASET_DIR = REPO_ROOT / "datasets"


def _load_module(monkeypatch, results_dir: Path):
    monkeypatch.setenv("ARL_DOTENV_PATH", NONEXISTENT_DOTENV_PATH)
    monkeypatch.setenv("ARL_RESULTS_DIR", str(results_dir))
    for key in ("ARL_RUN_ID", "ARL_GIT_COMMIT_SHA", "EMBEDDING_PROVIDER", "EMBEDDING_MODEL_NAME", "EMBEDDING_DIMENSION", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    module_name = f"run_retrieval_eval_config_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _full_coverage_fake_provider_for_v2(dimension: int = 1536, model_name: str = "text-embedding-3-small") -> FakeEmbeddingProvider:
    """A provider with a length-`dimension` vector configured for every
    real synthetic-v2 evidence/query text -- so the official-config happy
    path can run to completion with zero network."""

    dataset = load_dataset(DATASET_DIR, version="synthetic-v2")
    corpus = build_full_version_corpus(dataset)

    vectors: dict[str, tuple[float, ...]] = {}
    for evidence in corpus.evidence:
        vectors[evidence_embedding_text(evidence)] = tuple([float(len(evidence.evidence_id))] * dimension)
    for case in dataset.cases:
        control = dataset.controls[case.control_id]
        query = build_retrieval_query(case, control)
        vectors[retrieval_query_embedding_text(query)] = tuple([float(len(case.case_id))] * dimension)

    return FakeEmbeddingProvider(vectors, model_name=model_name)


def _write_config_variant(tmp_path: Path, **overrides) -> Path:
    base = json.loads(FROZEN_CONFIG_PATH.read_text())
    base.update(overrides)
    path = tmp_path / "variant-config.json"
    path.write_text(json.dumps(base))
    return path


class TestOfficialConfigDrivenRunHappyPath:
    def test_succeeds_end_to_end_with_frozen_config(self, monkeypatch, tmp_path):
        module = _load_module(monkeypatch, tmp_path)
        provider = _full_coverage_fake_provider_for_v2()
        monkeypatch.setattr(module, "_build_vector_embedding_provider", lambda: provider)
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "fake-sha-for-this-test")

        exit_code = module.main(["--config", str(FROZEN_CONFIG_PATH)])

        assert exit_code == 0
        files = list((tmp_path / "retrieval_experiments").glob("*.json"))
        assert len(files) == 1
        assert files[0].name.startswith("synthetic-v2__retrieval-baseline-v1__")
        report = RetrievalReport.model_validate_json(files[0].read_text())
        assert report.retriever_config_id == "retrieval-baseline-v1"
        assert report.dataset_version == "synthetic-v2"
        assert len(report.results) == 30
        assert report.top_k_values == [1, 3, 5, 10]

    def test_dataset_version_and_retriever_flags_ignored_when_config_given(self, monkeypatch, tmp_path):
        """--dataset-version synthetic-v1 / --retriever lexical must be
        overridden by the config's own synthetic-v2/vector settings."""
        module = _load_module(monkeypatch, tmp_path)
        provider = _full_coverage_fake_provider_for_v2()
        monkeypatch.setattr(module, "_build_vector_embedding_provider", lambda: provider)
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "fake-sha-for-this-test")

        exit_code = module.main([
            "--config", str(FROZEN_CONFIG_PATH), "--dataset-version", "synthetic-v1", "--retriever", "lexical",
        ])

        assert exit_code == 0
        report = RetrievalReport.model_validate_json(
            next((tmp_path / "retrieval_experiments").glob("*.json")).read_text()
        )
        assert report.dataset_version == "synthetic-v2"  # config wins, not the CLI flag


class TestConfigDriftRejectedBeforeAnyProviderCall:
    def test_dataset_fingerprint_mismatch_rejected(self, monkeypatch, tmp_path, capsys):
        module = _load_module(monkeypatch, tmp_path)
        bad_config_path = _write_config_variant(tmp_path, dataset_fingerprint="0" * 64)
        provider = _full_coverage_fake_provider_for_v2()
        monkeypatch.setattr(module, "_build_vector_embedding_provider", lambda: provider)
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "fake-sha-for-this-test")

        exit_code = module.main(["--config", str(bad_config_path)])

        assert exit_code == 1
        assert "dataset_fingerprint" in capsys.readouterr().err
        assert provider.embed_call_count == 0
        assert not (tmp_path / "retrieval_experiments").exists()

    def test_corpus_fingerprint_mismatch_rejected(self, monkeypatch, tmp_path, capsys):
        module = _load_module(monkeypatch, tmp_path)
        # dataset_fingerprint stays correct (real value) so the run gets
        # past THAT check specifically -- only corpus_fingerprint is wrong.
        bad_config_path = _write_config_variant(tmp_path, corpus_fingerprint="0" * 64)
        provider = _full_coverage_fake_provider_for_v2()
        monkeypatch.setattr(module, "_build_vector_embedding_provider", lambda: provider)
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "fake-sha-for-this-test")

        exit_code = module.main(["--config", str(bad_config_path)])

        assert exit_code == 1
        assert "corpus_fingerprint" in capsys.readouterr().err
        assert provider.embed_call_count == 0
        assert not (tmp_path / "retrieval_experiments").exists()

    def test_embedding_model_mismatch_rejected(self, monkeypatch, tmp_path, capsys):
        module = _load_module(monkeypatch, tmp_path)
        provider = _full_coverage_fake_provider_for_v2(model_name="a-different-model")
        monkeypatch.setattr(module, "_build_vector_embedding_provider", lambda: provider)
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "fake-sha-for-this-test")

        exit_code = module.main(["--config", str(FROZEN_CONFIG_PATH)])

        assert exit_code == 1
        assert "embedding_model" in capsys.readouterr().err
        assert provider.embed_call_count == 0
        assert not (tmp_path / "retrieval_experiments").exists()

    def test_embedding_dimension_mismatch_rejected(self, monkeypatch, tmp_path, capsys):
        module = _load_module(monkeypatch, tmp_path)
        provider = _full_coverage_fake_provider_for_v2(dimension=8)
        monkeypatch.setattr(module, "_build_vector_embedding_provider", lambda: provider)
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "fake-sha-for-this-test")

        exit_code = module.main(["--config", str(FROZEN_CONFIG_PATH)])

        assert exit_code == 1
        assert "embedding_dimension" in capsys.readouterr().err
        assert provider.embed_call_count == 0
        assert not (tmp_path / "retrieval_experiments").exists()

    def test_missing_git_sha_still_rejected_for_config_driven_run(self, monkeypatch, tmp_path, capsys):
        module = _load_module(monkeypatch, tmp_path)
        provider = _full_coverage_fake_provider_for_v2()
        monkeypatch.setattr(module, "_build_vector_embedding_provider", lambda: provider)
        monkeypatch.setattr(module, "get_git_commit_sha", lambda repo_root: None)

        exit_code = module.main(["--config", str(FROZEN_CONFIG_PATH)])

        assert exit_code == 1
        assert provider.embed_call_count == 0
        assert not (tmp_path / "retrieval_experiments").exists()


class TestConfigFileErrors:
    def test_missing_config_file(self, monkeypatch, tmp_path, capsys):
        module = _load_module(monkeypatch, tmp_path)
        exit_code = module.main(["--config", str(tmp_path / "does-not-exist.json")])
        assert exit_code == 1
        assert "could not read config file" in capsys.readouterr().err

    def test_malformed_config_file(self, monkeypatch, tmp_path, capsys):
        module = _load_module(monkeypatch, tmp_path)
        bad_path = tmp_path / "bad.json"
        bad_path.write_text(json.dumps({"config_id": "x"}))
        exit_code = module.main(["--config", str(bad_path)])
        assert exit_code == 1
        assert "not a valid RetrievalBaselineConfig" in capsys.readouterr().err
