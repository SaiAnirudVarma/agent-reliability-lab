"""In-process tests for scripts/run_retrieval_eval.py: the precall safety
ordering (run_id -> git_commit_sha -> artifact path -> collision check,
ALL before any real embedding-provider call), the immutable artifact
naming/collision refusal, and the results-root override.

Loaded as an independent module object via importlib -- never as a
subprocess, and the real OpenAI SDK is never constructed -- exactly
mirroring tests/unit/test_run_eval_experiments.py's approach for the LLM
experiment script. ``_build_vector_embedding_provider`` is monkeypatched
to return a FakeEmbeddingProvider, so these tests exercise the script's
real control flow with zero network and zero real credentials.
"""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

from app.datasets.loader import load_dataset
from app.retrieval.corpus import build_full_version_corpus
from app.retrieval.embedding_provider import FakeEmbeddingProvider
from app.retrieval.runner import RetrievalReport
from app.retrieval.serialization import evidence_embedding_text, retrieval_query_embedding_text

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_retrieval_eval.py"
NONEXISTENT_DOTENV_PATH = str(REPO_ROOT / ".pytest-isolation-nonexistent-dir" / ".env")
DATASET_DIR = REPO_ROOT / "datasets"


def _load_module(monkeypatch, results_dir: Path):
    monkeypatch.setenv("ARL_DOTENV_PATH", NONEXISTENT_DOTENV_PATH)
    monkeypatch.setenv("ARL_RESULTS_DIR", str(results_dir))
    for key in ("ARL_RUN_ID", "ARL_GIT_COMMIT_SHA", "EMBEDDING_PROVIDER", "EMBEDDING_MODEL_NAME", "EMBEDDING_DIMENSION", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    module_name = f"run_retrieval_eval_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _placeholder_fake_provider() -> FakeEmbeddingProvider:
    """A provider with nothing real configured -- valid to CONSTRUCT (no
    network either way), but any real embed_documents/embed_query call on
    it that isn't for the exact placeholder text would raise KeyError.
    Used for tests proving the provider is never actually invoked."""

    return FakeEmbeddingProvider({"__never_actually_embedded__": (1.0,)})


def _full_coverage_fake_provider(dataset, corpus) -> FakeEmbeddingProvider:
    """A provider with a vector configured for every real text this
    dataset/corpus combination will actually ask to embed -- used for the
    one happy-path test that runs the full pipeline to completion."""

    from app.retrieval.contracts import build_retrieval_query

    vectors: dict[str, tuple[float, ...]] = {}
    for evidence in corpus.evidence:
        vectors[evidence_embedding_text(evidence)] = (float(len(evidence.evidence_id)),)
    for case in dataset.cases:
        control = dataset.controls[case.control_id]
        query = build_retrieval_query(case, control)
        vectors[retrieval_query_embedding_text(query)] = (float(len(case.case_id)),)
    return FakeEmbeddingProvider(vectors, model_name="fake-embed-script-test")


class TestLexicalRetrieverNeverTouchesEmbeddingMachinery:
    def test_default_run_writes_under_retrieval_experiments(self, monkeypatch, tmp_path):
        module = _load_module(monkeypatch, tmp_path)
        exit_code = module.main([])
        assert exit_code == 0
        files = list((tmp_path / "retrieval_experiments").glob("*.json"))
        assert len(files) == 1
        assert files[0].name.startswith("synthetic-v1__lexical-baseline-v1__")
        report = RetrievalReport.model_validate_json(files[0].read_text())
        assert report.dataset_version == "synthetic-v1"
        assert len(report.results) == 10
        assert report.embedding_provider is None

    def test_results_root_override_respected(self, monkeypatch, tmp_path):
        custom_dir = tmp_path / "custom-results-location"
        module = _load_module(monkeypatch, custom_dir)
        exit_code = module.main([])
        assert exit_code == 0
        assert (custom_dir / "retrieval_experiments").exists()
        assert not (tmp_path / "retrieval_experiments").exists()


class TestPrecallSafetyOrderingForVectorRetriever:
    def test_missing_git_sha_blocks_before_any_embedding_call(self, monkeypatch, tmp_path, capsys):
        module = _load_module(monkeypatch, tmp_path)
        placeholder = _placeholder_fake_provider()
        monkeypatch.setattr(module, "_build_vector_embedding_provider", lambda: placeholder)
        monkeypatch.setattr(module, "get_git_commit_sha", lambda repo_root: None)

        exit_code = module.main(["--retriever", "vector"])

        assert exit_code == 1
        assert "git commit" in capsys.readouterr().err.lower()
        assert placeholder.embed_call_count == 0
        assert not (tmp_path / "retrieval_experiments").exists()

    def test_collision_blocks_before_any_embedding_call(self, monkeypatch, tmp_path, capsys):
        module = _load_module(monkeypatch, tmp_path)
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "fake-sha-for-this-test")
        monkeypatch.setenv("ARL_RUN_ID", "colliding-run-id")
        placeholder = _placeholder_fake_provider()
        monkeypatch.setattr(module, "_build_vector_embedding_provider", lambda: placeholder)

        existing = tmp_path / "retrieval_experiments" / f"synthetic-v1__vector-{placeholder.model_name}__colliding-run-id.json"
        existing.parent.mkdir(parents=True)
        existing.write_text('{"sentinel": "pre-existing, must survive"}')

        exit_code = module.main(["--retriever", "vector"])

        assert exit_code == 1
        assert "already exists" in capsys.readouterr().err
        assert existing.read_text() == '{"sentinel": "pre-existing, must survive"}'
        assert placeholder.embed_call_count == 0  # never reached the real provider call

    def test_missing_embedding_config_fails_before_git_or_collision_checks(self, monkeypatch, tmp_path, capsys):
        """No EMBEDDING_PROVIDER/EMBEDDING_MODEL_NAME/EMBEDDING_DIMENSION
        set at all -- config validation must fail first, cleanly."""
        module = _load_module(monkeypatch, tmp_path)
        exit_code = module.main(["--retriever", "vector"])
        assert exit_code == 1
        assert "EMBEDDING_PROVIDER" in capsys.readouterr().err

    def test_successful_vector_run_writes_artifact_with_full_provenance(self, monkeypatch, tmp_path):
        """The happy path: once every safety check passes, the (fake)
        provider IS invoked exactly as expected, and the artifact carries
        full provenance."""
        module = _load_module(monkeypatch, tmp_path)
        dataset = load_dataset(DATASET_DIR, version="synthetic-v1")
        corpus = build_full_version_corpus(dataset)
        provider = _full_coverage_fake_provider(dataset, corpus)
        monkeypatch.setattr(module, "_build_vector_embedding_provider", lambda: provider)
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "fake-sha-for-this-test")

        exit_code = module.main(["--retriever", "vector"])

        assert exit_code == 0
        files = list((tmp_path / "retrieval_experiments").glob("*.json"))
        assert len(files) == 1
        assert files[0].name.startswith("synthetic-v1__vector-fake-embed-script-test__")
        report = RetrievalReport.model_validate_json(files[0].read_text())
        assert report.git_commit_sha == "fake-sha-for-this-test"
        assert report.requested_embedding_model == "fake-embed-script-test"
        assert report.corpus_fingerprint == corpus.fingerprint
        assert report.corpus_embedding_latency_ms is not None
        assert len(report.results) == 10
        # 1 corpus-embedding batch call + 1 embed_query per case
        assert provider.embed_call_count == 1 + 10
