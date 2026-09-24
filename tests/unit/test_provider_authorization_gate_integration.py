"""Integration tests proving the real-API-call authorization gate
(app.observability.real_api_gate) is actually wired into every real-
provider experiment CLI's ``_build_*`` function, BEFORE any real
provider client is constructed -- for scripts/run_eval.py (OpenAI LLM),
scripts/run_retrieval_eval.py (OpenAI embeddings), and
scripts/run_reranking_eval.py (Cohere).

scripts/run_reranked_llm_eval.py (Phase 8A, OpenAI LLM) is covered by its
own dedicated gate tests in tests/unit/test_run_reranked_llm_eval_script.py
instead of here, since that script (and app.integration.config, which its
gate tests construct) does not exist at this safety-fix commit.

Each script is loaded as an independent module object via importlib --
never as a subprocess, and no real provider SDK is ever constructed. A
fake, but real-SHAPED, credential is used throughout (e.g.
"sk-fake-not-a-real-key-for-this-test") specifically to prove that
credential presence ALONE is insufficient -- exactly the scenario that
caused docs/incidents/2026-09-24-unauthorized-real-openai-calls-during-phase-8a-smoketest.md.

The "blocked" tests additionally wrap the call in
tests.support.network_guard.block_network() as a defense-in-depth proof:
even if the gate itself somehow failed to block, no real socket
connection could complete during the call.
"""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

import pytest

from app.observability.real_api_gate import RealApiCallsNotAuthorizedError
from tests.support.network_guard import block_network

REPO_ROOT = Path(__file__).resolve().parents[2]
NONEXISTENT_DOTENV_PATH = str(REPO_ROOT / ".pytest-isolation-nonexistent-dir" / ".env")

FAKE_OPENAI_KEY = "sk-fake-not-a-real-key-for-this-test"
FAKE_COHERE_KEY = "fake-cohere-key-for-this-test"


def _load_module(monkeypatch, script_name: str, results_dir: Path, extra_env: dict[str, str]):
    monkeypatch.setenv("ARL_DOTENV_PATH", NONEXISTENT_DOTENV_PATH)
    monkeypatch.setenv("ARL_RESULTS_DIR", str(results_dir))
    monkeypatch.setenv("ARL_SOURCE_RESULTS_DIR", str(results_dir))
    for key in (
        "ARL_ALLOW_REAL_API_CALLS", "AGENT_MODE", "MODEL_PROVIDER", "MODEL_NAME",
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "RERANKER_PROVIDER", "COHERE_API_KEY",
        "EMBEDDING_PROVIDER", "EMBEDDING_MODEL_NAME", "EMBEDDING_DIMENSION",
        "ARL_RUN_ID", "ARL_GIT_COMMIT_SHA",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in extra_env.items():
        monkeypatch.setenv(key, value)

    script_path = REPO_ROOT / "scripts" / script_name
    module_name = f"{script_name.replace('.py', '')}_gate_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _CountingConstructor:
    """A stand-in for a real provider class: records how many times it
    was constructed, accepts any arguments, and never touches a
    network."""

    call_count = 0

    def __init__(self, *args, **kwargs):
        type(self).call_count += 1
        self.args = args
        self.kwargs = kwargs


class TestRunEvalLLMAgentGate:
    """scripts/run_eval.py -- _build_llm_agent (OpenAI)."""

    def _module(self, monkeypatch, tmp_path):
        return _load_module(
            monkeypatch, "run_eval.py", tmp_path,
            {"MODEL_PROVIDER": "openai", "MODEL_NAME": "fake-model-for-this-test", "OPENAI_API_KEY": FAKE_OPENAI_KEY},
        )

    @pytest.mark.parametrize("bad_value", [None, "0", "false", "true"])
    def test_blocked_without_exact_authorization_zero_provider_construction(self, monkeypatch, tmp_path, bad_value):
        module = self._module(monkeypatch, tmp_path)
        if bad_value is not None:
            monkeypatch.setenv("ARL_ALLOW_REAL_API_CALLS", bad_value)

        counter = _CountingConstructor
        counter.call_count = 0
        import app.providers.openai_provider as openai_provider_module

        monkeypatch.setattr(openai_provider_module, "OpenAIProvider", counter)

        with block_network():
            with pytest.raises(RealApiCallsNotAuthorizedError):
                module._build_llm_agent()

        assert counter.call_count == 0

    def test_blocked_via_full_cli_returns_exit_1(self, monkeypatch, tmp_path, capsys):
        module = self._module(monkeypatch, tmp_path)
        with block_network():
            exit_code = module.main(["--agent", "llm", "--dataset-version", "synthetic-v1"])
        assert exit_code == 1
        assert "ARL_ALLOW_REAL_API_CALLS" in capsys.readouterr().err
        assert not (tmp_path / "experiments").exists()

    def test_explicit_authorization_allows_construction_to_proceed(self, monkeypatch, tmp_path):
        module = self._module(monkeypatch, tmp_path)
        monkeypatch.setenv("ARL_ALLOW_REAL_API_CALLS", "1")

        counter = _CountingConstructor
        counter.call_count = 0
        import app.providers.openai_provider as openai_provider_module

        monkeypatch.setattr(openai_provider_module, "OpenAIProvider", counter)

        agent, provider_enum, model_name, llm_config = module._build_llm_agent()
        assert counter.call_count == 1
        assert model_name == "fake-model-for-this-test"


class TestRunRetrievalEmbeddingProviderGate:
    """scripts/run_retrieval_eval.py -- _build_vector_embedding_provider (OpenAI embeddings)."""

    def _module(self, monkeypatch, tmp_path):
        return _load_module(
            monkeypatch, "run_retrieval_eval.py", tmp_path,
            {
                "EMBEDDING_PROVIDER": "openai", "EMBEDDING_MODEL_NAME": "fake-embedding-model",
                "EMBEDDING_DIMENSION": "8", "OPENAI_API_KEY": FAKE_OPENAI_KEY,
            },
        )

    @pytest.mark.parametrize("bad_value", [None, "0", "false", "true"])
    def test_blocked_without_exact_authorization_zero_provider_construction(self, monkeypatch, tmp_path, bad_value):
        module = self._module(monkeypatch, tmp_path)
        if bad_value is not None:
            monkeypatch.setenv("ARL_ALLOW_REAL_API_CALLS", bad_value)

        counter = _CountingConstructor
        counter.call_count = 0
        import app.retrieval.openai_embedding_provider as embedding_module

        monkeypatch.setattr(embedding_module, "OpenAIEmbeddingProvider", counter)

        with block_network():
            with pytest.raises(RealApiCallsNotAuthorizedError):
                module._build_vector_embedding_provider()

        assert counter.call_count == 0

    def test_blocked_via_full_cli_returns_exit_1(self, monkeypatch, tmp_path, capsys):
        module = self._module(monkeypatch, tmp_path)
        with block_network():
            exit_code = module.main(["--retriever", "vector", "--dataset-version", "synthetic-v1"])
        assert exit_code == 1
        assert "ARL_ALLOW_REAL_API_CALLS" in capsys.readouterr().err
        assert not (tmp_path / "retrieval_experiments").exists()

    def test_explicit_authorization_allows_construction_to_proceed(self, monkeypatch, tmp_path):
        module = self._module(monkeypatch, tmp_path)
        monkeypatch.setenv("ARL_ALLOW_REAL_API_CALLS", "1")

        counter = _CountingConstructor
        counter.call_count = 0
        import app.retrieval.openai_embedding_provider as embedding_module

        monkeypatch.setattr(embedding_module, "OpenAIEmbeddingProvider", counter)

        provider = module._build_vector_embedding_provider()
        assert counter.call_count == 1


class TestRunRerankingCohereGate:
    """scripts/run_reranking_eval.py -- _build_reranker (Cohere)."""

    def _module(self, monkeypatch, tmp_path):
        return _load_module(
            monkeypatch, "run_reranking_eval.py", tmp_path,
            {"RERANKER_PROVIDER": "cohere", "COHERE_API_KEY": FAKE_COHERE_KEY},
        )

    @pytest.mark.parametrize("bad_value", [None, "0", "false", "true"])
    def test_blocked_without_exact_authorization_zero_client_construction(self, monkeypatch, tmp_path, bad_value):
        module = self._module(monkeypatch, tmp_path)
        if bad_value is not None:
            monkeypatch.setenv("ARL_ALLOW_REAL_API_CALLS", bad_value)

        call_count = {"n": 0}

        def _counting_build_client(api_key):
            call_count["n"] += 1
            return object()

        import app.reranking.cohere_reranker as cohere_reranker_module

        monkeypatch.setattr(cohere_reranker_module, "build_cohere_client", _counting_build_client)

        with block_network():
            with pytest.raises(RealApiCallsNotAuthorizedError):
                module._build_reranker("fake-model", {}, "fake-config-id")

        assert call_count["n"] == 0

    def test_blocked_via_full_cli_returns_exit_1(self, monkeypatch, tmp_path, capsys):
        # Source-artifact verification happens BEFORE the authorization
        # gate in this script's documented safety ordering -- point the
        # source dir (an env var read at MODULE LOAD time, so this must
        # be set before loading the module, not after) at this repo's
        # real, preserved (read-only) artifact, so that check succeeds
        # and the authorization gate is what actually blocks this test,
        # not a fixture-setup gap.
        module = _load_module(
            monkeypatch, "run_reranking_eval.py", tmp_path,
            {
                "RERANKER_PROVIDER": "cohere", "COHERE_API_KEY": FAKE_COHERE_KEY,
                "ARL_SOURCE_RESULTS_DIR": str(REPO_ROOT / "results"),
            },
        )
        with block_network():
            exit_code = module.main(["--config", str(REPO_ROOT / "configs" / "reranker-baseline-v1.json")])
        assert exit_code == 1
        assert "ARL_ALLOW_REAL_API_CALLS" in capsys.readouterr().err
        assert not (tmp_path / "reranking_experiments").exists()

    def test_explicit_authorization_allows_construction_to_proceed(self, monkeypatch, tmp_path):
        module = self._module(monkeypatch, tmp_path)
        monkeypatch.setenv("ARL_ALLOW_REAL_API_CALLS", "1")

        call_count = {"n": 0}

        def _counting_build_client(api_key):
            call_count["n"] += 1
            return object()

        import app.reranking.cohere_reranker as cohere_reranker_module

        monkeypatch.setattr(cohere_reranker_module, "build_cohere_client", _counting_build_client)

        reranker = module._build_reranker("fake-model", {}, "fake-config-id")
        assert call_count["n"] == 1


class TestDotenvCannotImplicitlyAuthorize:
    """Proves that a credential loaded from .env (not the real shell
    environment) still cannot authorize a real call by itself -- the
    exact mechanism behind
    docs/incidents/2026-09-24-unauthorized-real-openai-calls-during-phase-8a-smoketest.md."""

    def test_dotenv_loaded_credential_alone_still_blocks(self, monkeypatch, tmp_path):
        fake_dotenv_path = tmp_path / "fake.env"
        # Deliberately contains ONLY a credential -- no ARL_ALLOW_REAL_API_CALLS
        # line at all -- mirroring this repository's real .env at the time
        # of the incident.
        fake_dotenv_path.write_text(f"OPENAI_API_KEY={FAKE_OPENAI_KEY}\n")

        monkeypatch.setenv("ARL_DOTENV_PATH", str(fake_dotenv_path))
        monkeypatch.setenv("ARL_RESULTS_DIR", str(tmp_path))
        monkeypatch.delenv("ARL_ALLOW_REAL_API_CALLS", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("MODEL_PROVIDER", "openai")
        monkeypatch.setenv("MODEL_NAME", "fake-model-for-this-test")

        script_path = REPO_ROOT / "scripts" / "run_eval.py"
        module_name = f"run_eval_dotenv_test_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # this is where load_dotenv(override=False) runs

        # load_dotenv populated OPENAI_API_KEY from the fake .env file --
        # confirm the credential IS present, so this test genuinely
        # exercises "credential present via dotenv, authorization absent",
        # not an accidental no-op.
        import os

        assert os.environ.get("OPENAI_API_KEY") == FAKE_OPENAI_KEY

        with block_network():
            with pytest.raises(RealApiCallsNotAuthorizedError):
                module._build_llm_agent()
