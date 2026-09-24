"""Unit tests for app.integration.config.RerankedLLMBaselineConfig -- the
frozen, non-secret experiment configuration file for the Phase 8A
reranked-evidence LLM evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.integration.config import RerankedLLMBaselineConfig, load_reranked_llm_baseline_config

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_CONFIG_PATH = REPO_ROOT / "configs" / "reranked-llm-baseline-v1.json"


def _valid_dict(**overrides) -> dict:
    defaults = dict(
        config_id="test-config",
        dataset_version="synthetic-v2", dataset_fingerprint="a" * 64, corpus_fingerprint="b" * 64,
        source_reranking_run_id="run-1", source_reranking_artifact_sha256="c" * 64,
        source_reranking_config_id="reranker-config-1", evidence_top_k=5,
        llm_provider="openai", requested_model="fake-model", served_model=None,
        prompt_version="llm-baseline-v1", temperature=0.0, max_output_tokens=800,
    )
    defaults.update(overrides)
    return defaults


class TestRerankedLLMBaselineConfigSchema:
    def test_valid_config_constructs(self):
        config = RerankedLLMBaselineConfig(**_valid_dict())
        assert config.config_id == "test-config"
        assert config.evidence_top_k == 5

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            RerankedLLMBaselineConfig(**_valid_dict(candidate_depth=10))

    def test_zero_evidence_top_k_rejected(self):
        with pytest.raises(ValidationError):
            RerankedLLMBaselineConfig(**_valid_dict(evidence_top_k=0))

    def test_negative_evidence_top_k_rejected(self):
        with pytest.raises(ValidationError):
            RerankedLLMBaselineConfig(**_valid_dict(evidence_top_k=-1))

    def test_malformed_sha256_rejected(self):
        with pytest.raises(ValidationError):
            RerankedLLMBaselineConfig(**_valid_dict(source_reranking_artifact_sha256="not-a-sha"))

    def test_served_model_defaults_to_none(self):
        config = RerankedLLMBaselineConfig(**{k: v for k, v in _valid_dict().items() if k != "served_model"})
        assert config.served_model is None

    def test_no_ground_truth_shaped_field_exists_on_schema(self):
        forbidden_substrings = ("expected_outcome", "required_evidence", "failure_mode", "ground_truth")
        for field_name in RerankedLLMBaselineConfig.model_fields:
            lowered = field_name.lower()
            assert not any(bad in lowered for bad in forbidden_substrings), field_name

    def test_no_secret_shaped_field_exists_on_schema(self):
        forbidden_substrings = ("api_key", "secret", "password", "credential")
        for field_name in RerankedLLMBaselineConfig.model_fields:
            lowered = field_name.lower()
            assert not any(bad in lowered for bad in forbidden_substrings), field_name


class TestLoadRerankedLLMBaselineConfig:
    def test_loads_the_real_frozen_file(self):
        config = load_reranked_llm_baseline_config(FROZEN_CONFIG_PATH)
        assert config.config_id == "reranked-llm-baseline-v1"
        assert config.evidence_top_k == 5
        assert config.llm_provider == "openai"
        assert config.requested_model == "gpt-5.4-mini-2026-03-17"
        assert config.prompt_version == "llm-baseline-v1"
        assert config.temperature == 0.0
        assert config.max_output_tokens == 800
        assert config.served_model is None

    def test_missing_file_raises_oserror(self, tmp_path):
        with pytest.raises(OSError):
            load_reranked_llm_baseline_config(tmp_path / "does-not-exist.json")

    def test_malformed_file_raises_validation_error(self, tmp_path):
        bad_path = tmp_path / "bad.json"
        bad_path.write_text(json.dumps({"config_id": "x"}))
        with pytest.raises(ValidationError):
            load_reranked_llm_baseline_config(bad_path)


class TestFrozenConfigContainsNoSecrets:
    def test_no_known_secret_patterns(self):
        raw = FROZEN_CONFIG_PATH.read_text()
        assert "OPENAI_API_KEY" not in raw
        assert "sk-" not in raw
