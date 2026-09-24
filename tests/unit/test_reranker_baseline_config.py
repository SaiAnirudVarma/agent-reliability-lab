"""Unit tests for app.reranking.baseline_config, and the frozen
configs/reranker-baseline-v1.json file's agreement with the actual
compatibility facts established before benchmark exposure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.reranking.baseline_config import RerankerBaselineConfig, load_reranker_baseline_config

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_CONFIG_PATH = REPO_ROOT / "configs" / "reranker-baseline-v1.json"


def _valid_config_dict(**overrides) -> dict:
    defaults = dict(
        config_id="test-config",
        provider="cohere",
        requested_model="rerank-v4.0-pro",
        served_model=None,
        candidate_source_type="preserved_retrieval_artifact",
        candidate_source_run_id="5b363b87-f452-4d1b-90e1-1f2cd1c2ef36",
        candidate_source_artifact_sha256="a" * 64,
        candidate_source_retriever_config_id="retrieval-baseline-v1",
        candidate_depth=10,
        reranker_output_depth=10,
        evaluation_k_values=[1, 3, 5, 10],
        dataset_version="synthetic-v2",
        dataset_fingerprint="b" * 64,
        corpus_fingerprint="c" * 64,
        query_serialization="x",
        document_serialization="x",
    )
    defaults.update(overrides)
    return defaults


class TestRerankerBaselineConfigSchema:
    def test_valid_config_constructs(self):
        config = RerankerBaselineConfig(**_valid_config_dict())
        assert config.config_id == "test-config"

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            RerankerBaselineConfig(**_valid_config_dict(required_evidence_ids=["EV-1"]))

    def test_served_model_defaults_to_none(self):
        data = _valid_config_dict()
        del data["served_model"]
        config = RerankerBaselineConfig(**data)
        assert config.served_model is None

    def test_output_depth_exceeding_candidate_depth_rejected(self):
        with pytest.raises(ValidationError, match="cannot exceed"):
            RerankerBaselineConfig(**_valid_config_dict(candidate_depth=5, reranker_output_depth=10))

    def test_output_depth_equal_to_candidate_depth_accepted(self):
        config = RerankerBaselineConfig(**_valid_config_dict(candidate_depth=10, reranker_output_depth=10))
        assert config.reranker_output_depth == 10

    def test_output_depth_less_than_candidate_depth_accepted(self):
        config = RerankerBaselineConfig(**_valid_config_dict(candidate_depth=10, reranker_output_depth=5))
        assert config.reranker_output_depth == 5

    def test_k_value_exceeding_candidate_depth_rejected(self):
        with pytest.raises(ValidationError, match="out of valid range"):
            RerankerBaselineConfig(**_valid_config_dict(candidate_depth=10, evaluation_k_values=[1, 3, 20]))

    def test_zero_k_value_rejected(self):
        with pytest.raises(ValidationError, match="out of valid range"):
            RerankerBaselineConfig(**_valid_config_dict(evaluation_k_values=[0, 1]))

    def test_empty_k_values_rejected(self):
        with pytest.raises(ValidationError):
            RerankerBaselineConfig(**_valid_config_dict(evaluation_k_values=[]))

    def test_zero_candidate_depth_rejected(self):
        with pytest.raises(ValidationError):
            RerankerBaselineConfig(**_valid_config_dict(candidate_depth=0, reranker_output_depth=0))

    def test_malformed_sha256_rejected(self):
        with pytest.raises(ValidationError):
            RerankerBaselineConfig(**_valid_config_dict(candidate_source_artifact_sha256="not-a-hex-hash"))

    def test_no_secret_or_ground_truth_shaped_field_exists_on_schema(self):
        forbidden_substrings = (
            "key", "secret", "token", "password", "credential",
            "required_evidence", "expected_finding", "rationale", "failure_mode",
        )
        for field_name in RerankerBaselineConfig.model_fields:
            lowered = field_name.lower()
            assert not any(bad in lowered for bad in forbidden_substrings), field_name


class TestLoadRerankerBaselineConfig:
    def test_loads_the_real_frozen_file(self):
        config = load_reranker_baseline_config(FROZEN_CONFIG_PATH)
        assert config.config_id == "cohere-rerank-v4-pro-baseline-v1"

    def test_missing_file_raises_oserror(self, tmp_path):
        with pytest.raises(OSError):
            load_reranker_baseline_config(tmp_path / "does-not-exist.json")

    def test_malformed_file_raises_validation_error(self, tmp_path):
        bad_path = tmp_path / "bad.json"
        bad_path.write_text(json.dumps({"config_id": "x"}))
        with pytest.raises(ValidationError):
            load_reranker_baseline_config(bad_path)


class TestFrozenRerankerConfigContainsNoSecrets:
    def test_no_known_secret_patterns_in_raw_file(self):
        raw = FROZEN_CONFIG_PATH.read_text()
        assert "co_" not in raw  # a plausible cohere key prefix shape
        assert "COHERE_API_KEY" not in raw
        assert "sk-" not in raw

    def test_no_ground_truth_field_names_present(self):
        raw = FROZEN_CONFIG_PATH.read_text()
        for forbidden in ("required_evidence_ids", "expected_finding", "rationale", "failure_mode_tag"):
            assert forbidden not in raw


class TestFrozenRerankerConfigMatchesEstablishedFacts:
    """The compatibility facts established BEFORE benchmark exposure --
    checked against the frozen file directly, so it can never silently
    drift from what was actually verified."""

    def test_provider_and_model(self):
        config = load_reranker_baseline_config(FROZEN_CONFIG_PATH)
        assert config.provider == "cohere"
        assert config.requested_model == "rerank-v4.0-pro"

    def test_served_model_not_fabricated(self):
        config = load_reranker_baseline_config(FROZEN_CONFIG_PATH)
        assert config.served_model is None

    def test_candidate_source_matches_preserved_artifact(self):
        config = load_reranker_baseline_config(FROZEN_CONFIG_PATH)
        assert config.candidate_source_run_id == "5b363b87-f452-4d1b-90e1-1f2cd1c2ef36"
        assert config.candidate_source_artifact_sha256 == "e2e234a2cc8339e4a3d15410f04fcd63a74a15b1317e428defc32fce4a737b05"
        assert config.candidate_source_retriever_config_id == "retrieval-baseline-v1"

    def test_dataset_and_corpus_fingerprints(self):
        config = load_reranker_baseline_config(FROZEN_CONFIG_PATH)
        assert config.dataset_version == "synthetic-v2"
        assert config.dataset_fingerprint == "35e54a143b90b9d8bf30db7e1cbdb27346eddba0e0c71dd8070e69f22fe63a72"
        assert config.corpus_fingerprint == "8b9c0462d085addf206667e35aea7e7cc3cfb9473a4961ec902bc3e71e99a109"

    def test_depths_and_k_values(self):
        config = load_reranker_baseline_config(FROZEN_CONFIG_PATH)
        assert config.candidate_depth == 10
        assert config.reranker_output_depth == 10
        assert config.evaluation_k_values == [1, 3, 5, 10]
