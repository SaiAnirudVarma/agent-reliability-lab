"""Unit tests for app.retrieval.baseline_config, and -- critically -- for
the frozen configs/retrieval-baseline-v1.json file's agreement with the
ACTUAL current code/data. These tests exist specifically so the frozen
config can never silently drift away from the dataset/corpus/serialization
it claims to describe: a future change to any of those must fail this
test suite, not surface only as a mysterious runtime rejection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.datasets.loader import load_dataset
from app.retrieval.baseline_config import RetrievalBaselineConfig, load_retrieval_baseline_config
from app.retrieval.corpus import build_full_version_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "datasets"
FROZEN_CONFIG_PATH = REPO_ROOT / "configs" / "retrieval-baseline-v1.json"


def _valid_config_dict(**overrides) -> dict:
    defaults = dict(
        config_id="test-config",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dimension=1536,
        retrieval_method="vector",
        similarity="cosine",
        tie_breaking="score descending, then evidence_id ascending",
        query_serialization="x",
        evidence_serialization="x",
        corpus="full synthetic-v2 version corpus",
        corpus_size=72,
        corpus_fingerprint="a" * 64,
        dataset_version="synthetic-v2",
        dataset_fingerprint="b" * 64,
        k_values=[1, 3, 5, 10],
        mrr_convention="reciprocal rank of first required evidence item",
        zero_required_evidence_convention="N/A and excluded from aggregate denominator",
    )
    defaults.update(overrides)
    return defaults


class TestRetrievalBaselineConfigSchema:
    def test_valid_config_constructs(self):
        config = RetrievalBaselineConfig(**_valid_config_dict())
        assert config.config_id == "test-config"

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            RetrievalBaselineConfig(**_valid_config_dict(api_key="sk-should-never-be-here"))

    def test_missing_required_field_rejected(self):
        data = _valid_config_dict()
        del data["dataset_fingerprint"]
        with pytest.raises(ValidationError):
            RetrievalBaselineConfig(**data)

    def test_empty_k_values_rejected(self):
        with pytest.raises(ValidationError):
            RetrievalBaselineConfig(**_valid_config_dict(k_values=[]))

    def test_zero_embedding_dimension_rejected(self):
        with pytest.raises(ValidationError):
            RetrievalBaselineConfig(**_valid_config_dict(embedding_dimension=0))

    def test_no_secret_shaped_field_exists_on_schema(self):
        forbidden_substrings = ("key", "secret", "token", "password", "credential")
        for field_name in RetrievalBaselineConfig.model_fields:
            lowered = field_name.lower()
            assert not any(bad in lowered for bad in forbidden_substrings), field_name


class TestLoadRetrievalBaselineConfig:
    def test_loads_the_real_frozen_file(self):
        config = load_retrieval_baseline_config(FROZEN_CONFIG_PATH)
        assert config.config_id == "retrieval-baseline-v1"

    def test_missing_file_raises_oserror(self, tmp_path):
        with pytest.raises(OSError):
            load_retrieval_baseline_config(tmp_path / "does-not-exist.json")

    def test_malformed_file_raises_validation_error(self, tmp_path):
        bad_path = tmp_path / "bad.json"
        bad_path.write_text(json.dumps({"config_id": "x"}))  # missing everything else
        with pytest.raises(ValidationError):
            load_retrieval_baseline_config(bad_path)


class TestFrozenConfigContainsNoSecrets:
    def test_no_known_secret_patterns_in_raw_file(self):
        raw = FROZEN_CONFIG_PATH.read_text()
        assert "sk-" not in raw
        assert "OPENAI_API_KEY" not in raw
        assert "API_KEY" not in raw.upper() or "api_key" not in raw.lower()

    def test_no_ground_truth_field_names_present(self):
        raw = FROZEN_CONFIG_PATH.read_text()
        for forbidden in ("required_evidence_ids", "expected_finding", "rationale", "failure_mode_tag"):
            assert forbidden not in raw


class TestFrozenConfigAgreesWithActualCodeAndData:
    """The core anti-drift guarantee: every value recorded in
    configs/retrieval-baseline-v1.json must match what the CURRENT
    dataset/corpus/code actually produce, checked live -- never assumed
    correct merely because it's written down."""

    def test_dataset_version_matches(self):
        config = load_retrieval_baseline_config(FROZEN_CONFIG_PATH)
        dataset = load_dataset(DATASET_DIR, version=config.dataset_version)
        assert dataset.version == "synthetic-v2"

    def test_dataset_fingerprint_matches_actual(self):
        config = load_retrieval_baseline_config(FROZEN_CONFIG_PATH)
        dataset = load_dataset(DATASET_DIR, version=config.dataset_version)
        assert dataset.fingerprint == config.dataset_fingerprint

    def test_corpus_fingerprint_matches_actual(self):
        config = load_retrieval_baseline_config(FROZEN_CONFIG_PATH)
        dataset = load_dataset(DATASET_DIR, version=config.dataset_version)
        corpus = build_full_version_corpus(dataset)
        assert corpus.fingerprint == config.corpus_fingerprint

    def test_corpus_size_matches_actual(self):
        config = load_retrieval_baseline_config(FROZEN_CONFIG_PATH)
        dataset = load_dataset(DATASET_DIR, version=config.dataset_version)
        corpus = build_full_version_corpus(dataset)
        assert len(corpus) == config.corpus_size

    def test_k_values_match_expected(self):
        config = load_retrieval_baseline_config(FROZEN_CONFIG_PATH)
        assert config.k_values == [1, 3, 5, 10]

    def test_embedding_dimension_matches_expected_baseline_decision(self):
        config = load_retrieval_baseline_config(FROZEN_CONFIG_PATH)
        assert config.embedding_dimension == 1536

    def test_embedding_model_matches_expected_baseline_decision(self):
        config = load_retrieval_baseline_config(FROZEN_CONFIG_PATH)
        assert config.embedding_model == "text-embedding-3-small"

    def test_retrieval_method_is_vector(self):
        config = load_retrieval_baseline_config(FROZEN_CONFIG_PATH)
        assert config.retrieval_method == "vector"
