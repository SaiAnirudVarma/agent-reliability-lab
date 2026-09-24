"""Unit tests for app.reranking.execution_policy.RerankerExecutionPolicy --
the SEPARATE, non-semantic execution/transport policy config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.reranking.execution_policy import RerankerExecutionPolicy, load_reranker_execution_policy

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_EXECUTION_CONFIG_PATH = REPO_ROOT / "configs" / "reranker-execution-cohere-trial-v1.json"


def _valid_dict(**overrides) -> dict:
    defaults = dict(
        execution_policy_id="test-policy", minimum_call_start_interval_seconds=7.0,
        retry_policy="none", max_attempts_per_case=1,
    )
    defaults.update(overrides)
    return defaults


class TestRerankerExecutionPolicySchema:
    def test_valid_policy_constructs(self):
        policy = RerankerExecutionPolicy(**_valid_dict())
        assert policy.execution_policy_id == "test-policy"

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            RerankerExecutionPolicy(**_valid_dict(candidate_depth=10))

    def test_retry_none_with_more_than_one_attempt_rejected(self):
        with pytest.raises(ValidationError, match="max_attempts_per_case"):
            RerankerExecutionPolicy(**_valid_dict(retry_policy="none", max_attempts_per_case=2))

    def test_zero_attempts_rejected(self):
        with pytest.raises(ValidationError):
            RerankerExecutionPolicy(**_valid_dict(max_attempts_per_case=0))

    def test_negative_interval_rejected(self):
        with pytest.raises(ValidationError):
            RerankerExecutionPolicy(**_valid_dict(minimum_call_start_interval_seconds=-1.0))

    def test_zero_interval_accepted(self):
        policy = RerankerExecutionPolicy(**_valid_dict(minimum_call_start_interval_seconds=0.0))
        assert policy.minimum_call_start_interval_seconds == 0.0

    def test_no_secret_shaped_field_exists_on_schema(self):
        forbidden_substrings = ("key", "secret", "token", "password", "credential")
        for field_name in RerankerExecutionPolicy.model_fields:
            lowered = field_name.lower()
            assert not any(bad in lowered for bad in forbidden_substrings), field_name


class TestLoadRerankerExecutionPolicy:
    def test_loads_the_real_frozen_file(self):
        policy = load_reranker_execution_policy(FROZEN_EXECUTION_CONFIG_PATH)
        assert policy.execution_policy_id == "cohere-trial-pacing-v1"
        assert policy.minimum_call_start_interval_seconds == 7.0
        assert policy.retry_policy == "none"
        assert policy.max_attempts_per_case == 1

    def test_missing_file_raises_oserror(self, tmp_path):
        with pytest.raises(OSError):
            load_reranker_execution_policy(tmp_path / "does-not-exist.json")

    def test_malformed_file_raises_validation_error(self, tmp_path):
        bad_path = tmp_path / "bad.json"
        bad_path.write_text(json.dumps({"execution_policy_id": "x"}))
        with pytest.raises(ValidationError):
            load_reranker_execution_policy(bad_path)


class TestFrozenExecutionConfigContainsNoSecrets:
    def test_no_known_secret_patterns(self):
        raw = FROZEN_EXECUTION_CONFIG_PATH.read_text()
        assert "COHERE_API_KEY" not in raw
        assert "co_" not in raw
