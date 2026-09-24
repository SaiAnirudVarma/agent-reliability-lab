"""Unit tests for app.reranking.provenance.RerankExperimentProvenance --
schema only (no runner produces one yet)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.reranking.provenance import RerankExperimentProvenance


def _valid_dict(**overrides) -> dict:
    defaults = dict(
        run_id="run-1",
        git_commit_sha="a" * 40,
        dataset_version="synthetic-v2",
        dataset_fingerprint="b" * 64,
        corpus_fingerprint="c" * 64,
        retriever_config_id="retrieval-baseline-v1",
        retrieval_result_run_id="5b363b87-f452-4d1b-90e1-1f2cd1c2ef36",
        candidate_depth=10,
        reranker_config_id="pass-through-v1",
        reranker_provider=None,
        requested_reranker_model=None,
        served_reranker_model=None,
        final_k_values=[1, 3, 5],
        generated_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return defaults


class TestRerankExperimentProvenanceSchema:
    def test_valid_provenance_constructs(self):
        provenance = RerankExperimentProvenance(**_valid_dict())
        assert provenance.reranker_config_id == "pass-through-v1"

    def test_provider_model_fields_default_to_none(self):
        data = _valid_dict()
        del data["reranker_provider"], data["requested_reranker_model"], data["served_reranker_model"]
        provenance = RerankExperimentProvenance(**data)
        assert provenance.reranker_provider is None
        assert provenance.requested_reranker_model is None
        assert provenance.served_reranker_model is None

    def test_git_commit_sha_optional(self):
        data = _valid_dict()
        del data["git_commit_sha"]
        provenance = RerankExperimentProvenance(**data)
        assert provenance.git_commit_sha is None

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            RerankExperimentProvenance(**_valid_dict(api_key="sk-should-never-be-here"))

    def test_empty_final_k_values_rejected(self):
        with pytest.raises(ValidationError):
            RerankExperimentProvenance(**_valid_dict(final_k_values=[]))

    def test_missing_required_field_rejected(self):
        data = _valid_dict()
        del data["retrieval_result_run_id"]
        with pytest.raises(ValidationError):
            RerankExperimentProvenance(**data)

    def test_no_secret_shaped_field_exists_on_schema(self):
        forbidden_substrings = ("key", "secret", "token", "password", "credential")
        for field_name in RerankExperimentProvenance.model_fields:
            lowered = field_name.lower()
            assert not any(bad in lowered for bad in forbidden_substrings), field_name

    def test_can_express_a_real_reranker_provider_when_legitimately_known(self):
        """Confirms the schema CAN carry provider info for a future API
        reranker -- the None default is a convention, not a limitation."""
        provenance = RerankExperimentProvenance(
            **_valid_dict(reranker_provider="openai", requested_reranker_model="some-rerank-model", served_reranker_model="some-rerank-model")
        )
        assert provenance.reranker_provider == "openai"
