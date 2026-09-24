"""Unit tests for app.retrieval.embedding_provider: validate_vector and
FakeEmbeddingProvider.
"""

from __future__ import annotations

import math

import pytest

from app.models.contracts import ModelProvider
from app.retrieval.embedding_provider import EmbeddingProvider, FakeEmbeddingProvider, validate_vector


class TestValidateVector:
    def test_valid_vector_passes_through_as_tuple(self):
        assert validate_vector([1.0, 2.0, 3.0]) == (1.0, 2.0, 3.0)

    def test_empty_vector_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            validate_vector([])

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            validate_vector([1.0, float("nan"), 3.0])

    def test_positive_infinity_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            validate_vector([1.0, float("inf")])

    def test_negative_infinity_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            validate_vector([float("-inf"), 1.0])

    def test_dimension_mismatch_rejected_when_expected_given(self):
        with pytest.raises(ValueError, match="dimension"):
            validate_vector([1.0, 2.0], expected_dimension=3)

    def test_matching_dimension_accepted(self):
        assert validate_vector([1.0, 2.0, 3.0], expected_dimension=3) == (1.0, 2.0, 3.0)

    def test_zero_vector_is_valid(self):
        """A zero vector is a legitimate (if geometrically degenerate)
        embedding -- validation only rejects malformed values, not
        zero-magnitude ones. Cosine similarity's own zero-vector handling
        lives in app.retrieval.vector_math, not here."""
        assert validate_vector([0.0, 0.0, 0.0]) == (0.0, 0.0, 0.0)


class TestFakeEmbeddingProvider:
    def test_returns_configured_vector_for_known_text(self):
        provider = FakeEmbeddingProvider({"hello": (1.0, 0.0), "world": (0.0, 1.0)})
        assert provider.embed_query("hello").vector == (1.0, 0.0)

    def test_embed_documents_returns_in_order(self):
        provider = FakeEmbeddingProvider({"a": (1.0, 0.0), "b": (0.0, 1.0)})
        assert [item.vector for item in provider.embed_documents(["b", "a"])] == [(0.0, 1.0), (1.0, 0.0)]

    def test_served_model_name_defaults_to_none_never_fabricated(self):
        provider = FakeEmbeddingProvider({"a": (1.0,)})
        assert provider.embed_query("a").served_model_name is None
        assert provider.embed_documents(["a"])[0].served_model_name is None

    def test_served_model_name_explicit_override_for_testing(self):
        provider = FakeEmbeddingProvider({"a": (1.0,)}, served_model_name="fake-served-v1")
        assert provider.embed_query("a").served_model_name == "fake-served-v1"
        assert provider.embed_documents(["a"])[0].served_model_name == "fake-served-v1"

    def test_unknown_text_raises_key_error(self):
        provider = FakeEmbeddingProvider({"a": (1.0, 0.0)})
        with pytest.raises(KeyError):
            provider.embed_query("unconfigured text")

    def test_dimension_inferred_and_consistent(self):
        provider = FakeEmbeddingProvider({"a": (1.0, 0.0, 0.0), "b": (0.0, 1.0, 0.0)})
        assert provider.embedding_dimension == 3

    def test_inconsistent_dimensions_rejected_at_construction(self):
        with pytest.raises(ValueError, match="dimension"):
            FakeEmbeddingProvider({"a": (1.0, 0.0), "b": (0.0, 1.0, 0.0)})

    def test_empty_mapping_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            FakeEmbeddingProvider({})

    def test_call_count_and_embedded_texts_tracked(self):
        provider = FakeEmbeddingProvider({"a": (1.0,), "b": (2.0,), "c": (3.0,)})
        provider.embed_documents(["a", "b"])
        provider.embed_query("c")
        assert provider.embed_call_count == 2
        assert provider.embedded_texts == ["a", "b", "c"]

    def test_satisfies_embedding_provider_protocol(self):
        provider = FakeEmbeddingProvider({"a": (1.0,)})
        assert isinstance(provider, EmbeddingProvider)

    def test_provider_identity_and_model_name(self):
        provider = FakeEmbeddingProvider({"a": (1.0,)}, model_name="fake-v2")
        assert provider.provider == ModelProvider.OPENAI
        assert provider.model_name == "fake-v2"

    def test_nan_in_configured_vector_rejected_at_construction(self):
        with pytest.raises(ValueError, match="non-finite"):
            FakeEmbeddingProvider({"a": (1.0, math.nan)})
