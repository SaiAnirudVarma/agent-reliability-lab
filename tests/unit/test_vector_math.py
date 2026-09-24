"""Unit tests for app.retrieval.vector_math.cosine_similarity."""

from __future__ import annotations

import math

import pytest

from app.retrieval.vector_math import cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors_are_similarity_one(self):
        assert cosine_similarity((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)) == pytest.approx(1.0)

    def test_orthogonal_vectors_are_similarity_zero(self):
        assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)

    def test_opposite_vectors_are_similarity_negative_one(self):
        assert cosine_similarity((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(-1.0)

    def test_scale_invariant(self):
        a = (1.0, 2.0, 3.0)
        b = (2.0, 4.0, 6.0)  # same direction, different magnitude
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_known_value(self):
        # cos(theta) between (1,0) and (1,1) = 1/sqrt(2)
        assert cosine_similarity((1.0, 0.0), (1.0, 1.0)) == pytest.approx(1 / math.sqrt(2))

    def test_zero_vector_a_returns_zero_not_error(self):
        assert cosine_similarity((0.0, 0.0), (1.0, 1.0)) == 0.0

    def test_zero_vector_b_returns_zero_not_error(self):
        assert cosine_similarity((1.0, 1.0), (0.0, 0.0)) == 0.0

    def test_both_zero_vectors_returns_zero(self):
        assert cosine_similarity((0.0, 0.0), (0.0, 0.0)) == 0.0

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity((1.0, 2.0), (1.0, 2.0, 3.0))
