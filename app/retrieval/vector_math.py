"""Vector similarity, used by VectorRetriever. Pure functions, no
dependency on any embedding provider or corpus type.
"""

from __future__ import annotations

import math

from app.retrieval.embedding_provider import Vector


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity of two equal-length vectors.

    Zero-vector convention: cosine similarity is mathematically undefined
    for a zero vector (0/0 in the denominator). Rather than raising, this
    returns ``0.0`` explicitly -- "no defined direction, therefore no
    similarity" -- the same "define the empty case rather than let it
    fall out of arithmetic" convention used throughout this codebase (see
    e.g. ``compute_citation_validity``'s zero-citation rule,
    ``percentile``'s empty-input rule).
    """

    if len(a) != len(b):
        raise ValueError(f"vector dimension mismatch: {len(a)} vs {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
