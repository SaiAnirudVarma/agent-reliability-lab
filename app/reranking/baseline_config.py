"""Loader/validator for a frozen reranker-baseline configuration file
(e.g. ``configs/reranker-baseline-v1.json``): the non-secret, tracked
record of exactly which reproducibility parameters an official reranking
experiment run must use. Mirrors
``app.retrieval.baseline_config.RetrievalBaselineConfig`` exactly --
credentials/API keys are NEVER stored here (they remain environment-only),
and no evaluator ground truth (required_evidence_ids, expected outcomes,
failure_mode_tag, or any case-specific setting) belongs in this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class RerankerBaselineConfig(BaseModel):
    """A frozen reranking experiment configuration.

    ``served_model`` defaults to ``None`` and must never be set to a
    fabricated value -- it records what the compatibility check actually
    observed (nothing authoritative), not a guess.

    ``candidate_source_*`` fields identify EXACTLY which preserved
    retrieval artifact a future reranking run must consume -- see
    ``app.reranking.candidate_source`` for the loader that verifies these
    against the actual artifact before any provider call.
    """

    model_config = ConfigDict(extra="forbid")

    config_id: str = Field(min_length=1)

    provider: str = Field(min_length=1)
    requested_model: str = Field(min_length=1)
    served_model: Optional[str] = None

    candidate_source_type: str = Field(min_length=1)
    candidate_source_run_id: str = Field(min_length=1)
    candidate_source_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_source_retriever_config_id: str = Field(min_length=1)

    candidate_depth: int = Field(gt=0)
    reranker_output_depth: int = Field(gt=0)
    evaluation_k_values: list[int] = Field(min_length=1)

    dataset_version: str = Field(min_length=1)
    dataset_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    corpus_fingerprint: str = Field(pattern=_SHA256_PATTERN)

    query_serialization: str = Field(min_length=1)
    document_serialization: str = Field(min_length=1)

    @model_validator(mode="after")
    def _output_depth_cannot_exceed_candidate_depth(self) -> "RerankerBaselineConfig":
        if self.reranker_output_depth > self.candidate_depth:
            raise ValueError(
                f"reranker_output_depth ({self.reranker_output_depth}) cannot exceed "
                f"candidate_depth ({self.candidate_depth}) -- a reranker cannot output more "
                "candidates than it was given."
            )
        return self

    @model_validator(mode="after")
    def _k_values_within_candidate_depth(self) -> "RerankerBaselineConfig":
        invalid = [k for k in self.evaluation_k_values if k < 1 or k > self.candidate_depth]
        if invalid:
            raise ValueError(
                f"evaluation_k_values {invalid} out of valid range [1, {self.candidate_depth}] "
                "(candidate_depth) -- Recall@K cannot be evaluated at a K larger than the "
                "candidate set actually supplied."
            )
        return self


def load_reranker_baseline_config(path: Path) -> RerankerBaselineConfig:
    """Loads and validates a frozen reranker config file. Raises
    ``OSError`` if unreadable, or ``pydantic.ValidationError`` on a schema
    mismatch -- the caller must treat both as fatal."""

    return RerankerBaselineConfig.model_validate_json(path.read_text())
