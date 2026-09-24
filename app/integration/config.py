"""Loader/validator for a frozen reranked-LLM baseline configuration file
(e.g. ``configs/reranked-llm-baseline-v1.json``): the non-secret, tracked
record of exactly which reproducibility parameters an official
reranked-evidence LLM experiment run must use.

Mirrors ``app.reranking.baseline_config.RerankerBaselineConfig`` exactly --
credentials/API keys are NEVER stored here (they remain environment-only),
and no evaluator ground truth (required_evidence_ids, expected outcomes,
failure_mode_tag, or any case-specific setting) belongs in this file.

``evidence_top_k`` is frozen HERE, before any LLM exposure -- see the
module docstring on ``app.integration.runner`` for why this value must
never be tuned after seeing LLM results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class RerankedLLMBaselineConfig(BaseModel):
    """A frozen reranked-evidence LLM experiment configuration.

    ``served_model`` defaults to ``None`` and must never be set to a
    fabricated value -- it records what a real run actually observed
    (nothing authoritative until then), not a guess.

    ``source_reranking_*`` fields identify EXACTLY which preserved
    reranking artifact a future run must consume -- see
    ``app.integration.source`` for the loader that verifies these against
    the actual artifact before any LLM provider call.
    """

    model_config = ConfigDict(extra="forbid")

    config_id: str = Field(min_length=1)

    dataset_version: str = Field(min_length=1)
    dataset_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    corpus_fingerprint: str = Field(pattern=_SHA256_PATTERN)

    source_reranking_run_id: str = Field(min_length=1)
    source_reranking_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_reranking_config_id: str = Field(min_length=1)

    evidence_top_k: int = Field(gt=0)

    llm_provider: str = Field(min_length=1)
    requested_model: str = Field(min_length=1)
    served_model: Optional[str] = None

    prompt_version: str = Field(min_length=1)
    temperature: float
    max_output_tokens: int = Field(gt=0)


def load_reranked_llm_baseline_config(path: Path) -> RerankedLLMBaselineConfig:
    """Loads and validates a frozen reranked-LLM baseline config file.
    Raises ``OSError`` if unreadable, or ``pydantic.ValidationError`` on a
    schema mismatch -- the caller must treat both as fatal."""

    return RerankedLLMBaselineConfig.model_validate_json(path.read_text())
