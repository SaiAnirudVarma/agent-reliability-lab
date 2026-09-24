"""Loader/validator for a reranker EXECUTION policy configuration file
(e.g. ``configs/reranker-execution-cohere-trial-v1.json``).

Deliberately a SEPARATE config type from
``app.reranking.baseline_config.RerankerBaselineConfig``: the execution
policy is transport/pacing/retry behavior, never the experiment's
*semantic* identity (candidate source, candidate depth, model, K values).
Mutating an execution policy must never be able to change which
experiment a run represents -- see
``docs/incidents/2026-09-24-cohere-reranking-rate-limit.md``.

No secrets belong here either.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RerankerExecutionPolicy(BaseModel):
    """``retry_policy = "none"`` and ``max_attempts_per_case = 1`` together
    encode this project's current, deliberate choice: no automatic retry
    of any provider failure (429, timeout, 5xx, or otherwise). Recovery
    from a provider failure remains a human decision -- this schema
    enforces that a config cannot claim "no retry" while also declaring
    more than one attempt per case, which would be self-contradictory.
    """

    model_config = ConfigDict(extra="forbid")

    execution_policy_id: str = Field(min_length=1)
    minimum_call_start_interval_seconds: float = Field(ge=0.0)
    retry_policy: str = Field(min_length=1)
    max_attempts_per_case: int = Field(ge=1)

    @model_validator(mode="after")
    def _no_retry_implies_single_attempt(self) -> "RerankerExecutionPolicy":
        if self.retry_policy == "none" and self.max_attempts_per_case != 1:
            raise ValueError(
                f"retry_policy='none' requires max_attempts_per_case == 1, "
                f"got {self.max_attempts_per_case}"
            )
        return self


def load_reranker_execution_policy(path: Path) -> RerankerExecutionPolicy:
    """Loads and validates an execution policy file. Raises ``OSError`` if
    unreadable, or ``pydantic.ValidationError`` on a schema mismatch."""

    return RerankerExecutionPolicy.model_validate_json(path.read_text())
