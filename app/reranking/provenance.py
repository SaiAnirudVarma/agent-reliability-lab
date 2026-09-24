"""Provenance schema for a FUTURE immutable reranking experiment.

Phase 7C is foundation only -- no runner in this codebase produces a
``RerankExperimentProvenance`` yet, and no real reranking experiment has
been run. This model exists so that when a real experiment is eventually
authorized, its provenance discipline is already designed and reviewed
rather than improvised under time pressure -- the same reasoning that led
``docs/ARTIFACT_POLICY.md``'s "Future real-experiment naming" section to
be adopted as policy ahead of Phase 6's LLM experiment infrastructure.

Once produced, this is meant to answer: "exactly which frozen retrieval
result produced the candidate set that this reranker evaluated?" --
``retrieval_result_run_id`` (paired with ``retriever_config_id``) is that
answer, distinct from this reranking run's own ``run_id``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class RerankExperimentProvenance(BaseModel):
    """Non-secret, tracked provenance for one reranking experiment run.

    ``reranker_provider``/``requested_reranker_model``/``served_reranker_model``
    are all ``None`` for a pass-through or other local/deterministic
    reranker -- there is no provider or model to report, and none is
    fabricated (mirrors ``EmbeddedVector.served_model_name``'s "never
    invent it" rule in ``app.retrieval.embedding_provider``). A future API
    reranker populates whichever of these it can legitimately report.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    git_commit_sha: Optional[str] = None

    dataset_version: str = Field(min_length=1)
    dataset_fingerprint: str = Field(min_length=1)
    corpus_fingerprint: str = Field(min_length=1)

    retriever_config_id: str = Field(min_length=1)
    retrieval_result_run_id: str = Field(min_length=1)

    candidate_depth: int = Field(ge=1)

    reranker_config_id: str = Field(min_length=1)
    reranker_provider: Optional[str] = None
    requested_reranker_model: Optional[str] = None
    served_reranker_model: Optional[str] = None

    final_k_values: list[int] = Field(min_length=1)

    generated_at: datetime
