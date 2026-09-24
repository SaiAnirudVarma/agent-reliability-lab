"""The Reranker interface every reranking implementation must satisfy.

Mirrors ``app.retrieval.interface.Retriever``'s minimal shape deliberately:
one identity attribute, one method, provider-agnostic -- so a future
deterministic/local reranker, cross-encoder reranker, or API reranker can
all implement this exact interface without evaluation code needing to
change, and without a caller depending on any concrete class.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from app.reranking.contracts import RerankInput, RerankResult


@runtime_checkable
class Reranker(Protocol):
    """``reranker_config_id`` identifies this reranker's exact
    configuration (mirroring ``Retriever.retriever_config_id``), so a
    persisted ``RerankResult`` remains traceable to exactly what produced
    it.

    ``rerank`` takes an already-built ``RerankInput`` (never looks up a
    candidate set itself, never reaches ground truth) and an optional
    ``final_k`` -- the number of candidates to keep in the output, letting
    a caller freeze e.g. "vector Top-10 -> reranker -> final Top-5"
    without the reranker choosing that number itself. ``final_k=None``
    (the default) means "keep the whole candidate set, just reordered."
    """

    reranker_config_id: str

    def rerank(self, rerank_input: RerankInput, *, final_k: Optional[int] = None) -> RerankResult: ...
