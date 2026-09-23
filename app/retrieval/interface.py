"""The Retriever interface every retrieval implementation must satisfy.

Deliberately minimal and provider-agnostic -- mirrors
``app.agent.interface.AgentRunner``'s own minimal Protocol shape (one
identity attribute, one method), so a future vector-embedding retriever or
a reranking retriever can implement this exact same interface without it
needing to change, and so callers (the eventual Phase 7B pipeline) can
depend on this Protocol rather than any concrete retriever class.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.retrieval.contracts import RetrievalQuery, RetrievalResult
from app.retrieval.corpus import EvidenceCorpus


@runtime_checkable
class Retriever(Protocol):
    """``retriever_config_id`` identifies this retriever's exact
    configuration (mirroring ``AgentRunner.agent_config_id``), so a
    persisted ``RetrievalResult`` remains traceable to exactly what
    produced it without needing to inspect source code.

    ``retrieve`` takes an already-built, ground-truth-free
    ``RetrievalQuery`` and a ``corpus`` explicitly supplied by the caller
    (never looked up internally by case ID) -- a Retriever implementation
    has no legitimate way to reach ground truth or a different case's
    intended answer through this interface at all.
    """

    retriever_config_id: str

    def retrieve(self, query: RetrievalQuery, corpus: EvidenceCorpus, *, top_k: int) -> RetrievalResult: ...
