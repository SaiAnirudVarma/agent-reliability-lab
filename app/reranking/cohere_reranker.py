"""Cohere implementation of the Reranker interface.

Isolates all Cohere SDK/API-specific details (request/response shape,
index-based candidate identity, relevance-score field name) inside this
one module -- ``app.reranking.contracts``/``interface``/``metrics`` know
nothing about Cohere at all.

``CohereReranker`` is constructed with an ALREADY-BUILT client object
(dependency injection), never by importing or calling the real ``cohere``
package itself -- so tests inject a fake/mock client with no
``COHERE_API_KEY`` and no network dependency at all. ``build_cohere_client``
is the one place that actually imports ``cohere`` and constructs a real
client from an API key, for a future production caller -- lazily,
mirroring ``OpenAIProvider``'s own lazy-import convention. No test in this
repository calls ``build_cohere_client``.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Protocol, runtime_checkable

from app.models.contracts import Evidence
from app.reranking.contracts import RerankedEvidence, RerankInput, RerankResult
from app.retrieval.contracts import RetrievedEvidence
from app.retrieval.serialization import evidence_embedding_text, retrieval_query_embedding_text

RERANKER_CONFIG_ID = "cohere-rerank-v4-pro-baseline-v1"


class CohereRerankerError(Exception):
    """Raised on any malformed/invalid Cohere response: an index outside
    range, a duplicate index, a missing result, a non-finite score, or a
    response shape this adapter cannot interpret. Always fails closed --
    no silent recovery, no benchmark-specific workaround."""


@runtime_checkable
class CohereClient(Protocol):
    """The minimal shape this adapter needs from a Cohere client -- real
    or a test double. Matches the Cohere SDK's rerank() call: pass the
    model, the query, the document texts (in a fixed order), and how many
    results to return; get back an object exposing ``.results``, each with
    ``.index`` (position in ``documents``) and ``.relevance_score``.
    """

    def rerank(self, *, model: str, query: str, documents: list[str], top_n: int) -> Any: ...


def build_cohere_client(api_key: str) -> CohereClient:
    """Constructs a real Cohere client from an API key. Lazily imports
    ``cohere`` -- never required unless this function is actually called.
    Production use only; not invoked by any test in this repository."""

    try:
        import cohere
    except ImportError as exc:
        raise ImportError(
            "build_cohere_client requires the 'cohere' package. Install it with: pip install cohere"
        ) from exc
    return cohere.Client(api_key)


class CohereReranker:
    """Reranker backed by Cohere's rerank API.

    ``evidence_by_id`` supplies the full ``Evidence`` record for every
    ``evidence_id`` this instance might be asked to rerank -- injected at
    construction time (mirroring ``VectorRetriever``'s pre-built
    ``CorpusEmbeddingIndex``), so ``rerank()``'s own call signature matches
    the generic ``Reranker`` protocol exactly, with no extra required
    parameter. Only evidence_ids actually present in a given
    ``RerankInput.candidates`` are ever looked up or sent to Cohere -- this
    class has no way to reach any other record, and never sends more than
    the supplied candidate set.

    Query text is produced by
    ``app.retrieval.serialization.retrieval_query_embedding_text`` -- the
    SAME deterministic serializer retrieval already uses -- rather than a
    new, semantically different query built ad hoc for Cohere. That
    serializer already excludes ``case_id``, so no separate exclusion rule
    is needed here: reuse itself is the reason case_id never becomes
    semantic query content.

    Document text is produced by
    ``app.retrieval.serialization.evidence_embedding_text`` for the actual
    ``Evidence`` record behind each candidate -- never the evidence_id,
    document_id, or original vector score alone.

    ``served_reranker_model``: Cohere's rerank API response is not known to
    carry an independently-authoritative "served model" field distinct
    from the requested model (unlike a provider with rolling/dated model
    aliases) -- ``last_served_model_name`` is populated only if a response
    object happens to expose one; it is expected to be ``None`` in
    practice, by design, never a copy of the requested model presented as
    if it were confirmed.
    """

    def __init__(
        self,
        client: CohereClient,
        model_name: str,
        evidence_by_id: dict[str, Evidence],
        reranker_config_id: str = RERANKER_CONFIG_ID,
    ):
        self._client = client
        self.model_name = model_name
        self._evidence_by_id = evidence_by_id
        self.reranker_config_id = reranker_config_id
        self.last_served_model_name: Optional[str] = None
        self.last_request_id: Optional[str] = None

    def rerank(self, rerank_input: RerankInput, *, final_k: Optional[int] = None) -> RerankResult:
        candidates: list[RetrievedEvidence] = sorted(rerank_input.candidates, key=lambda c: c.rank)
        limit = final_k if final_k is not None else len(candidates)
        if limit < 1:
            raise ValueError(f"final_k must be >= 1, got {limit}")

        documents: list[str] = []
        for candidate in candidates:
            evidence = self._evidence_by_id.get(candidate.evidence_id)
            if evidence is None:
                raise CohereRerankerError(
                    f"no Evidence record available for candidate evidence_id {candidate.evidence_id!r} -- "
                    "cannot serialize its content for reranking."
                )
            documents.append(evidence_embedding_text(evidence))

        query_text = retrieval_query_embedding_text(rerank_input.query)

        response = self._client.rerank(model=self.model_name, query=query_text, documents=documents, top_n=limit)

        # Best-effort, never fabricated -- see this class's own docstring.
        self.last_served_model_name = getattr(response, "model", None) or None
        self.last_request_id = getattr(response, "id", None) or getattr(response, "response_id", None) or None

        results = getattr(response, "results", None)
        if results is None:
            raise CohereRerankerError(f"Cohere response has no 'results' field: {response!r}")

        if len(results) != limit:
            raise CohereRerankerError(f"expected {limit} results from Cohere, got {len(results)}")

        validated: list[tuple[RetrievedEvidence, float]] = []
        seen_indices: set[int] = set()
        for item in results:
            index = getattr(item, "index", None)
            score = getattr(item, "relevance_score", None)
            if index is None or score is None:
                raise CohereRerankerError(f"malformed Cohere result item (missing index/relevance_score): {item!r}")
            if isinstance(index, bool) or not isinstance(index, int) or not (0 <= index < len(candidates)):
                raise CohereRerankerError(
                    f"Cohere returned index {index!r} outside the valid candidate range [0, {len(candidates)})"
                )
            if index in seen_indices:
                raise CohereRerankerError(f"Cohere returned duplicate index {index}")
            seen_indices.add(index)

            score = float(score)
            if not math.isfinite(score):
                raise CohereRerankerError(f"Cohere returned a non-finite relevance_score {score} for index {index}")

            validated.append((candidates[index], score))

        # Deterministic tie-break: relevance score descending, then
        # evidence_id ascending -- never raw response order, matching
        # every other ranker in this codebase (LexicalBaselineRetriever,
        # VectorRetriever, PassThroughReranker, FakeReranker).
        validated.sort(key=lambda pair: (-pair[1], pair[0].evidence_id))

        reranked = [
            RerankedEvidence(
                evidence_id=original.evidence_id,
                document_id=original.document_id,
                original_rank=original.rank,
                reranked_rank=rank,
                original_score=original.score,
                reranker_score=score,
                reranker_method=self.reranker_config_id,
            )
            for rank, (original, score) in enumerate(validated, start=1)
        ]

        return RerankResult(
            case_id=rerank_input.case_id,
            input=rerank_input,
            candidates=reranked,
            reranker_config_id=self.reranker_config_id,
        )
