"""VectorRetriever: a Retriever implementation backed by an
EmbeddingProvider and a pre-built CorpusEmbeddingIndex.

Cosine similarity ranking, deterministic tie-break, no case-specific
logic, no access to any benchmark answer, and no filtering down to
``case.evidence_pool`` -- it always ranks the full corpus its index was
built from. Documents are embedded exactly once (by
``build_corpus_embedding_index``, before this class is ever constructed);
``retrieve()`` embeds only the query, once per call.
"""

from __future__ import annotations

from typing import Optional

from app.retrieval.contracts import RetrievalQuery, RetrievalResult, RetrievedEvidence
from app.retrieval.corpus import EvidenceCorpus
from app.retrieval.corpus_index import CorpusEmbeddingIndex
from app.retrieval.embedding_provider import EmbeddingProvider
from app.retrieval.serialization import retrieval_query_embedding_text
from app.retrieval.vector_math import cosine_similarity


class VectorRetriever:
    """``retriever_config_id`` defaults to ``f"vector-{embedding_provider.model_name}"``
    so a persisted result is traceable to exactly which embedding model
    produced it without inspecting source code; a caller may override it
    to distinguish two retrievers that happen to share a model name but
    differ in some other configuration detail.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        index: CorpusEmbeddingIndex,
        retriever_config_id: Optional[str] = None,
    ):
        if embedding_provider.model_name != index.model_name:
            raise ValueError(
                f"embedding_provider model_name {embedding_provider.model_name!r} does not match "
                f"index model_name {index.model_name!r} -- queries and documents must be embedded "
                "with the same model."
            )
        self._embedding_provider = embedding_provider
        self._index = index
        self.retriever_config_id = retriever_config_id or f"vector-{embedding_provider.model_name}"

    def retrieve(self, query: RetrievalQuery, corpus: EvidenceCorpus, *, top_k: int) -> RetrievalResult:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if corpus.version != self._index.version:
            raise ValueError(
                f"corpus version {corpus.version!r} does not match this retriever's prebuilt index "
                f"version {self._index.version!r} -- build a new CorpusEmbeddingIndex for this corpus."
            )
        # The authoritative check: content identity, not just a version
        # label (see app.retrieval.corpus.compute_corpus_fingerprint and
        # docs/RETRIEVAL_DESIGN.md). A version string can coincidentally
        # match while the actual evidence content underneath it has
        # changed; the fingerprint cannot.
        if corpus.fingerprint != self._index.corpus_fingerprint:
            raise ValueError(
                f"corpus fingerprint {corpus.fingerprint!r} does not match this retriever's prebuilt "
                f"index fingerprint {self._index.corpus_fingerprint!r} -- the corpus content has "
                "changed since it was embedded. Build a new CorpusEmbeddingIndex for this corpus."
            )

        query_embedded = self._embedding_provider.embed_query(retrieval_query_embedding_text(query))
        query_vector = query_embedded.vector

        scored = [
            (evidence_id, cosine_similarity(query_vector, vector))
            for evidence_id, vector in self._index.vectors_by_evidence_id.items()
        ]
        # Deterministic tie-break: score descending, then evidence_id
        # ascending -- never dict iteration order, never case_id.
        scored.sort(key=lambda pair: (-pair[1], pair[0]))

        top = scored[:top_k]
        candidates = [
            RetrievedEvidence(
                evidence_id=evidence_id,
                document_id=self._index.evidence_by_id[evidence_id].document_id,
                score=score,
                rank=rank,
                retrieval_method=self.retriever_config_id,
            )
            for rank, (evidence_id, score) in enumerate(top, start=1)
        ]

        return RetrievalResult(
            case_id=query.case_id,
            query=query,
            candidates=candidates,
            top_k=top_k,
            retriever_config_id=self.retriever_config_id,
        )
