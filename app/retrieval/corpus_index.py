"""The embed-once corpus cache: 72 evidence documents embedded a single
time, reused for every subsequent ``VectorRetriever.retrieve()`` call
(AC-001..AC-030), never re-embedded per case.

``CorpusEmbeddingIndex`` is a plain in-memory dict-backed structure
deliberately: for a 72-document benchmark this is trivial to inspect and
reproduce, and keeping it a simple dataclass (rather than reaching for
pgvector/FAISS/etc. now) means a future swap to a real vector index only
has to reproduce this same shape (``vectors_by_evidence_id`` lookup), not
change every caller.

Also carries full provenance -- corpus content identity
(``corpus_fingerprint``) and embedding-provider identity (``provider``,
``model_name``, ``served_model_name``, ``embedding_dimension``) -- so a
``VectorRetriever`` built from this index, and any report generated from
it, can prove exactly what was embedded and by what, not merely assume it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.models.contracts import Evidence, ModelProvider
from app.retrieval.corpus import EvidenceCorpus
from app.retrieval.embedding_provider import EmbeddingProvider, Vector
from app.retrieval.serialization import evidence_embedding_text


@dataclass(frozen=True)
class CorpusEmbeddingIndex:
    """Every evidence record in one corpus, embedded once, keyed by
    ``evidence_id``.

    ``corpus_fingerprint`` is copied from the exact ``EvidenceCorpus`` that
    was embedded (see ``app.retrieval.corpus.compute_corpus_fingerprint``)
    -- content identity, not just a version label. ``model_name`` is the
    REQUESTED embedding model; ``served_model_name`` is what the provider's
    response reported for this batch, or ``None`` if the provider cannot
    legitimately report one (never fabricated by copying ``model_name``).
    """

    version: str
    corpus_fingerprint: str
    provider: ModelProvider
    model_name: str
    served_model_name: Optional[str]
    embedding_dimension: int
    evidence_by_id: dict[str, Evidence]
    vectors_by_evidence_id: dict[str, Vector]

    def __len__(self) -> int:
        return len(self.vectors_by_evidence_id)


def build_corpus_embedding_index(corpus: EvidenceCorpus, embedding_provider: EmbeddingProvider) -> CorpusEmbeddingIndex:
    """Embeds every record in ``corpus`` via exactly ONE
    ``embed_documents()`` call -- not one call per record, and this
    function is meant to be called exactly once per corpus; ``VectorRetriever``
    reuses the resulting index across every subsequent ``retrieve()`` call
    rather than calling this again.
    """

    texts = [evidence_embedding_text(evidence) for evidence in corpus.evidence]
    embedded = embedding_provider.embed_documents(texts)
    if len(embedded) != len(corpus.evidence):
        raise ValueError(
            f"embedding_provider returned {len(embedded)} vectors for {len(corpus.evidence)} texts"
        )

    served_model_names = {item.served_model_name for item in embedded}
    if len(served_model_names) > 1:
        raise ValueError(
            f"embedding_provider reported inconsistent served model names within one batch: "
            f"{sorted(n for n in served_model_names if n is not None)}"
        )
    served_model_name = served_model_names.pop() if served_model_names else None

    evidence_by_id: dict[str, Evidence] = {}
    vectors_by_evidence_id: dict[str, Vector] = {}
    for evidence, item in zip(corpus.evidence, embedded):
        if len(item.vector) != embedding_provider.embedding_dimension:
            raise ValueError(
                f"embedding for {evidence.evidence_id} has dimension {len(item.vector)}, "
                f"expected {embedding_provider.embedding_dimension}"
            )
        evidence_by_id[evidence.evidence_id] = evidence
        vectors_by_evidence_id[evidence.evidence_id] = item.vector

    return CorpusEmbeddingIndex(
        version=corpus.version,
        corpus_fingerprint=corpus.fingerprint,
        provider=embedding_provider.provider,
        model_name=embedding_provider.model_name,
        served_model_name=served_model_name,
        embedding_dimension=embedding_provider.embedding_dimension,
        evidence_by_id=evidence_by_id,
        vectors_by_evidence_id=vectors_by_evidence_id,
    )
