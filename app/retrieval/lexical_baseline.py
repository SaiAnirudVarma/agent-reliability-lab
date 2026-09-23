"""LexicalBaselineRetriever: a deterministic, token-overlap retriever used
ONLY to validate Phase 7 plumbing (query -> corpus -> ranked candidates ->
RetrievalResult).

This is explicitly NOT the Phase 7 experimental retrieval method. It must
never be tuned against AC-001..AC-030 -- doing so would make it an
overfit oracle rather than a plumbing check, exactly the failure mode
``docs/RETRIEVAL_DESIGN.md`` documents this class as existing to avoid.
"""

from __future__ import annotations

import re

from app.retrieval.contracts import RetrievalQuery, RetrievalResult, RetrievedEvidence
from app.retrieval.corpus import EvidenceCorpus
from app.models.contracts import Evidence

RETRIEVER_CONFIG_ID = "lexical-baseline-v1"

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.lower()))


def _query_tokens(query: RetrievalQuery) -> set[str]:
    return _tokenize(
        " ".join(
            [
                query.control.name,
                query.control.description,
                query.control.requirement_text,
                query.scenario_description,
            ]
        )
    )


def _evidence_tokens(evidence: Evidence) -> set[str]:
    return _tokenize(" ".join([evidence.title, evidence.content]))


class LexicalBaselineRetriever:
    """Scores every corpus evidence record by token-overlap count against
    the query's control text + scenario description, and returns the
    ``top_k`` highest-scoring records.

    Deterministic by construction: scoring depends only on the query's and
    each evidence record's own text content, never on ``query.case_id``,
    never on iteration/insertion order (ties break on ``evidence_id``,
    which is stable and independent of corpus construction order), and
    never on any external state. Never inspects ``ExpectedOutcome`` or
    ``required_evidence_ids`` -- it cannot, since neither reaches this
    class through ``RetrievalQuery`` or ``EvidenceCorpus`` at all.
    """

    retriever_config_id = RETRIEVER_CONFIG_ID

    def retrieve(self, query: RetrievalQuery, corpus: EvidenceCorpus, *, top_k: int) -> RetrievalResult:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        query_tokens = _query_tokens(query)

        scored = [
            (evidence, float(len(query_tokens & _evidence_tokens(evidence))))
            for evidence in corpus.evidence
        ]
        # Deterministic tie-break: score descending, then evidence_id
        # ascending -- never corpus iteration order, never case_id.
        scored.sort(key=lambda pair: (-pair[1], pair[0].evidence_id))

        top = scored[:top_k]
        candidates = [
            RetrievedEvidence(
                evidence_id=evidence.evidence_id,
                document_id=evidence.document_id,
                score=score,
                rank=rank,
                retrieval_method=RETRIEVER_CONFIG_ID,
            )
            for rank, (evidence, score) in enumerate(top, start=1)
        ]

        return RetrievalResult(
            case_id=query.case_id,
            query=query,
            candidates=candidates,
            top_k=top_k,
            retriever_config_id=RETRIEVER_CONFIG_ID,
        )
