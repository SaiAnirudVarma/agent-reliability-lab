"""PassThroughReranker: a deterministic reranker that preserves the exact
incoming ranking.

Purpose: plumbing validation and a control condition -- proof that the act
of running candidates through the reranking evaluation pipeline does not,
by itself, alter any metric. It contains no lexical/entity/employee-ID/
case-ID logic and no benchmark-specific behavior of any kind; it reads
nothing from ``rerank_input`` except ``candidates`` and (when given)
trims to ``final_k`` without reordering.
"""

from __future__ import annotations

from typing import Optional

from app.reranking.contracts import RerankedEvidence, RerankInput, RerankResult

RERANKER_CONFIG_ID = "pass-through-v1"


class PassThroughReranker:
    reranker_config_id = RERANKER_CONFIG_ID

    def rerank(self, rerank_input: RerankInput, *, final_k: Optional[int] = None) -> RerankResult:
        candidates = sorted(rerank_input.candidates, key=lambda c: c.rank)
        limit = final_k if final_k is not None else len(candidates)
        if limit < 1:
            raise ValueError(f"final_k must be >= 1, got {limit}")

        selected = candidates[:limit]
        reranked = [
            RerankedEvidence(
                evidence_id=c.evidence_id,
                document_id=c.document_id,
                original_rank=c.rank,
                reranked_rank=new_rank,
                original_score=c.score,
                reranker_score=c.score,
                reranker_method=RERANKER_CONFIG_ID,
            )
            for new_rank, c in enumerate(selected, start=1)
        ]

        return RerankResult(
            case_id=rerank_input.case_id,
            input=rerank_input,
            candidates=reranked,
            reranker_config_id=RERANKER_CONFIG_ID,
        )
