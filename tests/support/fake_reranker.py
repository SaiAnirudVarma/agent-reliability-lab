"""A test-only Reranker that assigns caller-supplied evidence_id -> score
values, with no network calls and no production heuristic of any kind.

Exists ONLY to validate reranking plumbing (contract enforcement, rank
movement, metric deltas) -- mirrors ``tests.support.fake_llm_provider.FakeLLMProvider``'s
role for ``LLMAgent``. Must never be reused as, or mistaken for, a real
reranking method.
"""

from __future__ import annotations

from typing import Optional

from app.reranking.contracts import RerankedEvidence, RerankInput, RerankResult


class FakeReranker:
    """Ties break by ``evidence_id`` ascending -- the same deterministic
    convention every other ranker in this codebase uses. Scoring a
    candidate not present in ``scores_by_evidence_id`` is a test-authoring
    error, not a legitimate "unknown evidence" case a real reranker would
    need to handle -- it raises ``KeyError`` immediately.
    """

    def __init__(self, scores_by_evidence_id: dict[str, float], reranker_config_id: str = "fake-reranker-v1"):
        self._scores = scores_by_evidence_id
        self.reranker_config_id = reranker_config_id

    def rerank(self, rerank_input: RerankInput, *, final_k: Optional[int] = None) -> RerankResult:
        candidates = rerank_input.candidates
        limit = final_k if final_k is not None else len(candidates)
        if limit < 1:
            raise ValueError(f"final_k must be >= 1, got {limit}")

        scored = []
        for candidate in candidates:
            if candidate.evidence_id not in self._scores:
                raise KeyError(f"FakeReranker has no configured score for evidence_id: {candidate.evidence_id!r}")
            scored.append((candidate, self._scores[candidate.evidence_id]))
        scored.sort(key=lambda pair: (-pair[1], pair[0].evidence_id))

        selected = scored[:limit]
        reranked = [
            RerankedEvidence(
                evidence_id=candidate.evidence_id,
                document_id=candidate.document_id,
                original_rank=candidate.rank,
                reranked_rank=new_rank,
                original_score=candidate.score,
                reranker_score=score,
                reranker_method=self.reranker_config_id,
            )
            for new_rank, (candidate, score) in enumerate(selected, start=1)
        ]

        return RerankResult(
            case_id=rerank_input.case_id,
            input=rerank_input,
            candidates=reranked,
            reranker_config_id=self.reranker_config_id,
        )
