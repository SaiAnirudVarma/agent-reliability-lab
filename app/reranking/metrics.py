"""Reranking metrics: retrieval quality AFTER reranking, and the delta
relative to BEFORE reranking (the same case's original retrieval
candidates). Mirrors ``app.retrieval.metrics``'s Recall@K/MRR conventions
exactly, applied to ``RerankedEvidence.reranked_rank`` instead of
``RetrievedEvidence.rank``. Deliberately a separate module from
``app.retrieval.metrics`` -- candidate generation and reranking stay
independently measurable, per the Phase 7C architectural goal.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.reranking.contracts import RerankedEvidence
from app.retrieval.contracts import RetrievedEvidence


def recall_at_k(required_evidence_ids: list[str], candidates: list[RerankedEvidence], k: int) -> Optional[float]:
    """Recall@K after reranking. Same formula and the same empty-required
    ``None`` convention as ``app.retrieval.metrics.recall_at_k`` -- see
    that function's docstring."""

    if not required_evidence_ids:
        return None
    required = set(required_evidence_ids)
    within_k = {c.evidence_id for c in candidates if c.reranked_rank <= k}
    hit = required & within_k
    return len(hit) / len(required)


def mrr(required_evidence_ids: list[str], candidates: list[RerankedEvidence]) -> Optional[float]:
    """MRR after reranking. Same convention as ``app.retrieval.metrics.mrr``
    -- earliest-ranked relevant hit only, ``0.0`` if none retrieved,
    ``None`` if nothing was required."""

    if not required_evidence_ids:
        return None
    required = set(required_evidence_ids)
    for candidate in sorted(candidates, key=lambda c: c.reranked_rank):
        if candidate.evidence_id in required:
            return 1.0 / candidate.reranked_rank
    return 0.0


def aggregate_recall_at_k(values: list[Optional[float]]) -> float:
    """Mean over applicable (non-``None``) cases; ``0.0`` if none are
    applicable -- identical convention to ``app.retrieval.metrics``."""

    applicable = [v for v in values if v is not None]
    if not applicable:
        return 0.0
    return sum(applicable) / len(applicable)


def aggregate_mrr(values: list[Optional[float]]) -> float:
    applicable = [v for v in values if v is not None]
    if not applicable:
        return 0.0
    return sum(applicable) / len(applicable)


def compute_delta(before: Optional[float], after: Optional[float]) -> Optional[float]:
    """``after - before``, for either Recall@K or MRR (both use the same
    "before/after, both real numbers" shape, so one function serves both
    rather than two near-identical copies). ``None`` if either side is
    ``None`` -- a delta is not applicable when the underlying metric
    itself wasn't (e.g. a case with no required evidence at all)."""

    if before is None or after is None:
        return None
    return after - before


class RequiredEvidenceRankMovement(BaseModel):
    """Where one required evidence item stood before and after reranking,
    for ONE case.

    ``in_candidate_set=False`` means candidate generation never supplied
    this item at all -- a candidate-generation failure, structurally
    impossible for the reranker to have fixed (it never saw the item), so
    ``reranked_rank``/``rank_delta`` are both ``None`` and this item is
    never counted against the reranker. ``in_candidate_set=True`` with
    ``reranked_rank=None`` means the reranker's OWN output dropped this
    item (e.g. by trimming to a smaller ``final_k``) -- a genuine
    reranking-stage outcome, distinguishable from the candidate-generation
    case by ``in_candidate_set`` alone.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    in_candidate_set: bool
    original_rank: Optional[int] = Field(default=None, ge=1)
    reranked_rank: Optional[int] = Field(default=None, ge=1)
    rank_delta: Optional[int] = None


def compute_required_evidence_rank_movement(
    required_evidence_ids: list[str],
    original_candidates: list[RetrievedEvidence],
    reranked_candidates: list[RerankedEvidence],
) -> list[RequiredEvidenceRankMovement]:
    """For each required evidence id, reports its rank before (in
    ``original_candidates``, the pre-rerank retrieval order) and after (in
    ``reranked_candidates``), and the movement between them.

    ``rank_delta`` is ``original_rank - reranked_rank``: positive means the
    item moved toward the front (an improvement), negative means it moved
    back. Only computed when the item is present in BOTH lists -- an item
    candidate generation never supplied, or one the reranker's own output
    dropped, gets ``None`` here rather than a fabricated number (see
    ``RequiredEvidenceRankMovement``'s docstring for why those two ``None``
    cases are still distinguishable from each other via ``in_candidate_set``).
    """

    original_rank_by_id = {c.evidence_id: c.rank for c in original_candidates}
    reranked_rank_by_id = {c.evidence_id: c.reranked_rank for c in reranked_candidates}

    movements = []
    for evidence_id in required_evidence_ids:
        original_rank = original_rank_by_id.get(evidence_id)
        in_candidate_set = original_rank is not None
        reranked_rank = reranked_rank_by_id.get(evidence_id) if in_candidate_set else None
        rank_delta = (
            original_rank - reranked_rank
            if original_rank is not None and reranked_rank is not None
            else None
        )
        movements.append(
            RequiredEvidenceRankMovement(
                evidence_id=evidence_id,
                in_candidate_set=in_candidate_set,
                original_rank=original_rank,
                reranked_rank=reranked_rank,
                rank_delta=rank_delta,
            )
        )
    return movements
