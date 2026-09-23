"""Retrieval-quality metrics, evaluated independently of agent finding
metrics (``app.evaluation.metrics``) -- these answer "did retrieval surface
the right evidence," never "did the agent reach the right conclusion."

Both per-case functions here read ``ExpectedOutcome.required_evidence_ids``,
so -- exactly like ``app.evaluation.evaluator`` -- they are only ever called
AFTER a ``RetrievalResult`` already exists, never consulted by any
``Retriever`` implementation itself (see ``app.retrieval.interface``).

Both use ``evidence_id`` identity (not ``document_id``), since
``required_evidence_ids`` is itself in evidence_id space -- the same
convention documented in
``app.evaluation.evaluator.compute_required_evidence_recall``.
"""

from __future__ import annotations

from typing import Optional

from app.retrieval.contracts import RetrievedEvidence


def recall_at_k(
    required_evidence_ids: list[str], candidates: list[RetrievedEvidence], k: int
) -> Optional[float]:
    """Recall@K for one case:

        recall_at_k =
            |unique required evidence_ids present among candidates with rank <= k|
            ----------------------------------------------------------------------
            |unique required evidence_ids|

    Returns ``None`` -- not ``0.0`` -- when ``required_evidence_ids`` is
    empty: nothing was required for this case's ground truth, so Recall@K
    is not applicable, not a measured failure. This mirrors
    ``compute_required_evidence_recall``'s documented convention exactly.
    """

    if not required_evidence_ids:
        return None
    required = set(required_evidence_ids)
    retrieved_within_k = {c.evidence_id for c in candidates if c.rank <= k}
    hit = required & retrieved_within_k
    return len(hit) / len(required)


def mrr(required_evidence_ids: list[str], candidates: list[RetrievedEvidence]) -> Optional[float]:
    """MRR for one case, under this project's explicit convention (MRR has
    no single universal definition once a case can have MORE THAN ONE
    required/relevant item):

        MRR = 1 / rank_of_the_FIRST-ranked candidate whose evidence_id is
              in required_evidence_ids

    Only the earliest-ranked relevant hit counts -- classic single-relevant
    -item MRR, extended to "first relevant among possibly several relevant"
    rather than averaged over every required item's individual rank (that
    would be a different metric -- mean reciprocal rank of ALL required
    items -- not MRR, and is deliberately not implemented here).

    If NO required item is retrieved at any rank, MRR = ``0.0`` -- unlike
    Recall@K's empty-numerator case, this is a real, measured retrieval
    failure for an applicable case, not "not applicable".

    Returns ``None`` when ``required_evidence_ids`` is empty: there is no
    "first relevant rank" to find, so MRR itself is not applicable (same
    empty-required convention as ``recall_at_k``).
    """

    if not required_evidence_ids:
        return None
    required = set(required_evidence_ids)
    for candidate in sorted(candidates, key=lambda c: c.rank):
        if candidate.evidence_id in required:
            return 1.0 / candidate.rank
    return 0.0


def aggregate_recall_at_k(values: list[Optional[float]]) -> float:
    """Mean of per-case ``recall_at_k`` values, over only the cases where
    it applies (``None`` values -- empty ``required_evidence_ids`` --
    excluded from the denominator entirely, never folded in as 0.0).

    ``0.0`` if there are ZERO applicable cases -- the same empty-denominator
    convention as ``app.evaluation.metrics.aggregate_required_evidence_recall``.
    """

    applicable = [v for v in values if v is not None]
    if not applicable:
        return 0.0
    return sum(applicable) / len(applicable)


def aggregate_mrr(values: list[Optional[float]]) -> float:
    """Mean of per-case ``mrr`` values, over only applicable (non-``None``)
    cases. ``0.0`` if there are zero applicable cases. Same convention as
    ``aggregate_recall_at_k`` -- documented separately since MRR's ``None``
    vs. ``0.0`` distinction (see ``mrr``'s docstring) is easy to conflate
    with Recall@K's, despite both aggregating the same way once computed.
    """

    applicable = [v for v in values if v is not None]
    if not applicable:
        return 0.0
    return sum(applicable) / len(applicable)
