"""The Phase 7B agent-integration boundary -- design/demonstration only.

Not called by any pipeline yet (no ``AgentRunner`` is wired to retrieval in
Phase 7A; see ``docs/RETRIEVAL_DESIGN.md``). This module exists so the
transformation

    RetrievalResult -> selected Evidence[] -> existing AgentInput -> existing AgentRunner

is demonstrated in real code (and exercised by
``tests/unit/test_retrieval_agent_integration.py``) rather than only
described in prose.
"""

from __future__ import annotations

from app.models.contracts import Evidence
from app.retrieval.contracts import RetrievalResult
from app.retrieval.corpus import EvidenceCorpus


def select_evidence_for_agent(result: RetrievalResult, corpus: EvidenceCorpus) -> list[Evidence]:
    """Resolve ``result.candidates`` (by ``evidence_id``, in retrieval-rank
    order) back into full ``Evidence`` records from ``corpus``.

    The returned list is exactly the shape ``AgentInput.evidence`` already
    expects (``list[Evidence]``) -- this is the entire integration point.
    Neither ``AgentInput`` nor any ``AgentRunner`` implementation needs to
    change: today, that list happens to come from ``case.evidence_pool``
    via ``Dataset.evidence_for_case``; tomorrow, it can come from here
    instead, and nothing downstream can tell the difference.

    Note this deliberately does NOT reuse
    ``app.agent.interface.build_agent_input`` -- that helper's own
    ``evidence_ids != set(case.evidence_pool)`` check exists specifically
    to guarantee the ORACLE pipeline never silently sees a different
    evidence set than the one the case defines, which is exactly the
    guarantee a real retrieval pipeline must NOT have (retrieval selecting
    a different subset than the oracle pool is the entire point). Phase 7B
    will need its own, separate constructor alongside
    ``build_agent_input`` -- not a change to it -- once real wiring lands.
    """

    evidence_by_id = {evidence.evidence_id: evidence for evidence in corpus.evidence}
    return [
        evidence_by_id[candidate.evidence_id]
        for candidate in sorted(result.candidates, key=lambda c: c.rank)
    ]
