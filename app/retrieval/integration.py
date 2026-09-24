"""The Phase 7B agent-integration boundary.

Not called by any pipeline yet -- no ``AgentRunner`` run in this repository
is currently wired to retrieval output; see ``docs/RETRIEVAL_DESIGN.md``.
This module exists so the transformation

    RetrievalResult -> selected Evidence[] -> existing AgentInput -> existing AgentRunner

is demonstrated (and exercised by
``tests/unit/test_retrieval_agent_integration.py``) in real code, using
the retrieval-context constructor
``app.agent.interface.build_retrieved_agent_input``.
"""

from __future__ import annotations

from app.agent.interface import AgentInput, build_retrieved_agent_input
from app.models.contracts import Control, Evidence, EvaluationCase
from app.retrieval.contracts import RetrievalResult
from app.retrieval.corpus import EvidenceCorpus


def select_evidence_for_agent(result: RetrievalResult, corpus: EvidenceCorpus) -> list[Evidence]:
    """Resolve ``result.candidates`` (by ``evidence_id``, in retrieval-rank
    order) back into full ``Evidence`` records from ``corpus``.

    The returned list is exactly the shape ``AgentInput.evidence`` already
    expects (``list[Evidence]``) -- today, that list happens to come from
    ``case.evidence_pool`` via ``Dataset.evidence_for_case``; with
    retrieval, it comes from here instead, and ``AgentInput`` itself
    cannot tell the difference.

    Note this deliberately does NOT reuse
    ``app.agent.interface.build_agent_input`` -- that helper's own
    ``evidence_ids != set(case.evidence_pool)`` check exists specifically
    to guarantee the ORACLE pipeline never silently sees a different
    evidence set than the one the case defines, which is exactly the
    guarantee a real retrieval pipeline must NOT have (retrieval selecting
    a different subset than the oracle pool is the entire point). See
    ``build_agent_input_from_retrieval`` below, which uses
    ``build_retrieved_agent_input`` -- the separate, retrieval-context
    constructor -- instead.
    """

    evidence_by_id = {evidence.evidence_id: evidence for evidence in corpus.evidence}
    return [
        evidence_by_id[candidate.evidence_id]
        for candidate in sorted(result.candidates, key=lambda c: c.rank)
    ]


def build_agent_input_from_retrieval(
    case: EvaluationCase, control: Control, result: RetrievalResult, corpus: EvidenceCorpus
) -> AgentInput:
    """Convenience composition of ``select_evidence_for_agent`` +
    ``app.agent.interface.build_retrieved_agent_input`` -- the realistic
    call site a future retrieval-integrated pipeline (or
    ``app.retrieval.runner``, if it is later extended to also run an
    agent) would actually use: ``RetrievalResult`` + the corpus it was
    retrieved from, straight to a ready ``AgentInput``.
    """

    selected_evidence = select_evidence_for_agent(result, corpus)
    return build_retrieved_agent_input(case, control, selected_evidence)
