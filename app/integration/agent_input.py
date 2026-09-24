"""The Phase 8A agent-integration boundary: reranked evidence -> existing
AgentInput -> existing AgentRunner.

Mirrors ``app.retrieval.integration`` exactly, one layer up the pipeline:
that module resolves a ``RetrievalResult``'s candidates back into full
``Evidence`` records; this module does the same for a reranking result's
``RerankedEvidence`` candidates, ordered by ``reranked_rank`` instead of
``rank``. Both ultimately call the same retrieval-context constructor,
``app.agent.interface.build_retrieved_agent_input`` -- never
``build_agent_input`` (the oracle constructor), since reranked evidence,
like retrieved evidence, may legitimately differ from a case's own
``evidence_pool``.
"""

from __future__ import annotations

from app.agent.interface import AgentInput, build_retrieved_agent_input
from app.models.contracts import Control, Evidence, EvaluationCase
from app.reranking.contracts import RerankedEvidence
from app.retrieval.corpus import EvidenceCorpus


def select_reranked_evidence_for_agent(
    candidates: list[RerankedEvidence], corpus: EvidenceCorpus
) -> list[Evidence]:
    """Resolve ``candidates`` (by ``evidence_id``, in ``reranked_rank``
    order) back into full ``Evidence`` records from ``corpus``.

    ``candidates`` is expected to already be truncated to the intended
    Top-K (see ``app.integration.source.build_top_k_candidates_by_case``)
    -- this function does not itself truncate or re-rank anything, it
    only hydrates IDs into full records, in the order given.
    """

    evidence_by_id = {evidence.evidence_id: evidence for evidence in corpus.evidence}
    return [
        evidence_by_id[candidate.evidence_id]
        for candidate in sorted(candidates, key=lambda c: c.reranked_rank)
    ]


def build_agent_input_from_reranking(
    case: EvaluationCase, control: Control, candidates: list[RerankedEvidence], corpus: EvidenceCorpus
) -> AgentInput:
    """Convenience composition of ``select_reranked_evidence_for_agent`` +
    ``app.agent.interface.build_retrieved_agent_input`` -- the realistic
    call site ``app.integration.runner.run_reranked_llm_evaluation``
    actually uses: a case's Top-K ``RerankedEvidence`` candidates + the
    corpus they were drawn from, straight to a ready ``AgentInput``.

    Reads only ``case.case_id``, ``case.scenario_description``,
    ``case.period`` (via ``build_retrieved_agent_input``) -- never
    ``case.expected_outcome`` or ``case.failure_mode_tag``.
    """

    selected_evidence = select_reranked_evidence_for_agent(candidates, corpus)
    return build_retrieved_agent_input(case, control, selected_evidence)
