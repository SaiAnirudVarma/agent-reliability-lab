"""Attributes a wrong or weak end-to-end outcome to the pipeline layer
most likely responsible, using only already-produced artifact evidence --
never hidden chain-of-thought, and never private model reasoning. This
module is evaluator-side (like ``app.evaluation.evaluator``): it is the
first point allowed to read ``EvaluationCase.expected_outcome``-derived
values, and has no path back into ``AgentInput`` construction.

Five layers, in the order a failure would actually occur along the
pipeline:

    A. CANDIDATE_GENERATION_MISS -- required evidence was never supplied
       by retrieval at all (absent from every reranked candidate,
       regardless of rank). Structurally impossible for reranking or the
       LLM to have fixed.
    B. RANKING_ISSUE -- required evidence WAS a candidate, but reranking
       placed it below ``evidence_top_k``, so it never reached the LLM.
    C. LLM_DECISION_ISSUE -- every required item was presented and fully
       cited (citation_validity=1.0, required_evidence_recall in {None,
       1.0}), yet the final finding was still wrong -- a synthesis/
       reasoning failure, not a grounding failure.
    D. CITATION_GROUNDING_ISSUE -- required evidence was presented (i.e.
       not an A or B miss for that item) but the agent's own finding did
       not cite it -- ``app.evaluation.evaluator.compute_required_evidence_recall``
       already restricts itself to evidence that was actually available
       to the agent, so a case's ``required_evidence_recall < 1.0`` here
       reflects a real citation gap, not an unreachable item.
    E. MALFORMED_SCHEMA_FAILURE -- the agent produced no valid
       ``AgentFinding`` at all (``schema_valid=False``); no other layer is
       assessable, since there is no finding to grade.

A single case may surface more than one layer (e.g. one required item
missed by candidate generation while another was mis-ranked) except E,
which is always reported alone -- a malformed response has nothing else
to attribute.
"""

from __future__ import annotations

from enum import Enum

from app.models.contracts import EvaluationResult
from app.reranking.contracts import RerankedEvidence


class FailureLayer(str, Enum):
    CANDIDATE_GENERATION_MISS = "candidate_generation_miss"
    RANKING_ISSUE = "ranking_issue"
    LLM_DECISION_ISSUE = "llm_decision_issue"
    CITATION_GROUNDING_ISSUE = "citation_grounding_issue"
    MALFORMED_SCHEMA_FAILURE = "malformed_schema_failure"


def attribute_case_failure_layers(
    required_evidence_ids: list[str],
    reranked_candidates: list[RerankedEvidence],
    evidence_top_k: int,
    evaluation_result: EvaluationResult,
) -> list[FailureLayer]:
    """Classifies one case's outcome into zero or more ``FailureLayer``s.

    ``reranked_candidates`` must be the FULL reranked candidate set for
    this case (e.g. ``RerankResult.candidates``, all ranks) -- not
    truncated to ``evidence_top_k`` -- so a required item ranked just
    below the cutoff is correctly attributed to ``RANKING_ISSUE`` rather
    than indistinguishably lumped in with a true candidate-generation
    miss.

    Returns an empty list when the case has nothing to attribute (the
    finding was correct and every required item, if any, was fully
    grounded) -- i.e. this function only ever reports on a wrong or weak
    outcome, never manufactures a positive finding of its own.
    """

    if not evaluation_result.schema_valid:
        return [FailureLayer.MALFORMED_SCHEMA_FAILURE]

    layers: list[FailureLayer] = []
    reranked_rank_by_id = {c.evidence_id: c.reranked_rank for c in reranked_candidates}

    for evidence_id in required_evidence_ids:
        rank = reranked_rank_by_id.get(evidence_id)
        if rank is None:
            if FailureLayer.CANDIDATE_GENERATION_MISS not in layers:
                layers.append(FailureLayer.CANDIDATE_GENERATION_MISS)
        elif rank > evidence_top_k:
            if FailureLayer.RANKING_ISSUE not in layers:
                layers.append(FailureLayer.RANKING_ISSUE)

    if evaluation_result.required_evidence_recall is not None and evaluation_result.required_evidence_recall < 1.0:
        layers.append(FailureLayer.CITATION_GROUNDING_ISSUE)

    fully_grounded = evaluation_result.citation_validity == 1.0 and (
        evaluation_result.required_evidence_recall is None or evaluation_result.required_evidence_recall == 1.0
    )
    if not evaluation_result.correct_finding and fully_grounded:
        layers.append(FailureLayer.LLM_DECISION_ISSUE)

    return layers
