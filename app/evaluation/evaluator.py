"""Deterministic evaluation of one AgentFinding against its EvaluationCase.

``evaluate_finding`` is the first point in the whole pipeline that is
allowed to read ``EvaluationCase.expected_outcome``. Its signature enforces
the dependency direction visibly: it takes an already-produced
``AgentFinding`` as an argument — there is no path from this module back
into ``AgentInput`` or ``AgentRunner``, so ground truth cannot leak forward
into agent execution even by accident.
"""

from __future__ import annotations

from typing import Optional

from app.models.contracts import AgentFinding, EvaluationCase, EvaluationResult, Evidence, Finding


def compute_citation_validity(
    cited_document_ids: list[str], available_document_ids: set[str], finding: Finding
) -> float:
    """Phase 1 citation validity: did the agent cite documents that were
    actually available to it?

        citation_validity = |unique cited IDs that ARE available| / |unique cited IDs|

    This answers only an *existence* question. It does NOT mean "the cited
    evidence semantically supports the claim" — that is a distinct,
    deliberately unimplemented metric, ``citation_support`` (see the
    roadmap), which would require model-based or otherwise semantic
    evaluation. Duplicate citations to the same document collapse to one
    entry on both sides of the ratio before it is computed, so citing the
    same valid document five times scores identically to citing it once.

    Zero-citation rule
    ------------------
    With zero unique citations the ratio above is undefined (0/0), so a
    rule is defined explicitly rather than left to fall out of arithmetic:

    - ``finding == INSUFFICIENT_EVIDENCE`` (an abstention) -> ``1.0``.
      An abstention is a claim of *absence* of sufficient grounding, not a
      claim resting on evidence. There is nothing false to have cited, so
      the ratio is vacuously valid — the same logic by which precision is
      conventionally undefined-but-harmless when nothing was predicted.
    - ``PASS`` or ``EXCEPTION`` (an affirmative conclusion) -> ``0.0``.
      An affirmative conclusion inherently claims to rest on evidence.
      Reaching one while citing nothing at all is exactly the "unsupported
      conclusion" failure mode and must be scored as a hard grounding
      failure, not treated as if there were nothing to criticize.

    A more elaborate alternative was considered and rejected: making the
    abstention case's score conditional on "was there an observable error
    record the agent could have cited instead." That would require this
    function to independently re-derive what evidence *should* have been
    cited for a good abstention — which is precisely the kind of ground-
    truth-adjacent judgment (`ExpectedOutcome.required_evidence_ids`-like)
    that citation validity must stay clear of, and it would also couple the
    evaluator to agent-specific heuristics (e.g. the baseline's own
    unavailability detector) that a future LLM agent has no obligation to
    replicate. Whether an abstention *should* have cited something is left
    to a future, explicitly ground-truth-aware metric (e.g. a
    required-evidence-recall metric), not to citation_validity.
    """

    unique_cited = set(cited_document_ids)
    if not unique_cited:
        return 1.0 if finding == Finding.INSUFFICIENT_EVIDENCE else 0.0
    valid_cited = unique_cited & available_document_ids
    return len(valid_cited) / len(unique_cited)


def compute_required_evidence_recall(
    required_evidence_ids: list[str],
    cited_document_ids: list[str],
    available_evidence: list[Evidence],
) -> Optional[float]:
    """Phase 6: did the agent cite the evidence that was genuinely necessary
    for the ground-truth decision?

        required_evidence_recall =
            |required evidence items actually cited| / |required evidence items|

    ``ExpectedOutcome.required_evidence_ids`` lives in evidence_id space
    (the dataset's internal primary key, e.g. "EV-201"); ``AgentFinding``
    citations live in document_id space (the business identifier an agent
    actually sees and cites, e.g. "priv_access_review_attestation_2025q2").
    This function maps explicitly between the two via ``available_evidence``
    (the same Evidence records the agent saw) -- it never compares an
    evidence_id to a document_id directly, since those are different ID
    spaces that could coincidentally collide.

    This is a distinct question from ``compute_citation_validity``: validity
    asks "were the agent's citations real," recall asks "did the agent cite
    what ground truth says was necessary." A finding can cite only real,
    valid documents (validity=1.0) while still missing the one that
    actually mattered (recall<1.0), and vice versa.

    Zero-denominator convention
    ----------------------------
    Returns ``None`` -- not ``0.0`` -- when ``required_evidence_ids`` is
    empty: nothing was required for this case's ground truth, so recall is
    not applicable, not a measured failure. Callers aggregating this metric
    across a run must treat ``None`` as "exclude this case from the mean,"
    matching the language of the metric itself ("mean ... across applicable
    cases") -- this is a deliberately different convention from every other
    metric in this codebase, which defaults an undefined ratio to ``0.0``.
    The difference is semantic: an undefined *rate* (like precision with no
    predictions) is conservatively scored 0.0, but "not applicable" is not
    a rate at all.
    """

    if not required_evidence_ids:
        return None

    evidence_id_to_document_id = {ev.evidence_id: ev.document_id for ev in available_evidence}
    required_document_ids = {
        evidence_id_to_document_id[eid]
        for eid in required_evidence_ids
        if eid in evidence_id_to_document_id
    }
    if not required_document_ids:
        # Should not happen given EvaluationCase's own validator enforces
        # required_evidence_ids ⊆ evidence_pool and available_evidence is
        # exactly evidence_for_case(case) -- guarded defensively rather than
        # raising KeyError if ever called with a mismatched pair.
        return None

    cited = set(cited_document_ids)
    hit = required_document_ids & cited
    return len(hit) / len(required_document_ids)


def evaluate_finding(
    case: EvaluationCase,
    available_evidence: list[Evidence],
    finding: Optional[AgentFinding],
    run_id: str,
) -> EvaluationResult:
    """Score one already-produced agent outcome against case.expected_outcome.

    ``available_evidence`` must be the same evidence the agent actually saw
    for this case (i.e. what its ``AgentInput.evidence`` contained) —
    citation validity is defined relative to what was available, never
    relative to ``ExpectedOutcome.required_evidence_ids``.

    ``finding`` is ``None`` exactly when the agent raised ``AgentOutputError``
    (i.e. ``ExecutionTrace.output`` was ``None``) — it received a response
    but could not produce a schema-valid ``AgentFinding``. In that case:

    - ``correct_finding`` is ``False`` — there is nothing to have gotten right.
    - ``schema_valid`` is ``False``.
    - ``citation_validity`` is ``0.0`` — no valid output is a harder
      grounding failure than a real finding that cited nothing.
    - ``abstention_correct`` is ``None`` — genuinely undefined, since there
      is no ``abstain`` flag to compare against ground truth.

    When ``finding`` IS present, ``schema_valid`` is trivially ``True``: by
    the time a caller has an ``AgentFinding`` instance at all, it has
    already passed Pydantic validation (Component 1). This field is real
    plumbing, not a placeholder — it only starts discriminating once an
    agent can actually fail to produce one, which is exactly the ``None``
    branch above.
    """

    expected = case.expected_outcome

    if finding is None:
        # No citations exist to compute recall from; still None (not 0.0)
        # when nothing was required, per compute_required_evidence_recall's
        # documented convention -- applicability is a property of the
        # case's ground truth, independent of what the agent did.
        no_output_recall = 0.0 if expected.required_evidence_ids else None
        return EvaluationResult(
            case_id=case.case_id,
            run_id=run_id,
            expected=expected,
            actual=None,
            correct_finding=False,
            schema_valid=False,
            citation_validity=0.0,
            required_evidence_recall=no_output_recall,
            abstention_correct=None,
        )

    available_document_ids = {evidence.document_id for evidence in available_evidence}
    cited_document_ids = [reference.document_id for reference in finding.evidence]

    return EvaluationResult(
        case_id=case.case_id,
        run_id=run_id,
        expected=expected,
        actual=finding,
        correct_finding=finding.finding == expected.expected_finding,
        schema_valid=True,
        required_evidence_recall=compute_required_evidence_recall(
            expected.required_evidence_ids, cited_document_ids, available_evidence
        ),
        citation_validity=compute_citation_validity(
            cited_document_ids, available_document_ids, finding.finding
        ),
        abstention_correct=finding.abstain == expected.should_abstain,
    )
