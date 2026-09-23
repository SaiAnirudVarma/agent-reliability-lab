"""Retrieval-facing contracts: the query a retriever may see, and the
result it returns.

``RetrievalQuery`` is the retrieval analogue of ``app.agent.interface.AgentInput``
-- the ONLY view of an ``EvaluationCase`` a ``Retriever`` implementation is
allowed to see. It structurally excludes every ground-truth field
(``ExpectedOutcome``, ``required_evidence_ids``, ``should_abstain``,
``rationale``, ``failure_mode_tag``) the same way ``AgentInput`` does: with
``extra="forbid"``, a caller cannot smuggle a forbidden field in even by
accident. It deliberately does NOT carry ``evidence`` -- unlike
``AgentInput``, a retrieval query is a request FOR evidence, not a
container that already holds it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.contracts import Control, EvaluationCase


class RetrievalQuery(BaseModel):
    """Everything -- and only what -- a retriever may legitimately use to
    search for evidence.

    ``case_id`` is opaque tracing metadata ONLY: it exists so a
    ``RetrievalResult``/trace can be tied back to the case it was produced
    for, never so a retriever's matching logic can branch on it (see
    ``LexicalBaselineRetriever``, which never reads ``query.case_id`` when
    scoring).
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    control: Control
    scenario_description: str = Field(min_length=1)
    period: str = Field(pattern=r"^\d{4}-Q[1-4]$")


def build_retrieval_query(case: EvaluationCase, control: Control) -> RetrievalQuery:
    """Project an ``EvaluationCase`` down to the safe retrieval-facing
    subset. Mirrors ``app.agent.interface.build_agent_input``: reads only
    ``case.case_id``, ``case.scenario_description``, ``case.period`` --
    never ``case.expected_outcome`` or ``case.failure_mode_tag`` -- and
    takes ``control`` already resolved by the caller, so this function has
    no dependency on the dataset loader and cannot itself go looking for
    ground truth.
    """

    if control.control_id != case.control_id:
        raise ValueError(
            f"control_id mismatch: case {case.case_id} references "
            f"{case.control_id!r} but was given control {control.control_id!r}"
        )

    return RetrievalQuery(
        case_id=case.case_id,
        control=control,
        scenario_description=case.scenario_description,
        period=case.period,
    )


class RetrievedEvidence(BaseModel):
    """One ranked candidate returned by a retriever.

    ``retrieval_method`` is a plain string, not an enum, matching
    ``AgentRunner.agent_config_id``'s precedent elsewhere in this codebase:
    it identifies which underlying method produced THIS candidate
    specifically (relevant once a future hybrid or reranking pipeline mixes
    candidates from more than one method within a single result) rather
    than being a small closed set decided now. It carries no relevance
    label -- retrieval and evaluation are deliberately kept separate (see
    app.retrieval.metrics).
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    score: float
    rank: int = Field(ge=1)
    retrieval_method: str = Field(min_length=1)


class RetrievalResult(BaseModel):
    """The full, persistable outcome of one retrieval call for one case.

    This doubles as the retrieval "trace": every field Phase 7's later
    persistence work needs (case ID, retriever config, top K, ranked
    evidence/document IDs, scores, method) is already here, in a shape with
    no evaluator-only ground-truth field anywhere in it -- see
    ``docs/RETRIEVAL_DESIGN.md``. Timing/latency provenance (mirroring
    ``ExecutionTrace.latency_ms``) is intentionally deferred to whenever
    Phase 7B actually wires a ``Retriever`` into the traced pipeline; adding
    it now would be speculative.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    query: RetrievalQuery
    candidates: list[RetrievedEvidence]
    top_k: int = Field(ge=1)
    retriever_config_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _case_id_matches_query(self) -> "RetrievalResult":
        if self.case_id != self.query.case_id:
            raise ValueError(
                f"case_id {self.case_id!r} does not match query.case_id {self.query.case_id!r}"
            )
        return self

    @model_validator(mode="after")
    def _candidates_do_not_exceed_top_k(self) -> "RetrievalResult":
        if len(self.candidates) > self.top_k:
            raise ValueError(
                f"{len(self.candidates)} candidates exceeds top_k={self.top_k}"
            )
        return self

    @model_validator(mode="after")
    def _ranks_are_dense_sequential_and_ordered(self) -> "RetrievalResult":
        ranks = [c.rank for c in self.candidates]
        expected = list(range(1, len(ranks) + 1))
        if ranks != expected:
            raise ValueError(
                f"candidates must be in rank order 1..N with no gaps/duplicates/reordering; got {ranks}"
            )
        return self

    @model_validator(mode="after")
    def _no_duplicate_evidence_ids(self) -> "RetrievalResult":
        evidence_ids = [c.evidence_id for c in self.candidates]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("candidates must not cite the same evidence_id more than once")
        return self
