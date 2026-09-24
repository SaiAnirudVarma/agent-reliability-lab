"""Reranking contracts: the input a Reranker may see, and the result it
returns.

``RerankInput`` reuses ``app.retrieval.contracts.RetrievalQuery`` and
``RetrievedEvidence`` directly rather than redefining equivalent types --
the candidate set a reranker receives IS a retriever's output, in its
original order, with nothing added or removed by this layer. Like
``RetrievalQuery``, ``RerankInput`` carries no ``ExpectedOutcome``-shaped
field, and ``extra="forbid"`` means one cannot be smuggled in.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.retrieval.contracts import RetrievalQuery, RetrievedEvidence


class RerankInput(BaseModel):
    """One case's candidate set, in original retrieval order, handed to a
    Reranker. ``candidate_depth`` is the exact size of that candidate set
    (``len(candidates)``) -- recorded explicitly, not merely implied by the
    list length, so a future experiment can assert "this reranker saw a
    Top-N candidate set" without recomputing it, and so a mismatch between
    a caller's intended depth and what was actually assembled is caught by
    validation rather than silently accepted.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    query: RetrievalQuery
    candidates: list[RetrievedEvidence]
    candidate_depth: int = Field(ge=1)

    @model_validator(mode="after")
    def _case_id_matches_query(self) -> "RerankInput":
        if self.case_id != self.query.case_id:
            raise ValueError(
                f"case_id {self.case_id!r} does not match query.case_id {self.query.case_id!r}"
            )
        return self

    @model_validator(mode="after")
    def _candidate_depth_matches_actual_count(self) -> "RerankInput":
        if self.candidate_depth != len(self.candidates):
            raise ValueError(
                f"candidate_depth={self.candidate_depth} does not match len(candidates)={len(self.candidates)}"
            )
        return self

    @model_validator(mode="after")
    def _candidates_ranks_dense_sequential_and_ordered(self) -> "RerankInput":
        ranks = [c.rank for c in self.candidates]
        expected = list(range(1, len(ranks) + 1))
        if ranks != expected:
            raise ValueError(
                f"candidates must be in rank order 1..N with no gaps/duplicates/reordering; got {ranks}"
            )
        return self

    @model_validator(mode="after")
    def _no_duplicate_evidence_ids(self) -> "RerankInput":
        evidence_ids = [c.evidence_id for c in self.candidates]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("candidates must not contain the same evidence_id more than once")
        return self


class RerankedEvidence(BaseModel):
    """One candidate after reranking, with BOTH its original and its new
    position/score retained -- ``original_rank``/``original_score`` are
    provenance copied verbatim from the ``RetrievedEvidence`` this came
    from (see ``RerankResult``'s own validator, which checks this), never
    recomputed or guessed.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    original_rank: int = Field(ge=1)
    reranked_rank: int = Field(ge=1)
    original_score: float
    reranker_score: float
    reranker_method: str = Field(min_length=1)


class RerankResult(BaseModel):
    """The full, persistable outcome of reranking one case's candidate set.

    ``input`` is the exact ``RerankInput`` this result was produced from --
    embedded here (mirroring ``RetrievalResult.query``) so a standalone
    ``RerankResult`` is self-checkable: every invariant below is enforced
    structurally, not merely by convention.

    ``candidates`` may be a SUBSET of ``input.candidates`` (a reranker may
    trim to a smaller final K -- e.g. vector Top-10 -> reranker -> final
    Top-5 -- see docs on candidate-set semantics) but must never contain an
    ``evidence_id`` absent from ``input.candidates``: a reranker cannot
    retrieve documents that were never in the candidate set supplied to it.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    input: RerankInput
    candidates: list[RerankedEvidence]
    reranker_config_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _case_id_matches_input(self) -> "RerankResult":
        if self.case_id != self.input.case_id:
            raise ValueError(
                f"case_id {self.case_id!r} does not match input.case_id {self.input.case_id!r}"
            )
        return self

    @model_validator(mode="after")
    def _reranked_ranks_dense_sequential_and_ordered(self) -> "RerankResult":
        ranks = [c.reranked_rank for c in self.candidates]
        expected = list(range(1, len(ranks) + 1))
        if ranks != expected:
            raise ValueError(
                f"candidates must be in reranked_rank order 1..N with no gaps/duplicates/reordering; got {ranks}"
            )
        return self

    @model_validator(mode="after")
    def _no_duplicate_evidence_ids(self) -> "RerankResult":
        evidence_ids = [c.evidence_id for c in self.candidates]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("candidates must not contain the same evidence_id more than once")
        return self

    @model_validator(mode="after")
    def _no_candidate_additions(self) -> "RerankResult":
        input_ids = {c.evidence_id for c in self.input.candidates}
        output_ids = {c.evidence_id for c in self.candidates}
        introduced = output_ids - input_ids
        if introduced:
            raise ValueError(
                f"reranker output introduced evidence_id(s) absent from the input candidate set: "
                f"{sorted(introduced)}"
            )
        return self

    @model_validator(mode="after")
    def _original_rank_and_score_provenance_retained(self) -> "RerankResult":
        input_by_id = {c.evidence_id: c for c in self.input.candidates}
        mismatches = []
        for candidate in self.candidates:
            original = input_by_id.get(candidate.evidence_id)
            if original is None:
                continue  # already reported by _no_candidate_additions
            if candidate.original_rank != original.rank or candidate.original_score != original.score:
                mismatches.append(candidate.evidence_id)
        if mismatches:
            raise ValueError(
                f"original_rank/original_score do not match the input candidate set's recorded "
                f"values for evidence_id(s): {sorted(mismatches)}"
            )
        return self
