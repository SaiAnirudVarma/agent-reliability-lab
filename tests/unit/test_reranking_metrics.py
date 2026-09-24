"""Unit tests for app.reranking.metrics: Recall@K/MRR after reranking,
before/after deltas, and required-evidence rank movement -- including the
critical distinction between a candidate-generation failure and a
reranking-stage failure.
"""

from __future__ import annotations

import pytest

from app.models.contracts import Control
from app.reranking.contracts import RerankedEvidence, RerankInput
from app.reranking.metrics import (
    aggregate_mrr,
    aggregate_recall_at_k,
    compute_delta,
    compute_required_evidence_rank_movement,
    mrr,
    recall_at_k,
)
from app.retrieval.contracts import RetrievalQuery, RetrievedEvidence
from tests.support.fake_reranker import FakeReranker


def _control() -> Control:
    return Control(control_id="CTRL-X", name="N", description="D", requirement_text="R")


def _query() -> RetrievalQuery:
    return RetrievalQuery(case_id="AC-999", control=_control(), scenario_description="S", period="2025-Q1")


def _orig(evidence_id, score, rank) -> RetrievedEvidence:
    return RetrievedEvidence(evidence_id=evidence_id, document_id=f"doc_{evidence_id}", score=score, rank=rank, retrieval_method="x")


def _reranked_cand(evidence_id, reranked_rank, original_rank=None, score=1.0) -> RerankedEvidence:
    return RerankedEvidence(
        evidence_id=evidence_id, document_id=f"doc_{evidence_id}", original_rank=original_rank or reranked_rank,
        reranked_rank=reranked_rank, original_score=score, reranker_score=score, reranker_method="x",
    )


def _input(*candidates) -> RerankInput:
    return RerankInput(case_id="AC-999", query=_query(), candidates=list(candidates), candidate_depth=len(candidates))


class TestRecallAtKAfterReranking:
    def test_found_within_k(self):
        candidates = [_reranked_cand("EV-1", 1), _reranked_cand("EV-2", 2)]
        assert recall_at_k(["EV-2"], candidates, k=2) == 1.0

    def test_outside_k_is_zero(self):
        candidates = [_reranked_cand("EV-1", 1), _reranked_cand("EV-2", 2)]
        assert recall_at_k(["EV-2"], candidates, k=1) == 0.0

    def test_zero_required_is_none(self):
        candidates = [_reranked_cand("EV-1", 1)]
        assert recall_at_k([], candidates, k=1) is None


class TestMrrAfterReranking:
    def test_earliest_relevant_rank(self):
        candidates = [_reranked_cand("EV-9", 1), _reranked_cand("EV-1", 2)]
        assert mrr(["EV-1"], candidates) == 0.5

    def test_zero_when_absent(self):
        candidates = [_reranked_cand("EV-9", 1)]
        assert mrr(["EV-1"], candidates) == 0.0

    def test_none_when_nothing_required(self):
        candidates = [_reranked_cand("EV-1", 1)]
        assert mrr([], candidates) is None


class TestAggregates:
    def test_aggregate_recall_excludes_none(self):
        assert aggregate_recall_at_k([1.0, 0.5, None]) == pytest.approx(0.75)

    def test_aggregate_mrr_excludes_none(self):
        assert aggregate_mrr([1.0, 0.0, None]) == pytest.approx(0.5)

    def test_aggregate_zero_applicable_is_zero(self):
        assert aggregate_recall_at_k([None, None]) == 0.0
        assert aggregate_mrr([None]) == 0.0


class TestComputeDelta:
    def test_improvement_is_positive(self):
        assert compute_delta(0.5, 0.8) == pytest.approx(0.3)

    def test_regression_is_negative(self):
        assert compute_delta(0.8, 0.5) == pytest.approx(-0.3)

    def test_none_before_propagates_none(self):
        assert compute_delta(None, 0.8) is None

    def test_none_after_propagates_none(self):
        assert compute_delta(0.5, None) is None

    def test_zero_delta_for_identical_values(self):
        assert compute_delta(0.5, 0.5) == 0.0


class TestRequiredEvidenceRankMovement:
    def test_moved_upward_by_fake_reranker(self):
        """Demonstrates a reranker CAN move relevant evidence upward when
        it explicitly assigns a higher score -- exactly the plumbing
        proof the task calls for."""
        original = [_orig("A", 0.5, 1), _orig("REQUIRED", 0.1, 2)]
        rerank_input = _input(*original)
        reranker = FakeReranker({"A": 1.0, "REQUIRED": 5.0})  # REQUIRED explicitly boosted
        result = reranker.rerank(rerank_input)

        movements = compute_required_evidence_rank_movement(["REQUIRED"], original, result.candidates)
        assert len(movements) == 1
        movement = movements[0]
        assert movement.in_candidate_set is True
        assert movement.original_rank == 2
        assert movement.reranked_rank == 1
        assert movement.rank_delta == 1  # moved from rank 2 to rank 1: 2 - 1 = 1

    def test_not_in_candidate_set_is_a_candidate_generation_failure(self):
        """A required item candidate generation never supplied is
        distinguishable from -- and never penalizes -- the reranker."""
        original = [_orig("A", 0.5, 1)]
        rerank_input = _input(*original)
        reranker = FakeReranker({"A": 1.0})
        result = reranker.rerank(rerank_input)

        movements = compute_required_evidence_rank_movement(["NEVER-RETRIEVED"], original, result.candidates)
        movement = movements[0]
        assert movement.in_candidate_set is False
        assert movement.original_rank is None
        assert movement.reranked_rank is None
        assert movement.rank_delta is None  # never penalizes the reranker for this

    def test_dropped_by_reranker_trimming_is_distinguishable_from_candidate_generation_failure(self):
        """Present in the candidate set but trimmed off by final_k --
        a genuine reranking-stage outcome, NOT the same as never having
        been retrieved at all (in_candidate_set differs between the two)."""
        original = [_orig("A", 0.9, 1), _orig("REQUIRED", 0.1, 2)]
        rerank_input = _input(*original)
        reranker = FakeReranker({"A": 5.0, "REQUIRED": 1.0})  # REQUIRED stays low-scored
        result = reranker.rerank(rerank_input, final_k=1)  # only "A" survives

        movements = compute_required_evidence_rank_movement(["REQUIRED"], original, result.candidates)
        movement = movements[0]
        assert movement.in_candidate_set is True  # it WAS supplied to the reranker
        assert movement.original_rank == 2
        assert movement.reranked_rank is None  # but dropped by the reranker's own trimming
        assert movement.rank_delta is None  # not fabricated

    def test_end_to_end_recall_still_reflects_missing_required_evidence(self):
        """Even though rank movement doesn't penalize the reranker for an
        absent item, the headline Recall@K metric computed from the
        reranked output must still correctly score it as missing."""
        original = [_orig("A", 0.5, 1)]
        rerank_input = _input(*original)
        result = FakeReranker({"A": 1.0}).rerank(rerank_input)
        assert recall_at_k(["NEVER-RETRIEVED"], result.candidates, k=10) == 0.0

    def test_multiple_required_items_mixed_outcomes(self):
        original = [_orig("FOUND", 0.9, 1), _orig("A", 0.1, 2)]
        rerank_input = _input(*original)
        result = FakeReranker({"FOUND": 5.0, "A": 1.0}).rerank(rerank_input)

        movements = compute_required_evidence_rank_movement(["FOUND", "NEVER-SUPPLIED"], original, result.candidates)
        by_id = {m.evidence_id: m for m in movements}
        assert by_id["FOUND"].in_candidate_set is True
        assert by_id["NEVER-SUPPLIED"].in_candidate_set is False
