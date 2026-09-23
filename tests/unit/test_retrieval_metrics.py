"""Unit tests for app.retrieval.metrics: Recall@K and MRR, with exact
known examples, N/A conventions, and aggregate behavior.
"""

from __future__ import annotations

import pytest

from app.retrieval.contracts import RetrievedEvidence
from app.retrieval.metrics import aggregate_mrr, aggregate_recall_at_k, mrr, recall_at_k


def _cand(evidence_id: str, rank: int, score: float = 1.0) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=evidence_id, document_id=f"doc_{evidence_id}", score=score, rank=rank,
        retrieval_method="lexical-baseline-v1",
    )


class TestRecallAtKExactExamples:
    def test_single_required_item_found_at_rank_1(self):
        candidates = [_cand("EV-1", 1), _cand("EV-2", 2), _cand("EV-3", 3)]
        assert recall_at_k(["EV-1"], candidates, k=3) == 1.0

    def test_single_required_item_found_at_rank_3_within_k(self):
        candidates = [_cand("EV-2", 1), _cand("EV-3", 2), _cand("EV-1", 3)]
        assert recall_at_k(["EV-1"], candidates, k=3) == 1.0

    def test_single_required_item_outside_k_is_zero(self):
        candidates = [_cand("EV-2", 1), _cand("EV-3", 2), _cand("EV-1", 3)]
        assert recall_at_k(["EV-1"], candidates, k=2) == 0.0

    def test_two_required_items_both_within_k_is_one(self):
        candidates = [_cand("EV-1", 1), _cand("EV-2", 2), _cand("EV-3", 3)]
        assert recall_at_k(["EV-1", "EV-2"], candidates, k=3) == 1.0

    def test_two_required_items_one_within_k_is_half(self):
        candidates = [_cand("EV-1", 1), _cand("EV-3", 2), _cand("EV-2", 3)]
        # required EV-1 (rank 1) and EV-2 (rank 3); k=2 only catches EV-1
        assert recall_at_k(["EV-1", "EV-2"], candidates, k=2) == pytest.approx(0.5)

    def test_no_required_items_retrieved_is_zero(self):
        candidates = [_cand("EV-9", 1), _cand("EV-8", 2)]
        assert recall_at_k(["EV-1"], candidates, k=2) == 0.0

    def test_zero_required_evidence_is_none_not_zero(self):
        candidates = [_cand("EV-1", 1)]
        assert recall_at_k([], candidates, k=5) is None

    def test_no_candidates_at_all_with_required_items_is_zero(self):
        assert recall_at_k(["EV-1"], [], k=5) == 0.0

    def test_duplicate_required_ids_collapse_before_dividing(self):
        candidates = [_cand("EV-1", 1)]
        assert recall_at_k(["EV-1", "EV-1"], candidates, k=1) == 1.0


class TestMrrExactExamples:
    def test_first_relevant_at_rank_1_gives_mrr_1(self):
        candidates = [_cand("EV-1", 1), _cand("EV-2", 2)]
        assert mrr(["EV-1"], candidates) == 1.0

    def test_first_relevant_at_rank_4_gives_mrr_one_quarter(self):
        candidates = [_cand("EV-9", 1), _cand("EV-8", 2), _cand("EV-7", 3), _cand("EV-1", 4)]
        assert mrr(["EV-1"], candidates) == pytest.approx(0.25)

    def test_multiple_required_items_uses_earliest_ranked_hit_only(self):
        """Two required items at ranks 2 and 5 -- MRR uses the EARLIEST
        relevant rank (2), not an average over both required items' ranks."""
        candidates = [_cand("EV-9", 1), _cand("EV-2", 2), _cand("EV-8", 3), _cand("EV-7", 4), _cand("EV-1", 5)]
        assert mrr(["EV-1", "EV-2"], candidates) == pytest.approx(0.5)  # 1/2, NOT mean(1/2, 1/5)

    def test_zero_required_items_retrieved_is_zero_not_none(self):
        candidates = [_cand("EV-9", 1), _cand("EV-8", 2)]
        assert mrr(["EV-1"], candidates) == 0.0

    def test_zero_required_evidence_is_none(self):
        candidates = [_cand("EV-1", 1)]
        assert mrr([], candidates) is None

    def test_no_candidates_with_required_items_is_zero(self):
        assert mrr(["EV-1"], []) == 0.0

    def test_unsorted_candidate_input_still_finds_earliest_rank(self):
        """mrr sorts by rank internally -- input order must not matter."""
        candidates = [_cand("EV-1", 5), _cand("EV-2", 2), _cand("EV-9", 1)]
        assert mrr(["EV-1", "EV-2"], candidates) == pytest.approx(0.5)


class TestAggregateRecallAtK:
    def test_mean_over_applicable_cases_only(self):
        assert aggregate_recall_at_k([1.0, 0.5, None]) == pytest.approx(0.75)

    def test_zero_applicable_cases_is_zero(self):
        assert aggregate_recall_at_k([None, None]) == 0.0

    def test_empty_list_is_zero(self):
        assert aggregate_recall_at_k([]) == 0.0

    def test_all_applicable_perfect(self):
        assert aggregate_recall_at_k([1.0, 1.0, 1.0]) == 1.0


class TestAggregateMrr:
    def test_mean_over_applicable_cases_only(self):
        assert aggregate_mrr([1.0, 0.0, None]) == pytest.approx(0.5)

    def test_zero_applicable_cases_is_zero(self):
        assert aggregate_mrr([None]) == 0.0

    def test_empty_list_is_zero(self):
        assert aggregate_mrr([]) == 0.0
