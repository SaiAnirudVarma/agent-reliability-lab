"""Unit tests for tests.support.fake_reranker.FakeReranker -- plumbing
validation only, no benchmark-specific logic anywhere in this file.
"""

from __future__ import annotations

import pytest

from app.models.contracts import Control
from app.reranking.contracts import RerankInput
from app.retrieval.contracts import RetrievalQuery, RetrievedEvidence
from tests.support.fake_reranker import FakeReranker


def _control() -> Control:
    return Control(control_id="CTRL-X", name="N", description="D", requirement_text="R")


def _query() -> RetrievalQuery:
    return RetrievalQuery(case_id="AC-999", control=_control(), scenario_description="S", period="2025-Q1")


def _candidate(evidence_id, score, rank) -> RetrievedEvidence:
    return RetrievedEvidence(evidence_id=evidence_id, document_id=f"doc_{evidence_id}", score=score, rank=rank, retrieval_method="x")


def _input(*candidates) -> RerankInput:
    return RerankInput(case_id="AC-999", query=_query(), candidates=list(candidates), candidate_depth=len(candidates))


class TestFakeRerankerExactExample:
    def test_c_beats_a_beats_b_reorders_as_expected(self):
        """The exact example from the task: candidate ranks A=1, B=2,
        C=3; fake scores C > A > B; result must be C=1, A=2, B=3."""
        rerank_input = _input(_candidate("A", 0.5, 1), _candidate("B", 0.3, 2), _candidate("C", 0.9, 3))
        reranker = FakeReranker({"A": 2.0, "B": 1.0, "C": 3.0})

        result = reranker.rerank(rerank_input)

        assert [c.evidence_id for c in result.candidates] == ["C", "A", "B"]
        assert [c.reranked_rank for c in result.candidates] == [1, 2, 3]
        # Original positions/scores are preserved as provenance, unchanged.
        by_id = {c.evidence_id: c for c in result.candidates}
        assert by_id["A"].original_rank == 1
        assert by_id["B"].original_rank == 2
        assert by_id["C"].original_rank == 3


class TestFakeRerankerCandidateSetSemantics:
    def test_cannot_recover_evidence_absent_from_candidate_set(self):
        """A required item never supplied to the reranker cannot appear
        in its output no matter what scores are configured -- there is
        structurally nothing for it to promote."""
        rerank_input = _input(_candidate("A", 0.5, 1), _candidate("B", 0.3, 2))
        reranker = FakeReranker({"A": 1.0, "B": 2.0})
        result = reranker.rerank(rerank_input)
        output_ids = {c.evidence_id for c in result.candidates}
        assert "MISSING-EVIDENCE-NEVER-IN-CANDIDATE-SET" not in output_ids

    def test_unknown_evidence_id_score_raises(self):
        rerank_input = _input(_candidate("A", 0.5, 1))
        reranker = FakeReranker({"SOME-OTHER-ID": 1.0})
        with pytest.raises(KeyError):
            reranker.rerank(rerank_input)

    def test_final_k_trims_after_reordering(self):
        rerank_input = _input(_candidate("A", 0.5, 1), _candidate("B", 0.3, 2), _candidate("C", 0.9, 3))
        reranker = FakeReranker({"A": 2.0, "B": 1.0, "C": 3.0})
        result = reranker.rerank(rerank_input, final_k=2)
        assert [c.evidence_id for c in result.candidates] == ["C", "A"]

    def test_tie_breaks_by_evidence_id_ascending(self):
        rerank_input = _input(_candidate("Z", 0.1, 1), _candidate("A", 0.2, 2))
        reranker = FakeReranker({"Z": 1.0, "A": 1.0})  # tied scores
        result = reranker.rerank(rerank_input)
        assert [c.evidence_id for c in result.candidates] == ["A", "Z"]

    def test_invalid_final_k_raises(self):
        rerank_input = _input(_candidate("A", 0.5, 1))
        reranker = FakeReranker({"A": 1.0})
        with pytest.raises(ValueError, match="final_k"):
            reranker.rerank(rerank_input, final_k=-1)
