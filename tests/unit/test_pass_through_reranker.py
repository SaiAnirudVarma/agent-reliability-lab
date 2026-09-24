"""Unit tests for app.reranking.pass_through.PassThroughReranker."""

from __future__ import annotations

import pytest

from app.models.contracts import Control
from app.reranking.interface import Reranker
from app.reranking.pass_through import RERANKER_CONFIG_ID, PassThroughReranker
from app.retrieval.contracts import RetrievalQuery, RetrievedEvidence
from app.reranking.contracts import RerankInput


def _control() -> Control:
    return Control(control_id="CTRL-X", name="N", description="D", requirement_text="R")


def _query() -> RetrievalQuery:
    return RetrievalQuery(case_id="AC-999", control=_control(), scenario_description="S", period="2025-Q1")


def _candidate(evidence_id, score, rank) -> RetrievedEvidence:
    return RetrievedEvidence(evidence_id=evidence_id, document_id=f"doc_{evidence_id}", score=score, rank=rank, retrieval_method="x")


def _input(*candidates) -> RerankInput:
    return RerankInput(case_id="AC-999", query=_query(), candidates=list(candidates), candidate_depth=len(candidates))


class TestPassThroughIdentity:
    def test_order_is_preserved_exactly(self):
        rerank_input = _input(_candidate("A", 0.9, 1), _candidate("B", 0.5, 2), _candidate("C", 0.1, 3))
        result = PassThroughReranker().rerank(rerank_input)
        assert [c.evidence_id for c in result.candidates] == ["A", "B", "C"]

    def test_scores_are_unchanged(self):
        rerank_input = _input(_candidate("A", 0.9, 1), _candidate("B", 0.5, 2))
        result = PassThroughReranker().rerank(rerank_input)
        assert result.candidates[0].reranker_score == 0.9
        assert result.candidates[1].reranker_score == 0.5

    def test_original_and_reranked_rank_are_identical_when_untrimmed(self):
        rerank_input = _input(_candidate("A", 0.9, 1), _candidate("B", 0.5, 2))
        result = PassThroughReranker().rerank(rerank_input)
        for candidate in result.candidates:
            assert candidate.original_rank == candidate.reranked_rank

    def test_produces_zero_metric_delta(self):
        """The explicit control-condition requirement: pass-through must
        not change Recall@K or MRR at all relative to the original
        retrieval order."""
        from app.reranking.metrics import compute_delta
        from app.reranking.metrics import mrr as reranked_mrr
        from app.retrieval.metrics import mrr as original_mrr

        rerank_input = _input(_candidate("A", 0.9, 1), _candidate("B", 0.5, 2), _candidate("C", 0.1, 3))
        result = PassThroughReranker().rerank(rerank_input)

        required = ["B"]
        before = original_mrr(required, rerank_input.candidates)
        after = reranked_mrr(required, result.candidates)
        assert compute_delta(before, after) == 0.0

    def test_final_k_trims_without_reordering(self):
        rerank_input = _input(_candidate("A", 0.9, 1), _candidate("B", 0.5, 2), _candidate("C", 0.1, 3))
        result = PassThroughReranker().rerank(rerank_input, final_k=2)
        assert [c.evidence_id for c in result.candidates] == ["A", "B"]

    def test_invalid_final_k_raises(self):
        rerank_input = _input(_candidate("A", 0.9, 1))
        with pytest.raises(ValueError, match="final_k"):
            PassThroughReranker().rerank(rerank_input, final_k=0)

    def test_reranker_method_is_the_documented_constant(self):
        rerank_input = _input(_candidate("A", 0.9, 1))
        result = PassThroughReranker().rerank(rerank_input)
        assert result.reranker_config_id == RERANKER_CONFIG_ID == "pass-through-v1"
        assert result.candidates[0].reranker_method == RERANKER_CONFIG_ID

    def test_satisfies_reranker_protocol(self):
        assert isinstance(PassThroughReranker(), Reranker)
