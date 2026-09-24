"""Unit tests for app.reranking.contracts: RerankInput, RerankedEvidence,
RerankResult -- validation, the no-additions invariant, and provenance
retention.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.contracts import Control
from app.reranking.contracts import RerankedEvidence, RerankInput, RerankResult
from app.retrieval.contracts import RetrievalQuery, RetrievedEvidence


def _control(**overrides) -> Control:
    defaults = dict(control_id="CTRL-X", name="N", description="D", requirement_text="R")
    defaults.update(overrides)
    return Control(**defaults)


def _query(case_id="AC-999") -> RetrievalQuery:
    return RetrievalQuery(case_id=case_id, control=_control(), scenario_description="S", period="2025-Q1")


def _candidate(evidence_id="EV-1", document_id="doc_1", score=1.0, rank=1) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=evidence_id, document_id=document_id, score=score, rank=rank, retrieval_method="x"
    )


def _rerank_input(case_id="AC-999", candidates=None) -> RerankInput:
    candidates = candidates if candidates is not None else [_candidate(rank=1)]
    return RerankInput(case_id=case_id, query=_query(case_id), candidates=candidates, candidate_depth=len(candidates))


def _reranked(evidence_id, document_id, original_rank, reranked_rank, original_score, reranker_score) -> RerankedEvidence:
    return RerankedEvidence(
        evidence_id=evidence_id, document_id=document_id, original_rank=original_rank, reranked_rank=reranked_rank,
        original_score=original_score, reranker_score=reranker_score, reranker_method="x",
    )


class TestRerankInput:
    def test_valid_input_constructs(self):
        rerank_input = _rerank_input()
        assert rerank_input.candidate_depth == 1

    def test_case_id_mismatch_rejected(self):
        with pytest.raises(ValidationError, match="does not match"):
            RerankInput(case_id="AC-OTHER", query=_query("AC-999"), candidates=[_candidate()], candidate_depth=1)

    def test_candidate_depth_mismatch_rejected(self):
        with pytest.raises(ValidationError, match="candidate_depth"):
            RerankInput(case_id="AC-999", query=_query(), candidates=[_candidate(rank=1)], candidate_depth=5)

    def test_non_dense_ranks_rejected(self):
        with pytest.raises(ValidationError, match="rank order"):
            RerankInput(
                case_id="AC-999", query=_query(),
                candidates=[_candidate("EV-1", rank=1), _candidate("EV-2", rank=3)], candidate_depth=2,
            )

    def test_duplicate_evidence_id_rejected(self):
        with pytest.raises(ValidationError, match="more than once"):
            RerankInput(
                case_id="AC-999", query=_query(),
                candidates=[_candidate("EV-1", rank=1), _candidate("EV-1", rank=2)], candidate_depth=2,
            )

    @pytest.mark.parametrize(
        "forbidden_field,value",
        [("expected_finding", "PASS"), ("required_evidence_ids", ["EV-1"]), ("failure_mode_tag", "x"), ("rationale", "x")],
    )
    def test_forbidden_ground_truth_fields_rejected(self, forbidden_field, value):
        kwargs = dict(case_id="AC-999", query=_query(), candidates=[_candidate()], candidate_depth=1)
        kwargs[forbidden_field] = value
        with pytest.raises(ValidationError):
            RerankInput(**kwargs)


class TestRerankResult:
    def test_valid_result_constructs(self):
        rerank_input = _rerank_input()
        result = RerankResult(
            case_id="AC-999", input=rerank_input,
            candidates=[_reranked("EV-1", "doc_1", 1, 1, 1.0, 1.0)],
            reranker_config_id="x",
        )
        assert result.candidates[0].reranked_rank == 1

    def test_case_id_must_match_input(self):
        rerank_input = _rerank_input()
        with pytest.raises(ValidationError, match="does not match"):
            RerankResult(case_id="AC-OTHER", input=rerank_input, candidates=[], reranker_config_id="x")

    def test_non_dense_reranked_ranks_rejected(self):
        rerank_input = _rerank_input(candidates=[_candidate("EV-1", rank=1), _candidate("EV-2", rank=2)])
        with pytest.raises(ValidationError, match="rank order"):
            RerankResult(
                case_id="AC-999", input=rerank_input,
                candidates=[
                    _reranked("EV-1", "doc_1", 1, 1, 1.0, 1.0),
                    _reranked("EV-2", "doc_1", 2, 3, 1.0, 1.0),  # gap: 1, 3
                ],
                reranker_config_id="x",
            )

    def test_duplicate_evidence_id_in_output_rejected(self):
        rerank_input = _rerank_input(candidates=[_candidate("EV-1", rank=1)])
        with pytest.raises(ValidationError, match="more than once"):
            RerankResult(
                case_id="AC-999", input=rerank_input,
                candidates=[
                    _reranked("EV-1", "doc_1", 1, 1, 1.0, 1.0),
                    _reranked("EV-1", "doc_1", 1, 2, 1.0, 1.0),
                ],
                reranker_config_id="x",
            )

    def test_output_smaller_than_input_is_allowed(self):
        """A reranker may trim to a smaller final K -- a SUBSET of the
        input candidate set is legitimate, not an error."""
        rerank_input = _rerank_input(candidates=[_candidate("EV-1", rank=1), _candidate("EV-2", rank=2)])
        result = RerankResult(
            case_id="AC-999", input=rerank_input,
            candidates=[_reranked("EV-1", "doc_1", 1, 1, 1.0, 1.0)],  # EV-2 dropped
            reranker_config_id="x",
        )
        assert len(result.candidates) == 1

    def test_candidate_addition_rejected(self):
        """The critical invariant: a reranker cannot introduce an
        evidence_id absent from its input candidate set."""
        rerank_input = _rerank_input(candidates=[_candidate("EV-1", rank=1)])
        with pytest.raises(ValidationError, match="introduced evidence_id"):
            RerankResult(
                case_id="AC-999", input=rerank_input,
                candidates=[_reranked("EV-999-NEVER-SUPPLIED", "doc_x", 1, 1, 1.0, 1.0)],
                reranker_config_id="x",
            )

    def test_original_rank_mismatch_rejected(self):
        rerank_input = _rerank_input(candidates=[_candidate("EV-1", score=0.5, rank=1)])
        with pytest.raises(ValidationError, match="original_rank/original_score"):
            RerankResult(
                case_id="AC-999", input=rerank_input,
                candidates=[_reranked("EV-1", "doc_1", original_rank=99, reranked_rank=1, original_score=0.5, reranker_score=1.0)],
                reranker_config_id="x",
            )

    def test_original_score_mismatch_rejected(self):
        rerank_input = _rerank_input(candidates=[_candidate("EV-1", score=0.5, rank=1)])
        with pytest.raises(ValidationError, match="original_rank/original_score"):
            RerankResult(
                case_id="AC-999", input=rerank_input,
                candidates=[_reranked("EV-1", "doc_1", original_rank=1, reranked_rank=1, original_score=0.999, reranker_score=1.0)],
                reranker_config_id="x",
            )

    def test_extra_field_rejected(self):
        rerank_input = _rerank_input()
        with pytest.raises(ValidationError):
            RerankResult(
                case_id="AC-999", input=rerank_input, candidates=[], reranker_config_id="x",
                required_evidence_ids=["EV-1"],
            )

    def test_result_json_never_contains_ground_truth_field_names(self):
        rerank_input = _rerank_input()
        result = RerankResult(case_id="AC-999", input=rerank_input, candidates=[], reranker_config_id="x")
        dumped = result.model_dump_json()
        for forbidden in ("expected_outcome", "required_evidence_ids", "should_abstain", "rationale", "failure_mode_tag"):
            assert forbidden not in dumped
