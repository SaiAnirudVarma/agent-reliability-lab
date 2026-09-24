"""Unit tests for app.reranking.cohere_reranker.CohereReranker.

Uses FakeCohereClient throughout -- no network, no COHERE_API_KEY, no real
cohere package required.
"""

from __future__ import annotations

import math

import pytest

from app.models.contracts import Control, Evidence, EvaluationCase
from app.reranking.cohere_reranker import CohereClient, CohereReranker, CohereRerankerError
from app.reranking.contracts import RerankInput
from app.retrieval.contracts import RetrievalQuery, RetrievedEvidence, build_retrieval_query
from app.retrieval.serialization import evidence_embedding_text, retrieval_query_embedding_text
from tests.support.fake_cohere_client import FakeCohereClient, FakeCohereResponse, FakeCohereResultItem


def _control(**overrides) -> Control:
    defaults = dict(control_id="CTRL-X", name="N", description="D", requirement_text="R")
    defaults.update(overrides)
    return Control(**defaults)


def _evidence(evidence_id, title="T", content="C") -> Evidence:
    return Evidence(evidence_id=evidence_id, document_id=f"doc_{evidence_id}", title=title, period="2025-Q1", content=content, structured_fields={})


def _query(case_id="AC-999", control=None) -> RetrievalQuery:
    return RetrievalQuery(case_id=case_id, control=control or _control(), scenario_description="Scenario.", period="2025-Q1")


def _candidate(evidence_id, score, rank) -> RetrievedEvidence:
    return RetrievedEvidence(evidence_id=evidence_id, document_id=f"doc_{evidence_id}", score=score, rank=rank, retrieval_method="x")


def _rerank_input(*candidates, case_id="AC-999", control=None) -> RerankInput:
    return RerankInput(case_id=case_id, query=_query(case_id, control), candidates=list(candidates), candidate_depth=len(candidates))


def _reranker(client, evidence_by_id=None, model_name="rerank-v4.0-pro") -> CohereReranker:
    return CohereReranker(client, model_name, evidence_by_id or {})


class TestResponseIndexToCandidateIdentityMapping:
    def test_maps_indices_back_to_original_candidates(self):
        ev_a, ev_b = _evidence("A"), _evidence("B")
        rerank_input = _rerank_input(_candidate("A", 0.1, 1), _candidate("B", 0.2, 2))
        client = FakeCohereClient(FakeCohereResponse(results=[
            FakeCohereResultItem(index=1, relevance_score=0.9),  # candidates[1] == B
            FakeCohereResultItem(index=0, relevance_score=0.1),  # candidates[0] == A
        ]))
        result = _reranker(client, {"A": ev_a, "B": ev_b}).rerank(rerank_input)
        assert [c.evidence_id for c in result.candidates] == ["B", "A"]

    def test_arbitrary_reordered_response_still_maps_correctly(self):
        evidence_by_id = {f"EV-{i}": _evidence(f"EV-{i}") for i in range(5)}
        candidates = [_candidate(f"EV-{i}", 0.5, i + 1) for i in range(5)]
        rerank_input = _rerank_input(*candidates)
        # Deliberately scrambled index order with distinct scores.
        client = FakeCohereClient(FakeCohereResponse(results=[
            FakeCohereResultItem(index=3, relevance_score=0.9),
            FakeCohereResultItem(index=0, relevance_score=0.8),
            FakeCohereResultItem(index=4, relevance_score=0.7),
            FakeCohereResultItem(index=1, relevance_score=0.6),
            FakeCohereResultItem(index=2, relevance_score=0.5),
        ]))
        result = _reranker(client, evidence_by_id).rerank(rerank_input)
        assert [c.evidence_id for c in result.candidates] == ["EV-3", "EV-0", "EV-4", "EV-1", "EV-2"]

    def test_score_propagation(self):
        rerank_input = _rerank_input(_candidate("A", 0.1, 1))
        client = FakeCohereClient(FakeCohereResponse(results=[FakeCohereResultItem(index=0, relevance_score=0.777)]))
        result = _reranker(client, {"A": _evidence("A")}).rerank(rerank_input)
        assert result.candidates[0].reranker_score == pytest.approx(0.777)

    def test_original_rank_and_score_preserved(self):
        filler = [_candidate(f"FILLER-{i}", 0.0, i + 1) for i in range(6)]
        rerank_input = _rerank_input(*filler, _candidate("A", 0.42, 7))
        client = FakeCohereClient(FakeCohereResponse(results=[
            FakeCohereResultItem(index=i, relevance_score=0.0) for i in range(6)
        ] + [FakeCohereResultItem(index=6, relevance_score=0.9)]))
        evidence_by_id = {f"FILLER-{i}": _evidence(f"FILLER-{i}") for i in range(6)}
        evidence_by_id["A"] = _evidence("A")
        result = _reranker(client, evidence_by_id).rerank(rerank_input)
        by_id = {c.evidence_id: c for c in result.candidates}
        assert by_id["A"].original_rank == 7
        assert by_id["A"].original_score == pytest.approx(0.42)

    def test_dense_reranked_ranks(self):
        evidence_by_id = {f"EV-{i}": _evidence(f"EV-{i}") for i in range(3)}
        candidates = [_candidate(f"EV-{i}", 0.5, i + 1) for i in range(3)]
        rerank_input = _rerank_input(*candidates)
        client = FakeCohereClient(FakeCohereResponse(results=[
            FakeCohereResultItem(index=2, relevance_score=0.9),
            FakeCohereResultItem(index=1, relevance_score=0.5),
            FakeCohereResultItem(index=0, relevance_score=0.1),
        ]))
        result = _reranker(client, evidence_by_id).rerank(rerank_input)
        assert [c.reranked_rank for c in result.candidates] == [1, 2, 3]


class TestNoCandidateAdditions:
    def test_result_contains_only_supplied_candidates(self):
        rerank_input = _rerank_input(_candidate("A", 0.1, 1), _candidate("B", 0.2, 2))
        client = FakeCohereClient(FakeCohereResponse(results=[
            FakeCohereResultItem(index=0, relevance_score=0.5),
            FakeCohereResultItem(index=1, relevance_score=0.6),
        ]))
        result = _reranker(client, {"A": _evidence("A"), "B": _evidence("B")}).rerank(rerank_input)
        assert {c.evidence_id for c in result.candidates} <= {"A", "B"}


class TestResponseValidationFailsClosed:
    def test_duplicate_index_rejected(self):
        rerank_input = _rerank_input(_candidate("A", 0.1, 1), _candidate("B", 0.2, 2))
        client = FakeCohereClient(FakeCohereResponse(results=[
            FakeCohereResultItem(index=0, relevance_score=0.5),
            FakeCohereResultItem(index=0, relevance_score=0.6),
        ]))
        with pytest.raises(CohereRerankerError, match="duplicate index"):
            _reranker(client, {"A": _evidence("A"), "B": _evidence("B")}).rerank(rerank_input)

    def test_index_above_range_rejected(self):
        rerank_input = _rerank_input(_candidate("A", 0.1, 1))
        client = FakeCohereClient(FakeCohereResponse(results=[FakeCohereResultItem(index=5, relevance_score=0.5)]))
        with pytest.raises(CohereRerankerError, match="outside the valid candidate range"):
            _reranker(client, {"A": _evidence("A")}).rerank(rerank_input)

    def test_negative_index_rejected(self):
        rerank_input = _rerank_input(_candidate("A", 0.1, 1))
        client = FakeCohereClient(FakeCohereResponse(results=[FakeCohereResultItem(index=-1, relevance_score=0.5)]))
        with pytest.raises(CohereRerankerError, match="outside the valid candidate range"):
            _reranker(client, {"A": _evidence("A")}).rerank(rerank_input)

    def test_missing_results_when_requesting_all_candidates_rejected(self):
        rerank_input = _rerank_input(_candidate("A", 0.1, 1), _candidate("B", 0.2, 2))
        client = FakeCohereClient(FakeCohereResponse(results=[FakeCohereResultItem(index=0, relevance_score=0.5)]))
        with pytest.raises(CohereRerankerError, match="expected 2 results"):
            _reranker(client, {"A": _evidence("A"), "B": _evidence("B")}).rerank(rerank_input)

    def test_non_finite_score_rejected_nan(self):
        rerank_input = _rerank_input(_candidate("A", 0.1, 1))
        client = FakeCohereClient(FakeCohereResponse(results=[FakeCohereResultItem(index=0, relevance_score=math.nan)]))
        with pytest.raises(CohereRerankerError, match="non-finite"):
            _reranker(client, {"A": _evidence("A")}).rerank(rerank_input)

    def test_non_finite_score_rejected_inf(self):
        rerank_input = _rerank_input(_candidate("A", 0.1, 1))
        client = FakeCohereClient(FakeCohereResponse(results=[FakeCohereResultItem(index=0, relevance_score=math.inf)]))
        with pytest.raises(CohereRerankerError, match="non-finite"):
            _reranker(client, {"A": _evidence("A")}).rerank(rerank_input)

    def test_missing_results_field_rejected(self):
        rerank_input = _rerank_input(_candidate("A", 0.1, 1))
        client = FakeCohereClient(object())  # no .results attribute at all
        with pytest.raises(CohereRerankerError, match="no 'results' field"):
            _reranker(client, {"A": _evidence("A")}).rerank(rerank_input)

    def test_missing_evidence_record_rejected(self):
        """The candidate is legitimate, but the caller failed to supply
        its Evidence content -- must fail closed, never send a blank/
        placeholder document to the provider."""
        rerank_input = _rerank_input(_candidate("A", 0.1, 1))
        client = FakeCohereClient(FakeCohereResponse(results=[]))
        with pytest.raises(CohereRerankerError, match="no Evidence record available"):
            _reranker(client, {}).rerank(rerank_input)  # evidence_by_id deliberately empty

    def test_non_integer_index_rejected(self):
        rerank_input = _rerank_input(_candidate("A", 0.1, 1))
        client = FakeCohereClient(FakeCohereResponse(results=[FakeCohereResultItem(index="0", relevance_score=0.5)]))
        with pytest.raises(CohereRerankerError, match="outside the valid candidate range"):
            _reranker(client, {"A": _evidence("A")}).rerank(rerank_input)

    def test_invalid_final_k_raises(self):
        rerank_input = _rerank_input(_candidate("A", 0.1, 1))
        client = FakeCohereClient(FakeCohereResponse(results=[]))
        with pytest.raises(ValueError, match="final_k"):
            _reranker(client, {"A": _evidence("A")}).rerank(rerank_input, final_k=0)


class TestModelProvenance:
    def test_requested_model_recorded(self):
        rerank_input = _rerank_input(_candidate("A", 0.1, 1))
        client = FakeCohereClient(FakeCohereResponse(results=[FakeCohereResultItem(index=0, relevance_score=0.5)]))
        reranker = _reranker(client, {"A": _evidence("A")}, model_name="rerank-v4.0-pro")
        reranker.rerank(rerank_input)
        assert reranker.model_name == "rerank-v4.0-pro"
        assert client.last_call["model"] == "rerank-v4.0-pro"

    def test_served_model_none_when_not_authoritatively_available(self):
        """The realistic default: Cohere's response carries no served-model
        field at all, so it must stay None -- never copied from requested."""
        rerank_input = _rerank_input(_candidate("A", 0.1, 1))
        client = FakeCohereClient(FakeCohereResponse(results=[FakeCohereResultItem(index=0, relevance_score=0.5)]))
        reranker = _reranker(client, {"A": _evidence("A")}, model_name="rerank-v4.0-pro")
        reranker.rerank(rerank_input)
        assert reranker.last_served_model_name is None

    def test_served_model_populated_when_response_reports_one(self):
        rerank_input = _rerank_input(_candidate("A", 0.1, 1))
        client = FakeCohereClient(FakeCohereResponse(
            results=[FakeCohereResultItem(index=0, relevance_score=0.5)], model="rerank-v4.0-pro-2026-01-01",
        ))
        reranker = _reranker(client, {"A": _evidence("A")}, model_name="rerank-v4.0-pro")
        reranker.rerank(rerank_input)
        assert reranker.last_served_model_name == "rerank-v4.0-pro-2026-01-01"

    def test_satisfies_reranker_protocol(self):
        from app.reranking.interface import Reranker

        client = FakeCohereClient(FakeCohereResponse(results=[]))
        assert isinstance(_reranker(client), Reranker)

    def test_client_satisfies_cohere_client_protocol(self):
        client = FakeCohereClient(FakeCohereResponse(results=[]))
        assert isinstance(client, CohereClient)


class TestNoGroundTruthSentToProvider:
    def test_case_id_excluded_from_semantic_query(self):
        rerank_input = _rerank_input(_candidate("A", 0.1, 1), case_id="AC-777-UNIQUE-MARKER")
        client = FakeCohereClient(FakeCohereResponse(results=[FakeCohereResultItem(index=0, relevance_score=0.5)]))
        _reranker(client, {"A": _evidence("A")}).rerank(rerank_input)
        assert "AC-777-UNIQUE-MARKER" not in client.last_call["query"]

    def test_query_text_matches_the_shared_deterministic_serializer_exactly(self):
        control = _control(name="Ctrl Name", description="Ctrl Desc", requirement_text="Ctrl Req")
        rerank_input = _rerank_input(_candidate("A", 0.1, 1), control=control)
        client = FakeCohereClient(FakeCohereResponse(results=[FakeCohereResultItem(index=0, relevance_score=0.5)]))
        _reranker(client, {"A": _evidence("A")}).rerank(rerank_input)
        assert client.last_call["query"] == retrieval_query_embedding_text(rerank_input.query)

    def test_document_text_matches_the_shared_deterministic_serializer_exactly(self):
        evidence = _evidence("A", title="Distinctive Title", content="Distinctive content.")
        rerank_input = _rerank_input(_candidate("A", 0.1, 1))
        client = FakeCohereClient(FakeCohereResponse(results=[FakeCohereResultItem(index=0, relevance_score=0.5)]))
        _reranker(client, {"A": evidence}).rerank(rerank_input)
        assert client.last_call["documents"] == [evidence_embedding_text(evidence)]

    def test_real_case_rationale_and_failure_mode_tag_never_transmitted(self):
        case = EvaluationCase(
            case_id="AC-042", control_id="CTRL-X", scenario_description="Evaluate the thing.",
            evidence_pool=["EV-1"], period="2025-Q1",
            expected_outcome={
                "expected_finding": "PASS", "required_evidence_ids": ["EV-1"], "should_abstain": False,
                "rationale": "SECRET-RATIONALE-MARKER-MUST-NEVER-APPEAR",
            },
            failure_mode_tag="SECRET-TAG-MARKER-MUST-NEVER-APPEAR",
        )
        control = _control(control_id="CTRL-X")
        query = build_retrieval_query(case, control)
        evidence = _evidence("EV-1", title="T", content="C")
        candidate = _candidate("EV-1", 0.5, 1)
        rerank_input = RerankInput(case_id=case.case_id, query=query, candidates=[candidate], candidate_depth=1)

        client = FakeCohereClient(FakeCohereResponse(results=[FakeCohereResultItem(index=0, relevance_score=0.5)]))
        _reranker(client, {"EV-1": evidence}).rerank(rerank_input)

        assert "SECRET-RATIONALE-MARKER-MUST-NEVER-APPEAR" not in client.last_call["query"]
        assert "SECRET-RATIONALE-MARKER-MUST-NEVER-APPEAR" not in "".join(client.last_call["documents"])
        assert "SECRET-TAG-MARKER-MUST-NEVER-APPEAR" not in client.last_call["query"]
        assert "SECRET-TAG-MARKER-MUST-NEVER-APPEAR" not in "".join(client.last_call["documents"])

    def test_only_supplied_candidates_evidence_is_sent_never_the_whole_pool(self):
        """Even if evidence_by_id contains MORE records than the candidate
        set (e.g. the full corpus), only the supplied candidates' content
        must be sent."""
        evidence_by_id = {"A": _evidence("A", content="content A"), "B": _evidence("B", content="content B"),
                           "C": _evidence("C", content="content C, never supplied as a candidate")}
        rerank_input = _rerank_input(_candidate("A", 0.1, 1), _candidate("B", 0.2, 2))
        client = FakeCohereClient(FakeCohereResponse(results=[
            FakeCohereResultItem(index=0, relevance_score=0.5), FakeCohereResultItem(index=1, relevance_score=0.6),
        ]))
        _reranker(client, evidence_by_id).rerank(rerank_input)
        assert len(client.last_call["documents"]) == 2
        assert "content C" not in "".join(client.last_call["documents"])


class TestFinalKTrimsDocumentCountSent:
    def test_top_n_matches_final_k(self):
        evidence_by_id = {f"EV-{i}": _evidence(f"EV-{i}") for i in range(3)}
        candidates = [_candidate(f"EV-{i}", 0.5, i + 1) for i in range(3)]
        rerank_input = _rerank_input(*candidates)
        client = FakeCohereClient(FakeCohereResponse(results=[FakeCohereResultItem(index=0, relevance_score=0.9)]))
        _reranker(client, evidence_by_id).rerank(rerank_input, final_k=1)
        assert client.last_call["top_n"] == 1
