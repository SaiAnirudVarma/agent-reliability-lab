"""Unit tests for app.retrieval.contracts: the retrieval leakage boundary
(RetrievalQuery) and the retrieval result contract (RetrievalResult /
RetrievedEvidence).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.datasets.loader import load_dataset
from app.models.contracts import Control
from app.retrieval.contracts import RetrievalQuery, RetrievalResult, RetrievedEvidence, build_retrieval_query

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATASET_DIR = REPO_ROOT / "datasets"


def _control() -> Control:
    return Control(
        control_id="CTRL-X",
        name="Test Control",
        description="A test control.",
        requirement_text="Do the thing.",
    )


def _query(**overrides) -> RetrievalQuery:
    defaults = dict(
        case_id="AC-999", control=_control(), scenario_description="Evaluate the thing.", period="2025-Q1"
    )
    defaults.update(overrides)
    return RetrievalQuery(**defaults)


def _candidate(evidence_id="EV-1", document_id="doc_1", score=1.0, rank=1, method="lexical-baseline-v1") -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=evidence_id, document_id=document_id, score=score, rank=rank, retrieval_method=method
    )


class TestRetrievalQueryLeakageBoundary:
    def test_valid_query_constructs(self):
        query = _query()
        assert query.case_id == "AC-999"

    @pytest.mark.parametrize(
        "forbidden_field,value",
        [
            ("expected_finding", "PASS"),
            ("required_evidence_ids", ["EV-1"]),
            ("should_abstain", False),
            ("rationale", "because"),
            ("failure_mode_tag", "some_tag"),
            ("evidence", []),
            ("evaluator_results", {}),
            ("previous_answer", "PASS"),
        ],
    )
    def test_forbidden_fields_are_rejected(self, forbidden_field, value):
        defaults = dict(
            case_id="AC-999", control=_control(), scenario_description="Evaluate.", period="2025-Q1"
        )
        defaults[forbidden_field] = value
        with pytest.raises(ValidationError):
            RetrievalQuery(**defaults)

    def test_build_retrieval_query_control_id_mismatch_raises(self):
        control = _control()
        control_mismatched = Control(
            control_id="CTRL-OTHER", name="x", description="x", requirement_text="x"
        )
        from app.models.contracts import EvaluationCase

        case = EvaluationCase(
            case_id="AC-001",
            control_id="CTRL-X",
            scenario_description="Evaluate.",
            evidence_pool=["EV-1"],
            period="2025-Q1",
            expected_outcome={
                "expected_finding": "PASS",
                "required_evidence_ids": [],
                "should_abstain": False,
                "rationale": "because",
            },
            failure_mode_tag="tag",
        )
        with pytest.raises(ValueError, match="control_id mismatch"):
            build_retrieval_query(case, control_mismatched)

    def test_build_retrieval_query_never_reads_expected_outcome_or_tag(self):
        """build_retrieval_query's own source reads only case_id,
        scenario_description, period -- verified behaviorally: the
        resulting query is identical regardless of what expected_outcome/
        failure_mode_tag contain."""
        from app.models.contracts import EvaluationCase

        base_kwargs = dict(
            case_id="AC-001",
            control_id="CTRL-X",
            scenario_description="Evaluate.",
            evidence_pool=["EV-1"],
            period="2025-Q1",
        )
        case_a = EvaluationCase(
            **base_kwargs,
            expected_outcome={
                "expected_finding": "PASS", "required_evidence_ids": [],
                "should_abstain": False, "rationale": "rationale A",
            },
            failure_mode_tag="tag_a",
        )
        case_b = EvaluationCase(
            **base_kwargs,
            expected_outcome={
                "expected_finding": "EXCEPTION", "required_evidence_ids": [],
                "should_abstain": False, "rationale": "totally different rationale B",
            },
            failure_mode_tag="tag_b",
        )
        control = _control()
        assert build_retrieval_query(case_a, control) == build_retrieval_query(case_b, control)

    def test_real_dataset_queries_never_contain_rationale_or_tag_text(self):
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        for case in dataset.cases:
            control = dataset.controls[case.control_id]
            query = build_retrieval_query(case, control)
            dumped = query.model_dump_json()
            assert "expected_outcome" not in dumped
            assert "failure_mode_tag" not in dumped
            assert case.expected_outcome.rationale not in dumped
            assert case.failure_mode_tag not in dumped

    def test_case_id_field_exists_but_query_schema_has_no_ground_truth_fields(self):
        assert "case_id" in RetrievalQuery.model_fields
        for forbidden in ("expected_outcome", "required_evidence_ids", "should_abstain", "rationale", "failure_mode_tag", "evidence"):
            assert forbidden not in RetrievalQuery.model_fields


class TestRetrievedEvidence:
    def test_valid_candidate_constructs(self):
        candidate = _candidate()
        assert candidate.rank == 1

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            RetrievedEvidence(
                evidence_id="EV-1", document_id="doc_1", score=1.0, rank=1,
                retrieval_method="lexical-baseline-v1", relevance_label=True,
            )

    def test_rank_must_be_positive(self):
        with pytest.raises(ValidationError):
            RetrievedEvidence(evidence_id="EV-1", document_id="doc_1", score=1.0, rank=0, retrieval_method="x")


class TestRetrievalResult:
    def test_valid_result_constructs(self):
        result = RetrievalResult(
            case_id="AC-999", query=_query(), candidates=[_candidate(rank=1), _candidate("EV-2", "doc_2", 0.5, 2)],
            top_k=2, retriever_config_id="lexical-baseline-v1",
        )
        assert len(result.candidates) == 2

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            RetrievalResult(
                case_id="AC-999", query=_query(), candidates=[], top_k=1,
                retriever_config_id="x", required_evidence_ids=["EV-1"],
            )

    def test_case_id_must_match_query_case_id(self):
        with pytest.raises(ValidationError, match="does not match"):
            RetrievalResult(
                case_id="AC-OTHER", query=_query(case_id="AC-999"), candidates=[], top_k=1,
                retriever_config_id="x",
            )

    def test_candidates_exceeding_top_k_rejected(self):
        with pytest.raises(ValidationError, match="exceeds top_k"):
            RetrievalResult(
                case_id="AC-999", query=_query(), candidates=[_candidate(rank=1), _candidate("EV-2", "doc_2", 0.5, 2)],
                top_k=1, retriever_config_id="x",
            )

    def test_candidates_may_be_fewer_than_top_k(self):
        result = RetrievalResult(
            case_id="AC-999", query=_query(), candidates=[_candidate(rank=1)], top_k=5, retriever_config_id="x",
        )
        assert len(result.candidates) == 1

    def test_ranks_must_be_dense_sequential_from_one(self):
        with pytest.raises(ValidationError, match="rank order"):
            RetrievalResult(
                case_id="AC-999", query=_query(),
                candidates=[_candidate("EV-1", "doc_1", 1.0, 1), _candidate("EV-2", "doc_2", 0.5, 3)],
                top_k=3, retriever_config_id="x",
            )

    def test_ranks_out_of_order_in_list_rejected(self):
        with pytest.raises(ValidationError, match="rank order"):
            RetrievalResult(
                case_id="AC-999", query=_query(),
                candidates=[_candidate("EV-1", "doc_1", 1.0, 2), _candidate("EV-2", "doc_2", 0.5, 1)],
                top_k=2, retriever_config_id="x",
            )

    def test_duplicate_evidence_id_rejected(self):
        with pytest.raises(ValidationError, match="more than once"):
            RetrievalResult(
                case_id="AC-999", query=_query(),
                candidates=[_candidate("EV-1", "doc_1", 1.0, 1), _candidate("EV-1", "doc_1", 0.5, 2)],
                top_k=2, retriever_config_id="x",
            )

    def test_result_json_never_contains_expected_outcome_fields(self):
        result = RetrievalResult(
            case_id="AC-999", query=_query(), candidates=[_candidate(rank=1)], top_k=1, retriever_config_id="x",
        )
        dumped = result.model_dump_json()
        for forbidden in ("expected_outcome", "required_evidence_ids", "should_abstain", "rationale", "failure_mode_tag"):
            assert forbidden not in dumped
