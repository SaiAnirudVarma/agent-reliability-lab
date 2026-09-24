"""Unit tests for app.integration.failure_attribution.attribute_case_failure_layers
-- a pure function over already-produced artifact evidence, never over
hidden model reasoning.
"""

from __future__ import annotations

from app.integration.failure_attribution import FailureLayer, attribute_case_failure_layers
from app.models.contracts import EvaluationResult, ExpectedOutcome, Finding
from app.reranking.contracts import RerankedEvidence


def _reranked(evidence_id: str, reranked_rank: int) -> RerankedEvidence:
    return RerankedEvidence(
        evidence_id=evidence_id, document_id=f"doc_{evidence_id}", original_rank=reranked_rank,
        reranked_rank=reranked_rank, original_score=1.0, reranker_score=1.0, reranker_method="pass-through",
    )


def _result(
    correct_finding: bool, schema_valid: bool, citation_validity: float,
    required_evidence_recall=None, actual_finding=Finding.PASS,
) -> EvaluationResult:
    return EvaluationResult(
        case_id="AC-001", run_id="run-1",
        expected=ExpectedOutcome(expected_finding=Finding.PASS, required_evidence_ids=[], should_abstain=False, rationale="r"),
        actual=None if not schema_valid else None,  # not inspected by the function under test
        correct_finding=correct_finding, schema_valid=schema_valid, citation_validity=citation_validity,
        required_evidence_recall=required_evidence_recall, abstention_correct=None,
    )


class TestMalformedSchemaFailure:
    def test_schema_invalid_reports_only_malformed_layer(self):
        candidates = [_reranked("EV-1", 1)]
        result = _result(correct_finding=False, schema_valid=False, citation_validity=0.0)
        layers = attribute_case_failure_layers(["EV-1"], candidates, evidence_top_k=5, evaluation_result=result)
        assert layers == [FailureLayer.MALFORMED_SCHEMA_FAILURE]

    def test_malformed_is_never_combined_with_other_layers(self):
        # Even with an obvious candidate-generation miss present, a
        # malformed response reports ONLY the malformed layer -- there is
        # no finding to grade any other layer against.
        candidates: list[RerankedEvidence] = []
        result = _result(correct_finding=False, schema_valid=False, citation_validity=0.0)
        layers = attribute_case_failure_layers(["EV-999"], candidates, evidence_top_k=5, evaluation_result=result)
        assert layers == [FailureLayer.MALFORMED_SCHEMA_FAILURE]


class TestCandidateGenerationMiss:
    def test_required_evidence_absent_from_every_reranked_candidate(self):
        candidates = [_reranked("EV-1", 1), _reranked("EV-2", 2)]
        result = _result(correct_finding=False, schema_valid=True, citation_validity=1.0, required_evidence_recall=0.0)
        layers = attribute_case_failure_layers(["EV-999"], candidates, evidence_top_k=5, evaluation_result=result)
        assert FailureLayer.CANDIDATE_GENERATION_MISS in layers

    def test_never_double_counts_the_same_layer_for_multiple_missing_items(self):
        candidates = [_reranked("EV-1", 1)]
        result = _result(correct_finding=False, schema_valid=True, citation_validity=1.0, required_evidence_recall=0.0)
        layers = attribute_case_failure_layers(["EV-998", "EV-999"], candidates, evidence_top_k=5, evaluation_result=result)
        assert layers.count(FailureLayer.CANDIDATE_GENERATION_MISS) == 1


class TestRankingIssue:
    def test_required_evidence_in_candidate_set_but_ranked_below_top_k(self):
        candidates = [_reranked(f"EV-{i}", i) for i in range(1, 11)]  # EV-7 at rank 7
        result = _result(correct_finding=False, schema_valid=True, citation_validity=1.0, required_evidence_recall=0.0)
        layers = attribute_case_failure_layers(["EV-7"], candidates, evidence_top_k=5, evaluation_result=result)
        assert FailureLayer.RANKING_ISSUE in layers
        assert FailureLayer.CANDIDATE_GENERATION_MISS not in layers

    def test_required_evidence_at_exactly_top_k_is_not_a_ranking_issue(self):
        candidates = [_reranked(f"EV-{i}", i) for i in range(1, 11)]
        result = _result(correct_finding=True, schema_valid=True, citation_validity=1.0, required_evidence_recall=1.0)
        layers = attribute_case_failure_layers(["EV-5"], candidates, evidence_top_k=5, evaluation_result=result)
        assert FailureLayer.RANKING_ISSUE not in layers


class TestCitationGroundingIssue:
    def test_required_evidence_recall_below_one_reports_citation_issue(self):
        candidates = [_reranked("EV-1", 1)]
        result = _result(correct_finding=False, schema_valid=True, citation_validity=1.0, required_evidence_recall=0.5)
        layers = attribute_case_failure_layers(["EV-1"], candidates, evidence_top_k=5, evaluation_result=result)
        assert FailureLayer.CITATION_GROUNDING_ISSUE in layers

    def test_required_evidence_recall_none_never_reports_citation_issue(self):
        candidates = [_reranked("EV-1", 1)]
        result = _result(correct_finding=True, schema_valid=True, citation_validity=1.0, required_evidence_recall=None)
        layers = attribute_case_failure_layers([], candidates, evidence_top_k=5, evaluation_result=result)
        assert FailureLayer.CITATION_GROUNDING_ISSUE not in layers


class TestLLMDecisionIssue:
    def test_wrong_finding_despite_full_grounding_reports_llm_decision_issue(self):
        candidates = [_reranked("EV-1", 1)]
        result = _result(correct_finding=False, schema_valid=True, citation_validity=1.0, required_evidence_recall=1.0)
        layers = attribute_case_failure_layers(["EV-1"], candidates, evidence_top_k=5, evaluation_result=result)
        assert layers == [FailureLayer.LLM_DECISION_ISSUE]

    def test_wrong_finding_with_no_required_evidence_at_all_still_reports_llm_decision_issue(self):
        candidates: list[RerankedEvidence] = []
        result = _result(correct_finding=False, schema_valid=True, citation_validity=1.0, required_evidence_recall=None)
        layers = attribute_case_failure_layers([], candidates, evidence_top_k=5, evaluation_result=result)
        assert layers == [FailureLayer.LLM_DECISION_ISSUE]

    def test_wrong_finding_with_low_citation_validity_is_not_reported_as_llm_decision_issue(self):
        """A poorly-grounded wrong answer is a grounding problem, not
        cleanly attributable to synthesis alone -- LLM_DECISION_ISSUE
        requires FULL grounding first."""

        candidates = [_reranked("EV-1", 1)]
        result = _result(correct_finding=False, schema_valid=True, citation_validity=0.5, required_evidence_recall=1.0)
        layers = attribute_case_failure_layers(["EV-1"], candidates, evidence_top_k=5, evaluation_result=result)
        assert FailureLayer.LLM_DECISION_ISSUE not in layers


class TestNoFailure:
    def test_correct_finding_with_full_grounding_reports_nothing(self):
        candidates = [_reranked("EV-1", 1)]
        result = _result(correct_finding=True, schema_valid=True, citation_validity=1.0, required_evidence_recall=1.0)
        layers = attribute_case_failure_layers(["EV-1"], candidates, evidence_top_k=5, evaluation_result=result)
        assert layers == []

    def test_correct_finding_with_no_required_evidence_reports_nothing(self):
        candidates: list[RerankedEvidence] = []
        result = _result(correct_finding=True, schema_valid=True, citation_validity=1.0, required_evidence_recall=None)
        layers = attribute_case_failure_layers([], candidates, evidence_top_k=5, evaluation_result=result)
        assert layers == []


class TestMultipleLayersCanCoexist:
    def test_one_missing_item_and_one_mis_ranked_item_both_reported(self):
        candidates = [_reranked(f"EV-{i}", i) for i in range(1, 11)]
        result = _result(correct_finding=False, schema_valid=True, citation_validity=1.0, required_evidence_recall=0.0)
        layers = attribute_case_failure_layers(["EV-999", "EV-7"], candidates, evidence_top_k=5, evaluation_result=result)
        assert FailureLayer.CANDIDATE_GENERATION_MISS in layers
        assert FailureLayer.RANKING_ISSUE in layers
        assert FailureLayer.CITATION_GROUNDING_ISSUE in layers
