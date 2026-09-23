"""Unit tests for app.evaluation.evaluator: the citation-validity formula
and per-case evaluation against ground truth."""

from pathlib import Path

import pytest

from app.datasets.loader import load_dataset
from app.evaluation.evaluator import compute_citation_validity, compute_required_evidence_recall, evaluate_finding
from app.models.contracts import (
    AgentFinding,
    Evidence,
    EvidenceReference,
    EvaluationCase,
    ExpectedOutcome,
    Finding,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATASET_DIR = REPO_ROOT / "datasets"


class TestCitationValidityFormula:
    def test_all_valid_citations(self):
        assert compute_citation_validity(["a", "b"], {"a", "b", "c"}, Finding.PASS) == 1.0

    def test_single_valid_citation(self):
        assert compute_citation_validity(["a"], {"a", "b"}, Finding.PASS) == 1.0

    def test_mixed_valid_and_invalid_citations(self):
        result = compute_citation_validity(["a", "z"], {"a", "b"}, Finding.EXCEPTION)
        assert result == pytest.approx(0.5)

    def test_all_invalid_citations(self):
        assert compute_citation_validity(["z", "y"], {"a", "b"}, Finding.EXCEPTION) == 0.0

    def test_duplicate_citations_are_deduplicated_before_scoring(self):
        # 3 raw citations of the same document -> denominator is 1, not 3.
        assert compute_citation_validity(["a", "a", "a"], {"a", "b"}, Finding.PASS) == 1.0

    def test_duplicate_valid_and_duplicate_invalid_mixed(self):
        # unique cited = {a, z}; valid = {a} -> 1/2, regardless of repeat counts.
        result = compute_citation_validity(["a", "a", "z", "z", "z"], {"a", "b"}, Finding.EXCEPTION)
        assert result == pytest.approx(0.5)

    def test_zero_citations_on_abstention_is_vacuously_valid(self):
        assert compute_citation_validity([], {"a", "b"}, Finding.INSUFFICIENT_EVIDENCE) == 1.0

    def test_zero_citations_on_pass_is_a_grounding_failure(self):
        assert compute_citation_validity([], {"a", "b"}, Finding.PASS) == 0.0

    def test_zero_citations_on_exception_is_a_grounding_failure(self):
        assert compute_citation_validity([], {"a", "b"}, Finding.EXCEPTION) == 0.0

    def test_zero_citations_with_zero_available_evidence_still_follows_the_rule(self):
        # Edge case: no evidence existed at all. Rule is about the finding
        # type, not about whether anything was ever available to cite.
        assert compute_citation_validity([], set(), Finding.INSUFFICIENT_EVIDENCE) == 1.0
        assert compute_citation_validity([], set(), Finding.PASS) == 0.0


def _make_evidence(evidence_id: str, document_id: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id, document_id=document_id, title=evidence_id, period="2025-Q1",
        content="content", structured_fields={},
    )


class TestComputeRequiredEvidenceRecall:
    """Explicit evidence_id <-> document_id mapping tests -- the core
    correctness requirement: never compare an evidence_id to a document_id
    directly."""

    def test_full_recall_when_all_required_cited(self):
        evidence = [_make_evidence("EV-1", "doc-alpha"), _make_evidence("EV-2", "doc-beta")]
        result = compute_required_evidence_recall(["EV-1", "EV-2"], ["doc-alpha", "doc-beta"], evidence)
        assert result == 1.0

    def test_partial_recall(self):
        evidence = [_make_evidence("EV-1", "doc-alpha"), _make_evidence("EV-2", "doc-beta")]
        # Only doc-alpha (mapped from EV-1) cited; EV-2/doc-beta missed.
        result = compute_required_evidence_recall(["EV-1", "EV-2"], ["doc-alpha"], evidence)
        assert result == pytest.approx(0.5)

    def test_zero_recall_when_none_of_the_required_evidence_cited(self):
        evidence = [_make_evidence("EV-1", "doc-alpha")]
        result = compute_required_evidence_recall(["EV-1"], ["doc-unrelated"], evidence)
        assert result == 0.0

    def test_citing_a_non_required_document_does_not_help_or_hurt(self):
        """Recall only counts required items -- citing extra, non-required
        (but valid) documents neither inflates nor deflates the score."""
        evidence = [_make_evidence("EV-1", "doc-alpha"), _make_evidence("EV-2", "doc-beta")]
        result = compute_required_evidence_recall(["EV-1"], ["doc-alpha", "doc-beta"], evidence)
        assert result == 1.0  # the one required item (doc-alpha) was cited; doc-beta is just extra

    def test_empty_required_evidence_ids_is_none_not_zero(self):
        """Not applicable, not a measured failure."""
        evidence = [_make_evidence("EV-1", "doc-alpha")]
        result = compute_required_evidence_recall([], ["doc-alpha"], evidence)
        assert result is None

    def test_duplicate_citations_do_not_inflate_recall(self):
        evidence = [_make_evidence("EV-1", "doc-alpha"), _make_evidence("EV-2", "doc-beta")]
        result = compute_required_evidence_recall(
            ["EV-1", "EV-2"], ["doc-alpha", "doc-alpha", "doc-alpha"], evidence
        )
        assert result == pytest.approx(0.5)  # doc-beta (EV-2) still never cited

    def test_never_compares_evidence_id_to_document_id_directly(self):
        """Adversarial case: evidence_id and document_id happen to look
        alike ('EV-1' vs a document_id that is literally 'EV-1'). Citing
        the evidence_id string itself (not the real document_id) must NOT
        count as a valid citation -- proves the mapping goes through
        Evidence records, not naive string comparison."""
        evidence = [_make_evidence("EV-1", "doc-alpha")]  # real document_id is "doc-alpha"
        result = compute_required_evidence_recall(["EV-1"], ["EV-1"], evidence)  # cites the ID string, not the doc
        assert result == 0.0


def _case_and_evidence(
    expected_finding: Finding, should_abstain: bool, evidence_pool: list[str], required_evidence_ids=None
) -> tuple[EvaluationCase, list[Evidence]]:
    evidence = [
        Evidence(
            evidence_id=eid,
            document_id=f"doc_{eid}",
            title=eid,
            period="2025-Q1",
            content="content",
            structured_fields={},
        )
        for eid in evidence_pool
    ]
    case = EvaluationCase(
        case_id="AC-999",
        control_id="CTRL-X",
        scenario_description="x",
        evidence_pool=evidence_pool,
        period="2025-Q1",
        expected_outcome=ExpectedOutcome(
            expected_finding=expected_finding,
            required_evidence_ids=required_evidence_ids or [],
            should_abstain=should_abstain,
            rationale="x",
        ),
        failure_mode_tag="x",
    )
    return case, evidence


class TestEvaluateFinding:
    def test_correct_finding_and_correct_abstention(self):
        case, evidence = _case_and_evidence(Finding.PASS, False, ["EV-1"])
        finding = AgentFinding(
            case_id="AC-999",
            finding=Finding.PASS,
            confidence=0.9,
            reasoning_summary="x",
            evidence=[EvidenceReference(document_id="doc_EV-1", reference="x")],
            missing_information=[],
            abstain=False,
        )
        result = evaluate_finding(case, evidence, finding, run_id="run-1")
        assert result.correct_finding is True
        assert result.abstention_correct is True
        assert result.schema_valid is True
        assert result.citation_validity == 1.0

    def test_required_evidence_recall_wired_through_evaluate_finding(self):
        case, evidence = _case_and_evidence(
            Finding.PASS, False, ["EV-1", "EV-2"], required_evidence_ids=["EV-1", "EV-2"]
        )
        finding = AgentFinding(
            case_id="AC-999", finding=Finding.PASS, confidence=0.9, reasoning_summary="x",
            evidence=[EvidenceReference(document_id="doc_EV-1", reference="x")],  # only EV-1 cited
            missing_information=[], abstain=False,
        )
        result = evaluate_finding(case, evidence, finding, run_id="run-1")
        assert result.required_evidence_recall == pytest.approx(0.5)

    def test_incorrect_finding_flagged_false(self):
        case, evidence = _case_and_evidence(Finding.PASS, False, ["EV-1"])
        finding = AgentFinding(
            case_id="AC-999",
            finding=Finding.EXCEPTION,
            confidence=0.9,
            reasoning_summary="x",
            evidence=[EvidenceReference(document_id="doc_EV-1", reference="x")],
            missing_information=[],
            abstain=False,
        )
        result = evaluate_finding(case, evidence, finding, run_id="run-1")
        assert result.correct_finding is False

    def test_wrong_finding_and_wrong_abstention_scored_independently(self):
        """Finding correctness and abstention correctness are two separate
        fields that must be computed independently, even though Component 1's
        invariant correlates them within a single AgentFinding."""
        case, evidence = _case_and_evidence(Finding.INSUFFICIENT_EVIDENCE, True, ["EV-1"])
        finding = AgentFinding(
            case_id="AC-999",
            finding=Finding.PASS,
            confidence=0.9,
            reasoning_summary="x",
            evidence=[],
            missing_information=[],
            abstain=False,
        )
        result = evaluate_finding(case, evidence, finding, run_id="run-1")
        assert result.correct_finding is False
        assert result.abstention_correct is False

    def test_evaluator_ignores_required_evidence_ids_for_citation_validity(self):
        """A citation to evidence that exists in the pool but is NOT in
        required_evidence_ids must still score as fully valid -- required
        evidence is a ground-truth concept unrelated to citation validity."""
        case, evidence = _case_and_evidence(
            Finding.PASS, False, ["EV-1", "EV-2"], required_evidence_ids=["EV-1"]
        )
        # Cites doc_EV-2, which is NOT required, but IS in the pool.
        finding = AgentFinding(
            case_id="AC-999",
            finding=Finding.PASS,
            confidence=0.9,
            reasoning_summary="x",
            evidence=[EvidenceReference(document_id="doc_EV-2", reference="x")],
            missing_information=[],
            abstain=False,
        )
        result = evaluate_finding(case, evidence, finding, run_id="run-1")
        assert result.citation_validity == 1.0

    def test_citation_to_distractor_not_in_evidence_at_all_is_invalid(self):
        case, evidence = _case_and_evidence(Finding.PASS, False, ["EV-1"])
        finding = AgentFinding(
            case_id="AC-999",
            finding=Finding.PASS,
            confidence=0.9,
            reasoning_summary="x",
            evidence=[EvidenceReference(document_id="doc_never_existed", reference="x")],
            missing_information=[],
            abstain=False,
        )
        result = evaluate_finding(case, evidence, finding, run_id="run-1")
        assert result.citation_validity == 0.0


class TestEvaluateFindingWithSchemaFailure:
    """Covers the finding=None path: the agent raised AgentOutputError and
    run_traced recorded output=None."""

    def test_none_finding_is_scored_as_schema_invalid(self):
        case, evidence = _case_and_evidence(Finding.PASS, False, ["EV-1"])
        result = evaluate_finding(case, evidence, None, run_id="run-1")
        assert result.actual is None
        assert result.schema_valid is False
        assert result.correct_finding is False
        assert result.citation_validity == 0.0
        assert result.abstention_correct is None
        # This case's required_evidence_ids is empty (helper default) -> not applicable.
        assert result.required_evidence_recall is None

    def test_none_finding_with_required_evidence_scores_zero_recall(self):
        """When there WAS required evidence but the agent produced no valid
        output at all, recall is a real 0.0 (a grounding failure), not None."""
        case, evidence = _case_and_evidence(
            Finding.PASS, False, ["EV-1"], required_evidence_ids=["EV-1"]
        )
        result = evaluate_finding(case, evidence, None, run_id="run-1")
        assert result.required_evidence_recall == 0.0

    def test_none_finding_result_is_a_valid_pydantic_model(self):
        """EvaluationResult.actual is Optional -- constructing one with
        actual=None must not raise."""
        case, evidence = _case_and_evidence(Finding.INSUFFICIENT_EVIDENCE, True, ["EV-1"])
        result = evaluate_finding(case, evidence, None, run_id="run-1")
        assert result.model_dump()["actual"] is None


class TestEvaluatorAgainstRealDataset:
    def test_deterministic_baseline_results_evaluate_as_fully_correct(self):
        from app.agent.deterministic_baseline import DeterministicBaselineAgent
        from app.agent.interface import build_agent_input

        dataset = load_dataset(REAL_DATASET_DIR)
        agent = DeterministicBaselineAgent()
        for case in dataset.cases:
            control = dataset.controls[case.control_id]
            evidence = dataset.evidence_for_case(case)
            task = build_agent_input(case, control, evidence)
            finding = agent.run(task)
            result = evaluate_finding(case, evidence, finding, run_id="run-1")
            assert result.correct_finding is True, case.case_id
            assert result.citation_validity == 1.0, case.case_id
            assert result.abstention_correct is True, case.case_id
