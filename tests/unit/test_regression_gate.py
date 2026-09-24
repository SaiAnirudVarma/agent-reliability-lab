"""Unit tests for app.regression.gate: pure, offline checks of a
candidate's EvaluationResults against a RegressionManifest's floors.

No file I/O of the manifest/candidate JSON themselves in this file --
see tests/unit/test_run_regression_gate_script.py for the CLI-level
(file-reading) equivalents.
"""

from __future__ import annotations

from app.models.contracts import EvaluationResult, ExpectedOutcome, Finding
from app.regression.gate import check_case, run_regression_gate
from app.regression.manifest import RegressionCase, RegressionManifest


def _case(**overrides) -> RegressionCase:
    defaults = dict(
        case_id="AC-001", category="finding_failure",
        observed_expected_finding=Finding.PASS, observed_should_abstain=False,
        observed_actual_finding=Finding.PASS, observed_correct_finding=True,
        require_correct_finding=False, require_abstention_correct=None,
        min_schema_valid=True, min_citation_validity=1.0, min_required_evidence_recall=1.0,
        notes="test case",
    )
    defaults.update(overrides)
    return RegressionCase(**defaults)


def _manifest(cases: list[RegressionCase]) -> RegressionManifest:
    return RegressionManifest(
        manifest_version="test-manifest-v1", dataset_version="synthetic-v2",
        source_run_id="run-1", source_artifact_sha256="a" * 64, source_config_id="test-config",
        cases=cases,
    )


def _result(
    case_id="AC-001", correct_finding=True, schema_valid=True, citation_validity=1.0,
    required_evidence_recall=1.0, abstention_correct=None, actual_finding=Finding.PASS,
) -> EvaluationResult:
    from app.models.contracts import AgentFinding

    actual = None
    if schema_valid:
        actual = AgentFinding(
            case_id=case_id, finding=actual_finding, confidence=0.9, reasoning_summary="r",
            evidence=[], missing_information=[], abstain=(actual_finding == Finding.INSUFFICIENT_EVIDENCE),
        )
    return EvaluationResult(
        case_id=case_id, run_id="candidate-run-1",
        expected=ExpectedOutcome(expected_finding=Finding.PASS, required_evidence_ids=[], should_abstain=False, rationale="r"),
        actual=actual, correct_finding=correct_finding, schema_valid=schema_valid,
        citation_validity=citation_validity, required_evidence_recall=required_evidence_recall,
        abstention_correct=abstention_correct,
    )


class TestPassingCase:
    def test_meets_every_floor_exactly(self):
        case = _case(min_citation_validity=1.0, min_required_evidence_recall=1.0)
        actual = _result(citation_validity=1.0, required_evidence_recall=1.0)
        result = check_case(case, actual)
        assert result.passed is True
        assert result.violations == []

    def test_exceeds_the_floor(self):
        case = _case(min_required_evidence_recall=0.5)
        actual = _result(required_evidence_recall=1.0)
        result = check_case(case, actual)
        assert result.passed is True


class TestWrongFindingRegression:
    def test_required_correct_finding_regresses_to_false(self):
        case = _case(require_correct_finding=True)
        actual = _result(correct_finding=False, actual_finding=Finding.EXCEPTION)
        result = check_case(case, actual)
        assert result.passed is False
        assert any("correct_finding" in v for v in result.violations)

    def test_not_required_correct_finding_does_not_fail_on_wrong_finding(self):
        """A known-wrong-finding case (require_correct_finding=False)
        must NOT fail the gate merely for still being wrong -- the
        manifest never demands a fix that wasn't already achieved."""

        case = _case(require_correct_finding=False)
        actual = _result(correct_finding=False, actual_finding=Finding.EXCEPTION)
        result = check_case(case, actual)
        assert result.passed is True


class TestCitationGroundingRegression:
    def test_citation_validity_below_floor_fails(self):
        case = _case(min_citation_validity=1.0)
        actual = _result(citation_validity=0.5)
        result = check_case(case, actual)
        assert result.passed is False
        assert any("citation_validity" in v for v in result.violations)

    def test_required_evidence_recall_below_floor_fails(self):
        case = _case(min_required_evidence_recall=1.0)
        actual = _result(required_evidence_recall=0.5)
        result = check_case(case, actual)
        assert result.passed is False
        assert any("required_evidence_recall" in v for v in result.violations)

    def test_required_evidence_recall_regressing_to_none_fails(self):
        case = _case(min_required_evidence_recall=0.5)
        actual = _result(required_evidence_recall=None)
        result = check_case(case, actual)
        assert result.passed is False

    def test_no_recall_floor_means_recall_is_never_checked(self):
        case = _case(min_required_evidence_recall=None)
        actual = _result(required_evidence_recall=0.0)
        result = check_case(case, actual)
        assert result.passed is True


class TestMalformedOrMissingCase:
    def test_schema_invalid_fails_when_floor_requires_valid(self):
        case = _case(min_schema_valid=True)
        actual = _result(schema_valid=False, correct_finding=False, citation_validity=0.0, required_evidence_recall=None)
        result = check_case(case, actual)
        assert result.passed is False
        assert any("schema_valid" in v for v in result.violations)

    def test_missing_case_from_candidate_reports_missing_not_a_crash(self):
        case = _case(case_id="AC-999")
        result = check_case(case, None)
        assert result.passed is False
        assert result.missing is True
        assert "AC-999" in result.violations[0]


class TestAbstentionBehavior:
    def test_required_abstention_correct_regresses_fails(self):
        case = _case(require_abstention_correct=True)
        actual = _result(abstention_correct=False)
        result = check_case(case, actual)
        assert result.passed is False
        assert any("abstention_correct" in v for v in result.violations)

    def test_required_abstention_correct_holds_passes(self):
        case = _case(require_abstention_correct=True)
        actual = _result(abstention_correct=True)
        result = check_case(case, actual)
        assert result.passed is True

    def test_no_abstention_requirement_means_abstention_is_never_checked(self):
        case = _case(require_abstention_correct=None)
        actual = _result(abstention_correct=False)
        result = check_case(case, actual)
        assert result.passed is True


class TestRunRegressionGateAggregation:
    def test_all_cases_pass_means_gate_passes(self):
        manifest = _manifest([_case(case_id="AC-001"), _case(case_id="AC-002")])
        results = {"AC-001": _result("AC-001"), "AC-002": _result("AC-002")}
        report = run_regression_gate(manifest, "synthetic-v2", results)
        assert report.gate_passed is True
        assert report.passed_cases == 2
        assert report.failed_cases == 0

    def test_one_failing_case_fails_the_whole_gate(self):
        manifest = _manifest([_case(case_id="AC-001"), _case(case_id="AC-002", require_correct_finding=True)])
        results = {"AC-001": _result("AC-001"), "AC-002": _result("AC-002", correct_finding=False)}
        report = run_regression_gate(manifest, "synthetic-v2", results)
        assert report.gate_passed is False
        assert report.failed_cases == 1

    def test_provenance_mismatch_fails_the_gate_even_if_all_cases_would_pass(self):
        manifest = _manifest([_case(case_id="AC-001")])
        results = {"AC-001": _result("AC-001")}
        report = run_regression_gate(manifest, "synthetic-v1", results)  # wrong dataset version
        assert report.provenance_ok is False
        assert report.gate_passed is False

    def test_unknown_candidate_dataset_version_is_not_treated_as_a_mismatch(self):
        manifest = _manifest([_case(case_id="AC-001")])
        results = {"AC-001": _result("AC-001")}
        report = run_regression_gate(manifest, None, results)
        assert report.provenance_ok is True
