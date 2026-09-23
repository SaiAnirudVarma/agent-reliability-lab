"""Unit tests for app.evaluation.metrics.

Covers perfect predictions, deliberately wrong predictions, macro-F1 across
all three Finding classes, exception/abstention precision-recall-F1, and
every division-by-zero convention this module defines.
"""

import pytest

from app.evaluation.metrics import (
    abstention_precision_recall_f1,
    aggregate_citation_validity,
    aggregate_required_evidence_recall,
    aggregate_schema_validity,
    compute_metric_summary,
    exception_precision_recall,
    finding_accuracy,
    finding_macro_f1,
    percentile,
)
from app.models.contracts import AgentFinding, EvaluationResult, ExpectedOutcome, Finding


def _result(
    expected_finding: Finding,
    actual_finding: Finding,
    should_abstain: bool,
    abstain: bool,
    citation_validity: float = 1.0,
    schema_valid: bool = True,
    required_evidence_recall=None,
) -> EvaluationResult:
    return EvaluationResult(
        case_id="AC-X",
        run_id="run-1",
        expected=ExpectedOutcome(
            expected_finding=expected_finding,
            required_evidence_ids=[],
            should_abstain=should_abstain,
            rationale="x",
        ),
        actual=AgentFinding(
            case_id="AC-X",
            finding=actual_finding,
            confidence=0.9,
            reasoning_summary="x",
            evidence=[],
            missing_information=[],
            abstain=abstain,
        ),
        correct_finding=expected_finding == actual_finding,
        schema_valid=schema_valid,
        citation_validity=citation_validity,
        required_evidence_recall=required_evidence_recall,
        abstention_correct=should_abstain == abstain,
    )


def _schema_invalid_result(expected_finding: Finding, should_abstain: bool) -> EvaluationResult:
    """A result for a case where the agent raised AgentOutputError -- no
    valid AgentFinding was ever produced."""
    return EvaluationResult(
        case_id="AC-X",
        run_id="run-1",
        expected=ExpectedOutcome(
            expected_finding=expected_finding,
            required_evidence_ids=[],
            should_abstain=should_abstain,
            rationale="x",
        ),
        actual=None,
        correct_finding=False,
        schema_valid=False,
        citation_validity=0.0,
        abstention_correct=None,
    )


class TestFindingAccuracy:
    def test_all_correct(self):
        results = [
            _result(Finding.PASS, Finding.PASS, False, False),
            _result(Finding.EXCEPTION, Finding.EXCEPTION, False, False),
        ]
        assert finding_accuracy(results) == 1.0

    def test_all_wrong(self):
        results = [
            _result(Finding.PASS, Finding.EXCEPTION, False, False),
            _result(Finding.EXCEPTION, Finding.PASS, False, False),
        ]
        assert finding_accuracy(results) == 0.0

    def test_partial(self):
        results = [
            _result(Finding.PASS, Finding.PASS, False, False),
            _result(Finding.EXCEPTION, Finding.PASS, False, False),
        ]
        assert finding_accuracy(results) == pytest.approx(0.5)

    def test_empty_results_is_zero_not_an_error(self):
        assert finding_accuracy([]) == 0.0


class TestMacroF1:
    def test_perfect_predictions_across_all_three_classes(self):
        results = [
            _result(Finding.PASS, Finding.PASS, False, False),
            _result(Finding.EXCEPTION, Finding.EXCEPTION, False, False),
            _result(Finding.INSUFFICIENT_EVIDENCE, Finding.INSUFFICIENT_EVIDENCE, True, True),
        ]
        assert finding_macro_f1(results) == 1.0

    def test_completely_wrong_predictions(self):
        results = [
            _result(Finding.PASS, Finding.EXCEPTION, False, False),
            _result(Finding.EXCEPTION, Finding.PASS, False, False),
        ]
        assert finding_macro_f1(results) == 0.0

    def test_class_never_predicted_does_not_raise_and_scores_by_convention(self):
        # Ground truth has one EXCEPTION case, but the agent always predicts
        # PASS. EXCEPTION's precision is 0/0 by raw arithmetic; the 0.0
        # convention applies rather than raising ZeroDivisionError.
        results = [
            _result(Finding.PASS, Finding.PASS, False, False),
            _result(Finding.EXCEPTION, Finding.PASS, False, False),
        ]
        # PASS: predicted=[T,T], actual=[T,F] -> TP=1, precision=1/2, recall=1/1 -> F1=2/3
        # EXCEPTION: predicted=[F,F], actual=[F,T] -> precision=0.0, recall=0.0 -> F1=0.0
        # INSUFFICIENT_EVIDENCE: never predicted, never true -> 0.0
        # macro = (2/3 + 0 + 0) / 3
        assert finding_macro_f1(results) == pytest.approx((2 / 3) / 3, abs=1e-9)

    def test_empty_results_is_zero(self):
        assert finding_macro_f1([]) == 0.0


class TestExceptionPrecisionRecall:
    def test_perfect(self):
        results = [
            _result(Finding.EXCEPTION, Finding.EXCEPTION, False, False),
            _result(Finding.PASS, Finding.PASS, False, False),
        ]
        precision, recall = exception_precision_recall(results)
        assert precision == 1.0
        assert recall == 1.0

    def test_false_positive_lowers_precision_only(self):
        # Predicted EXCEPTION once correctly, once incorrectly (ground truth PASS).
        results = [
            _result(Finding.EXCEPTION, Finding.EXCEPTION, False, False),
            _result(Finding.PASS, Finding.EXCEPTION, False, False),
        ]
        precision, recall = exception_precision_recall(results)
        assert precision == pytest.approx(0.5)
        assert recall == 1.0

    def test_false_negative_lowers_recall_only(self):
        # One true EXCEPTION correctly caught, one missed (predicted PASS
        # instead); no false positives, so precision stays perfect.
        results = [
            _result(Finding.EXCEPTION, Finding.EXCEPTION, False, False),
            _result(Finding.EXCEPTION, Finding.PASS, False, False),
        ]
        precision, recall = exception_precision_recall(results)
        assert precision == 1.0
        assert recall == pytest.approx(0.5)

    def test_no_exception_cases_and_none_predicted_is_zero_by_convention(self):
        results = [_result(Finding.PASS, Finding.PASS, False, False)]
        precision, recall = exception_precision_recall(results)
        assert precision == 0.0
        assert recall == 0.0


class TestAbstentionMetrics:
    def test_perfect_abstention_behavior(self):
        results = [
            _result(Finding.INSUFFICIENT_EVIDENCE, Finding.INSUFFICIENT_EVIDENCE, True, True),
            _result(Finding.PASS, Finding.PASS, False, False),
        ]
        precision, recall, f1 = abstention_precision_recall_f1(results)
        assert (precision, recall, f1) == (1.0, 1.0, 1.0)

    def test_unnecessary_abstention_lowers_precision_not_recall(self):
        # Abstained once correctly (recall satisfied), once when it
        # shouldn't have (an unnecessary refusal).
        results = [
            _result(Finding.INSUFFICIENT_EVIDENCE, Finding.INSUFFICIENT_EVIDENCE, True, True),
            _result(Finding.PASS, Finding.INSUFFICIENT_EVIDENCE, False, True),
        ]
        precision, recall, _ = abstention_precision_recall_f1(results)
        assert precision == pytest.approx(0.5)
        assert recall == 1.0

    def test_missed_abstention_lowers_recall(self):
        # Should have abstained but guessed instead -- unsafe overconfidence.
        results = [_result(Finding.INSUFFICIENT_EVIDENCE, Finding.PASS, True, False)]
        precision, recall, f1 = abstention_precision_recall_f1(results)
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_division_by_zero_when_abstention_never_predicted_or_required(self):
        results = [_result(Finding.PASS, Finding.PASS, False, False)]
        precision, recall, f1 = abstention_precision_recall_f1(results)
        assert (precision, recall, f1) == (0.0, 0.0, 0.0)


class TestAggregateCitationAndSchemaValidity:
    def test_citation_validity_is_the_mean_across_cases(self):
        results = [
            _result(Finding.PASS, Finding.PASS, False, False, citation_validity=1.0),
            _result(Finding.PASS, Finding.PASS, False, False, citation_validity=0.5),
        ]
        assert aggregate_citation_validity(results) == pytest.approx(0.75)

    def test_citation_validity_empty_results_is_zero(self):
        assert aggregate_citation_validity([]) == 0.0

    def test_schema_validity_fraction(self):
        results = [
            _result(Finding.PASS, Finding.PASS, False, False, schema_valid=True),
            _result(Finding.PASS, Finding.PASS, False, False, schema_valid=False),
        ]
        assert aggregate_schema_validity(results) == pytest.approx(0.5)

    def test_schema_validity_empty_results_is_zero(self):
        assert aggregate_schema_validity([]) == 0.0


class TestNoInternalRounding:
    def test_citation_validity_retains_full_precision(self):
        results = [
            _result(Finding.PASS, Finding.PASS, False, False, citation_validity=1 / 3),
            _result(Finding.PASS, Finding.PASS, False, False, citation_validity=1 / 3),
            _result(Finding.PASS, Finding.PASS, False, False, citation_validity=1 / 3),
        ]
        value = aggregate_citation_validity(results)
        assert value == pytest.approx(1 / 3, abs=1e-12)
        assert value != 0.33  # not rounded -- full float precision preserved

    def test_metric_summary_stores_unrounded_values(self):
        # 1/3 exception precision should not come out as a rounded 0.33.
        results = [
            _result(Finding.EXCEPTION, Finding.EXCEPTION, False, False),
            _result(Finding.PASS, Finding.EXCEPTION, False, False),
            _result(Finding.PASS, Finding.EXCEPTION, False, False),
        ]
        summary = compute_metric_summary(results, run_id="run-1")
        assert summary.exception_precision == pytest.approx(1 / 3, abs=1e-12)
        assert summary.exception_precision != 0.33


class TestMetricsWithSchemaInvalidResults:
    """A schema-invalid case (actual=None) must never crash any aggregate,
    must never count as a false positive for any class/abstention, but
    should still drag down recall for its true class."""

    def test_schema_invalid_case_does_not_crash_macro_f1(self):
        results = [
            _result(Finding.PASS, Finding.PASS, False, False),
            _schema_invalid_result(Finding.EXCEPTION, False),
        ]
        score = finding_macro_f1(results)
        assert 0.0 <= score <= 1.0

    def test_schema_invalid_case_lowers_exception_recall_not_precision(self):
        # One correctly-caught EXCEPTION, one EXCEPTION case that came back
        # schema-invalid (agent produced nothing usable).
        results = [
            _result(Finding.EXCEPTION, Finding.EXCEPTION, False, False),
            _schema_invalid_result(Finding.EXCEPTION, False),
        ]
        precision, recall = exception_precision_recall(results)
        assert precision == 1.0  # no false positives introduced
        assert recall == pytest.approx(0.5)  # one of two true EXCEPTIONs was missed

    def test_schema_invalid_case_is_never_counted_as_an_abstention(self):
        results = [_schema_invalid_result(Finding.INSUFFICIENT_EVIDENCE, True)]
        precision, recall, f1 = abstention_precision_recall_f1(results)
        # Not predicted as an abstention at all -> the one true abstention was missed.
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_schema_invalid_case_contributes_zero_to_citation_and_schema_validity(self):
        results = [
            _result(Finding.PASS, Finding.PASS, False, False, citation_validity=1.0),
            _schema_invalid_result(Finding.PASS, False),
        ]
        assert aggregate_citation_validity(results) == pytest.approx(0.5)
        assert aggregate_schema_validity(results) == pytest.approx(0.5)

    def test_finding_accuracy_counts_schema_invalid_as_incorrect(self):
        results = [_schema_invalid_result(Finding.PASS, False)]
        assert finding_accuracy(results) == 0.0

    def test_compute_metric_summary_does_not_crash_on_all_schema_invalid(self):
        results = [
            _schema_invalid_result(Finding.PASS, False),
            _schema_invalid_result(Finding.EXCEPTION, False),
        ]
        summary = compute_metric_summary(results, run_id="run-1")
        assert summary.finding_accuracy == 0.0
        assert summary.schema_validity == 0.0
        assert summary.citation_validity == 0.0


class TestAggregateRequiredEvidenceRecall:
    def test_mean_over_applicable_cases_only(self):
        results = [
            _result(Finding.PASS, Finding.PASS, False, False, required_evidence_recall=1.0),
            _result(Finding.PASS, Finding.PASS, False, False, required_evidence_recall=0.5),
            _result(Finding.PASS, Finding.PASS, False, False, required_evidence_recall=None),  # not applicable
        ]
        # The None case is EXCLUDED from the denominator, not folded in as 0.0:
        # mean of [1.0, 0.5] = 0.75, not mean of [1.0, 0.5, 0.0] = 0.5.
        assert aggregate_required_evidence_recall(results) == pytest.approx(0.75)

    def test_zero_applicable_cases_is_zero_by_convention(self):
        results = [
            _result(Finding.PASS, Finding.PASS, False, False, required_evidence_recall=None),
            _result(Finding.PASS, Finding.PASS, False, False, required_evidence_recall=None),
        ]
        assert aggregate_required_evidence_recall(results) == 0.0

    def test_empty_results_is_zero(self):
        assert aggregate_required_evidence_recall([]) == 0.0

    def test_all_applicable_perfect_recall(self):
        results = [
            _result(Finding.PASS, Finding.PASS, False, False, required_evidence_recall=1.0),
            _result(Finding.EXCEPTION, Finding.EXCEPTION, False, False, required_evidence_recall=1.0),
        ]
        assert aggregate_required_evidence_recall(results) == 1.0


class TestPercentile:
    """Known-array tests for the nearest-rank percentile implementation.
    Directly targets the bug reported against the v1 LLM baseline run: with
    10 samples, the old int()-truncating implementation returned the 9th
    value (index 8) for P95 instead of the max (index 9)."""

    def test_p50_of_one_to_ten(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # rank = ceil(0.5 * 10) = 5 -> sorted[4] = 5
        assert percentile(values, 0.5) == 5

    def test_p95_of_ten_samples_returns_the_max_not_the_second_highest(self):
        """The exact bug: 10 samples, P95 must land on the maximum."""
        values = [1749.0, 2223.7, 2270.4, 2318.1, 2357.2, 2597.9, 2602.5, 2646.2, 3177.2, 4171.9]
        # rank = ceil(0.95 * 10) = 10 -> sorted[9] = max
        assert percentile(values, 0.95) == 4171.9
        assert percentile(values, 0.95) == max(values)

    def test_p100_is_always_the_max(self):
        values = [5, 1, 9, 3, 7]
        assert percentile(values, 1.0) == max(values) == 9

    def test_p0_is_always_the_min(self):
        values = [5, 1, 9, 3, 7]
        assert percentile(values, 0.0) == min(values) == 1

    def test_single_value_returns_that_value_at_any_percentile(self):
        for pct in (0.0, 0.5, 0.95, 1.0):
            assert percentile([42.0], pct) == 42.0

    def test_empty_input_is_zero(self):
        assert percentile([], 0.5) == 0.0

    def test_p50_of_one_to_hundred(self):
        values = list(range(1, 101))
        # rank = ceil(0.5 * 100) = 50 -> sorted[49] = 50
        assert percentile(values, 0.5) == 50

    def test_unsorted_input_handled_correctly(self):
        values = [30, 10, 20, 50, 40]
        assert percentile(values, 1.0) == 50
        assert percentile(values, 0.0) == 10


class TestComputeMetricSummary:
    def test_produces_a_valid_metric_summary(self):
        results = [
            _result(Finding.PASS, Finding.PASS, False, False),
            _result(Finding.EXCEPTION, Finding.EXCEPTION, False, False),
            _result(Finding.INSUFFICIENT_EVIDENCE, Finding.INSUFFICIENT_EVIDENCE, True, True),
        ]
        summary = compute_metric_summary(results, run_id="run-1")
        assert summary.run_id == "run-1"
        assert summary.total_cases == 3
        assert summary.finding_accuracy == 1.0
        assert summary.macro_f1 == 1.0
