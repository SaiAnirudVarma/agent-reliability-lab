"""Deterministic aggregate metrics computed from a run's EvaluationResults.

Every formula here is intentionally simple enough to read directly rather
than reaching for a metrics library — this project's ambition is a small,
transparent evaluation harness, not a general-purpose ML metrics package.
sklearn is not used: these are a handful of ratios over at most a few dozen
cases, and a transparent local implementation is more inspectable than a
dependency for a portfolio project meant to be read.

Division-by-zero convention (applied uniformly across every ratio below):
whenever a metric's denominator would be zero, it is defined as ``0.0``
rather than raised as an error. This is deliberately conservative — it
never inflates a score for a condition the run couldn't actually assess —
and mirrors the common ``zero_division=0`` convention scikit-learn offers
for precision/recall/F1, without taking a dependency on scikit-learn for
arithmetic this small. Per-case ``citation_validity`` has its own,
differently-documented zero-case rule (computed in
``app.evaluation.evaluator`` — this module only aggregates already-computed
per-case values). ``required_evidence_recall`` is the one AGGREGATE with a
genuinely different rule from the rest of this module: per-case values of
``None`` (not applicable — see ``compute_required_evidence_recall``) are
EXCLUDED from the mean's denominator entirely, rather than folded in as
0.0; only when there are zero applicable cases does the 0.0 convention
apply, to the aggregate as a whole.

No value in this module is rounded. Rounding happens only at display time
(the CLI) — ``MetricSummary`` always stores full floating-point results.
``percentile`` below is the one exception to "aggregates only": it's a
general-purpose statistic (used for trace latency, not stored in
``MetricSummary``) that lives here for the same reason as everything else
— simple enough to read directly, and directly testable with known arrays.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Optional

from app.models.contracts import EvaluationResult, Finding, MetricSummary


def _predicted_finding(result: EvaluationResult) -> Optional[Finding]:
    """The agent's predicted Finding class, or None if schema validation
    failed and there is no valid AgentFinding to read a class from. A
    schema-invalid case is therefore never a false positive for any class
    (it was never "predicted" as anything) but still correctly drags down
    recall for whichever class the ground truth actually was, since the
    agent produced nothing to credit."""

    return result.actual.finding if result.actual is not None else None


def _predicted_abstain(result: EvaluationResult) -> bool:
    """Whether the agent's output counts as an abstention. A schema-invalid
    case (no valid output at all) is never counted as an abstention -- it
    is simply absent from the "predicted positive" set, for the same reason
    as _predicted_finding above."""

    return result.actual.abstain if result.actual is not None else False


def _binary_precision_recall_f1(
    predicted_positive: Sequence[bool], actual_positive: Sequence[bool]
) -> tuple[float, float, float]:
    """Precision/recall/F1 for one binary condition across a run's cases.

        precision = true_positive / predicted_count   (0.0 if nothing was predicted positive)
        recall    = true_positive / actual_count       (0.0 if nothing was actually positive)
        f1        = 2 * precision * recall / (precision + recall)  (0.0 if both are 0)
    """

    true_positive = sum(
        1 for predicted, actual in zip(predicted_positive, actual_positive) if predicted and actual
    )
    predicted_count = sum(predicted_positive)
    actual_count = sum(actual_positive)

    precision = true_positive / predicted_count if predicted_count > 0 else 0.0
    recall = true_positive / actual_count if actual_count > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def finding_accuracy(results: list[EvaluationResult]) -> float:
    """Correct findings / total cases. 0.0 on an empty run."""

    if not results:
        return 0.0
    return sum(1 for result in results if result.correct_finding) / len(results)


def finding_macro_f1(results: list[EvaluationResult]) -> float:
    """Unweighted mean of per-class F1 across exactly the three Finding
    classes (PASS, EXCEPTION, INSUFFICIENT_EVIDENCE) — always averaged over
    all three regardless of which classes this particular run's ground
    truth happens to contain, so a class the agent never predicts still
    counts against the average via its 0.0 precision/recall rather than
    being silently excluded from the mean.
    """

    f1_scores = []
    for finding_class in Finding:
        predicted_positive = [_predicted_finding(result) == finding_class for result in results]
        actual_positive = [result.expected.expected_finding == finding_class for result in results]
        _, _, f1 = _binary_precision_recall_f1(predicted_positive, actual_positive)
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores)


def exception_precision_recall(results: list[EvaluationResult]) -> tuple[float, float]:
    """Precision/recall of the EXCEPTION class specifically — the
    operationally critical "did we correctly flag a problem" signal, called
    out on its own per the original metrics spec rather than left buried
    inside the macro-F1 average.
    """

    predicted_positive = [_predicted_finding(result) == Finding.EXCEPTION for result in results]
    actual_positive = [result.expected.expected_finding == Finding.EXCEPTION for result in results]
    precision, recall, _ = _binary_precision_recall_f1(predicted_positive, actual_positive)
    return precision, recall


def abstention_precision_recall_f1(results: list[EvaluationResult]) -> tuple[float, float, float]:
    """Precision/recall/F1 of the binary "did the agent abstain" decision
    (``AgentFinding.abstain``) against "should it have"
    (``ExpectedOutcome.should_abstain``). A schema-invalid case (no valid
    output) is never counted as an abstention -- see ``_predicted_abstain``.
    """

    predicted_positive = [_predicted_abstain(result) for result in results]
    actual_positive = [result.expected.should_abstain for result in results]
    return _binary_precision_recall_f1(predicted_positive, actual_positive)


def aggregate_citation_validity(results: list[EvaluationResult]) -> float:
    """Mean of each case's already-computed citation_validity. 0.0 on an
    empty run."""

    if not results:
        return 0.0
    return sum(result.citation_validity for result in results) / len(results)


def aggregate_schema_validity(results: list[EvaluationResult]) -> float:
    """Fraction of cases whose output passed schema validation. 0.0 on an
    empty run."""

    if not results:
        return 0.0
    return sum(1 for result in results if result.schema_valid) / len(results)


def aggregate_required_evidence_recall(results: list[EvaluationResult]) -> float:
    """Mean of each case's already-computed ``required_evidence_recall``,
    over only the cases where it applies (``required_evidence_ids`` was
    non-empty for that case -- see
    ``app.evaluation.evaluator.compute_required_evidence_recall``).

    0.0 if there are ZERO applicable cases in this run -- the same
    empty-denominator convention used throughout this module, applied here
    to "no case in this run had any required evidence at all," which is a
    real (if unusual) condition about the run, not an error.
    """

    applicable = [result.required_evidence_recall for result in results if result.required_evidence_recall is not None]
    if not applicable:
        return 0.0
    return sum(applicable) / len(applicable)


def percentile(values: list[float], pct: float) -> float:
    """The p-th percentile of ``values`` using the nearest-rank method:
    sort ascending, take the smallest value at or above which at most
    ``(1 - pct)`` of the samples fall.

        rank = ceil(pct * n), clamped to [1, n]  (1-indexed)
        return sorted(values)[rank - 1]

    Chosen deliberately over linear interpolation (e.g. numpy's default):
    it always returns an ACTUAL observed value (never a number that didn't
    occur in the data), which matters for a latency report where "P95" is
    read as "an observed run took this long." It also guarantees
    percentile(values, 1.0) == max(values) and percentile(values, 0.0) ==
    min(values) for any non-empty input, including n=1.

    0.0 on an empty input. ``pct`` is expected in [0, 1] (e.g. 0.95 for P95).
    """

    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    rank = max(1, min(n, math.ceil(pct * n)))
    return ordered[rank - 1]


def compute_metric_summary(results: list[EvaluationResult], run_id: str) -> MetricSummary:
    """Aggregate every metric for one run's results into a MetricSummary."""

    exception_precision, exception_recall = exception_precision_recall(results)
    abstention_precision, abstention_recall, abstention_f1 = abstention_precision_recall_f1(results)

    return MetricSummary(
        run_id=run_id,
        total_cases=len(results),
        finding_accuracy=finding_accuracy(results),
        macro_f1=finding_macro_f1(results),
        exception_precision=exception_precision,
        exception_recall=exception_recall,
        citation_validity=aggregate_citation_validity(results),
        required_evidence_recall=aggregate_required_evidence_recall(results),
        schema_validity=aggregate_schema_validity(results),
        abstention_precision=abstention_precision,
        abstention_recall=abstention_recall,
        abstention_f1=abstention_f1,
        generated_at=datetime.now(timezone.utc),
    )
