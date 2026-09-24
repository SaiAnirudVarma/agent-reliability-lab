"""The regression gate: checks a candidate evaluation artifact's
per-case ``EvaluationResult``s against a ``RegressionManifest``'s
floors.

Deliberately generic over WHICH report produced the candidate results:
``app.models.contracts.RunReport`` (the oracle-context LLM baseline) and
``app.integration.runner.RerankedLLMReport`` (Phase 8) both carry a
``dataset_version: str`` and a ``results: list[EvaluationResult]`` --
this module reads exactly those two fields from a raw JSON object and
never imports either report type, so a future third report shape needs
no change here as long as it has the same two fields.

This module makes no provider call, no dataset load, and no benchmark
mutation -- it only compares already-computed ``EvaluationResult``
fields against already-frozen manifest floors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from app.models.contracts import EvaluationResult
from app.regression.manifest import RegressionCase, RegressionManifest


class CandidateLoadError(Exception):
    """Raised when a candidate artifact cannot be parsed into
    ``(dataset_version, results)`` -- e.g. missing/malformed ``results``
    entries. Always fails closed; the gate never partially trusts a
    malformed candidate file."""


class RegressionCaseCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    passed: bool
    missing: bool
    violations: list[str] = Field(default_factory=list)


class RegressionGateReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_version: str = Field(min_length=1)
    candidate_dataset_version: Optional[str] = None
    provenance_ok: bool
    total_cases: int = Field(ge=0)
    passed_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    case_results: list[RegressionCaseCheckResult]
    gate_passed: bool


def load_candidate_results(path: Path) -> tuple[Optional[str], dict[str, EvaluationResult]]:
    """Reads ``path`` as JSON and extracts ``dataset_version`` (if
    present) and ``results`` (validated as ``list[EvaluationResult]``),
    keyed by ``case_id``.

    Deliberately reads the raw JSON object rather than importing
    ``RunReport``/``RerankedLLMReport`` -- see module docstring. Raises
    ``CandidateLoadError`` if ``results`` is missing or any entry fails
    ``EvaluationResult`` validation; raises ``OSError``/``json.JSONDecodeError``
    unchanged if the file cannot be read or parsed at all.
    """

    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or "results" not in raw:
        raise CandidateLoadError(f"{path}: no top-level 'results' field found")

    dataset_version = raw.get("dataset_version")
    try:
        results = TypeAdapter(list[EvaluationResult]).validate_python(raw["results"])
    except Exception as exc:  # pydantic.ValidationError, or a non-list 'results'
        raise CandidateLoadError(f"{path}: 'results' does not validate as list[EvaluationResult]: {exc}") from exc

    results_by_case_id: dict[str, EvaluationResult] = {}
    for result in results:
        results_by_case_id[result.case_id] = result
    return dataset_version, results_by_case_id


def check_case(case: RegressionCase, actual: Optional[EvaluationResult]) -> RegressionCaseCheckResult:
    """Checks one manifest case's floors against the candidate's
    ``EvaluationResult`` for that ``case_id`` (``None`` if the candidate
    has no such case at all -- reported as ``missing``, never a crash).
    """

    if actual is None:
        return RegressionCaseCheckResult(
            case_id=case.case_id, passed=False, missing=True,
            violations=[f"case {case.case_id} is absent from the candidate results"],
        )

    violations: list[str] = []

    if case.min_schema_valid and not actual.schema_valid:
        violations.append("schema_valid regressed: expected True, got False")

    if actual.citation_validity < case.min_citation_validity:
        violations.append(
            f"citation_validity regressed: floor {case.min_citation_validity}, got {actual.citation_validity}"
        )

    if case.min_required_evidence_recall is not None:
        if actual.required_evidence_recall is None or actual.required_evidence_recall < case.min_required_evidence_recall:
            violations.append(
                f"required_evidence_recall regressed: floor {case.min_required_evidence_recall}, "
                f"got {actual.required_evidence_recall}"
            )

    if case.require_correct_finding and not actual.correct_finding:
        violations.append(
            "correct_finding regressed: required True (already achieved by the source run), got False"
        )

    if case.require_abstention_correct is not None:
        if actual.abstention_correct is not case.require_abstention_correct:
            violations.append(
                f"abstention_correct regressed: required {case.require_abstention_correct}, "
                f"got {actual.abstention_correct}"
            )

    return RegressionCaseCheckResult(
        case_id=case.case_id, passed=not violations, missing=False, violations=violations,
    )


def run_regression_gate(
    manifest: RegressionManifest, candidate_dataset_version: Optional[str], results_by_case_id: dict[str, EvaluationResult],
) -> RegressionGateReport:
    """Checks every case in ``manifest`` against ``results_by_case_id``.

    ``provenance_ok`` is ``False`` (and the whole gate fails) when
    ``candidate_dataset_version`` is known and does not match
    ``manifest.dataset_version`` -- checking case-level floors against
    results from the wrong benchmark version would be meaningless. A
    ``None`` candidate dataset_version (an older artifact predating that
    field) is treated as unknown, not a mismatch -- per-case checks
    still run.
    """

    provenance_ok = candidate_dataset_version is None or candidate_dataset_version == manifest.dataset_version

    case_results = [check_case(case, results_by_case_id.get(case.case_id)) for case in manifest.cases]
    passed_cases = sum(1 for cr in case_results if cr.passed)
    failed_cases = len(case_results) - passed_cases

    return RegressionGateReport(
        manifest_version=manifest.manifest_version,
        candidate_dataset_version=candidate_dataset_version,
        provenance_ok=provenance_ok,
        total_cases=len(case_results),
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        case_results=case_results,
        gate_passed=provenance_ok and failed_cases == 0,
    )


def format_report(report: RegressionGateReport) -> str:
    """A concise, human-readable rendering of a RegressionGateReport --
    used by the CLI, and by tests that assert on report content."""

    lines = [
        f"Regression gate: {report.manifest_version}",
        f"Candidate dataset_version: {report.candidate_dataset_version!r}",
        f"Provenance OK: {report.provenance_ok}",
        f"Cases: {report.passed_cases}/{report.total_cases} passed",
        "",
    ]
    for case_result in report.case_results:
        mark = "PASS" if case_result.passed else "FAIL"
        lines.append(f"[{mark}] {case_result.case_id}" + (" (MISSING)" if case_result.missing else ""))
        for violation in case_result.violations:
            lines.append(f"    - {violation}")
    lines.append("")
    lines.append(f"GATE: {'PASS' if report.gate_passed else 'FAIL'}")
    return "\n".join(lines)
