"""The regression manifest: a versioned, tracked record of observed
official-run failures, turned into persistent floors a future evaluation
artifact must not fall below.

This is NOT a claim that any listed case is fixed, and it is NOT a
second copy of benchmark ground truth -- ``observed_expected_finding``
and ``observed_should_abstain`` are read straight from the same
``EvaluationCase.expected_outcome`` every other evaluator in this
project already uses (never from hidden model reasoning), and no field
here alters that ground truth. ``require_correct_finding``/
``require_abstention_correct`` are ``True`` ONLY for a case that was
ALREADY correct in the official run being locked in -- the manifest
never demands a case become correct that wasn't already.

Floors (``min_citation_validity``, ``min_required_evidence_recall``,
``min_schema_valid``) are set to exactly what the referenced official run
observed for that case -- never a fabricated or aspirational target --
so the gate's only job is detecting REGRESSION below an already-achieved
baseline, not asserting model quality.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.models.contracts import Finding

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class RegressionCase(BaseModel):
    """One case's regression floor, derived from one official run's
    observed outcome for that case_id.

    ``category`` distinguishes a case whose FINAL FINDING was wrong
    (``finding_failure``) from one whose finding was correct but whose
    evidence grounding degraded (``grounding_degradation``) -- the
    distinction the Phase 8 report was corrected to preserve.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(pattern=r"^AC-\d{3}$")
    category: Literal["finding_failure", "grounding_degradation"]

    observed_expected_finding: Finding
    observed_should_abstain: bool
    observed_actual_finding: Optional[Finding] = None
    observed_correct_finding: bool

    require_correct_finding: bool
    require_abstention_correct: Optional[bool] = None

    min_schema_valid: bool = True
    min_citation_validity: float = Field(ge=0.0, le=1.0)
    min_required_evidence_recall: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    notes: str = Field(min_length=1, max_length=400)


class RegressionManifest(BaseModel):
    """Provenance-anchored collection of ``RegressionCase`` floors.

    ``source_run_id``/``source_artifact_sha256`` point at the exact
    preserved official artifact these floors were derived from -- a
    future reader can verify every floor against that artifact directly
    rather than trusting this file's numbers on faith.
    """

    model_config = ConfigDict(extra="forbid")

    manifest_version: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)

    source_run_id: str = Field(min_length=1)
    source_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_config_id: str = Field(min_length=1)

    cases: list[RegressionCase] = Field(min_length=1)


def load_regression_manifest(path: Path) -> RegressionManifest:
    """Loads and validates a regression manifest file. Raises
    ``OSError`` if unreadable, or ``pydantic.ValidationError`` on a
    schema mismatch -- the caller must treat both as fatal."""

    return RegressionManifest.model_validate_json(path.read_text())
