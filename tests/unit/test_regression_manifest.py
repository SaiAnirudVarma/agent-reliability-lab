"""Unit tests for app.regression.manifest.RegressionManifest -- the
frozen, non-secret regression manifest derived from the preserved
official reranked-LLM baseline (Phase 8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.regression.manifest import RegressionCase, RegressionManifest, load_regression_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_MANIFEST_PATH = REPO_ROOT / "regression" / "regression_manifest_v1.json"


def _valid_case(**overrides) -> dict:
    defaults = dict(
        case_id="AC-001", category="finding_failure",
        observed_expected_finding="PASS", observed_should_abstain=False,
        observed_actual_finding="PASS", observed_correct_finding=True,
        require_correct_finding=False, require_abstention_correct=None,
        min_schema_valid=True, min_citation_validity=1.0, min_required_evidence_recall=1.0,
        notes="test case",
    )
    defaults.update(overrides)
    return defaults


def _valid_manifest_dict(**overrides) -> dict:
    defaults = dict(
        manifest_version="test-manifest-v1", dataset_version="synthetic-v2",
        source_run_id="run-1", source_artifact_sha256="a" * 64, source_config_id="test-config",
        cases=[_valid_case()],
    )
    defaults.update(overrides)
    return defaults


class TestRegressionCaseSchema:
    def test_valid_case_constructs(self):
        case = RegressionCase(**_valid_case())
        assert case.case_id == "AC-001"

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            RegressionCase(**_valid_case(extra_field="x"))

    def test_malformed_case_id_rejected(self):
        with pytest.raises(ValidationError):
            RegressionCase(**_valid_case(case_id="not-a-case-id"))

    def test_bad_category_rejected(self):
        with pytest.raises(ValidationError):
            RegressionCase(**_valid_case(category="something_else"))

    def test_citation_validity_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            RegressionCase(**_valid_case(min_citation_validity=1.5))

    def test_no_ground_truth_rationale_or_reasoning_field_exists(self):
        """The manifest must never carry hidden model reasoning or a
        second, independently-driftable copy of ExpectedOutcome.rationale."""

        forbidden_substrings = ("rationale", "reasoning", "raw_output", "chain_of_thought")
        for field_name in RegressionCase.model_fields:
            lowered = field_name.lower()
            assert not any(bad in lowered for bad in forbidden_substrings), field_name


class TestRegressionManifestSchema:
    def test_valid_manifest_constructs(self):
        manifest = RegressionManifest(**_valid_manifest_dict())
        assert len(manifest.cases) == 1

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            RegressionManifest(**_valid_manifest_dict(extra_field="x"))

    def test_empty_cases_rejected(self):
        with pytest.raises(ValidationError):
            RegressionManifest(**_valid_manifest_dict(cases=[]))

    def test_malformed_source_sha256_rejected(self):
        with pytest.raises(ValidationError):
            RegressionManifest(**_valid_manifest_dict(source_artifact_sha256="not-a-sha"))


class TestLoadFrozenRegressionManifest:
    def test_loads_the_real_frozen_file(self):
        manifest = load_regression_manifest(FROZEN_MANIFEST_PATH)
        assert manifest.manifest_version == "regression-manifest-v1"
        assert manifest.dataset_version == "synthetic-v2"

    def test_provenance_matches_the_official_phase_8_run(self):
        manifest = load_regression_manifest(FROZEN_MANIFEST_PATH)
        assert manifest.source_run_id == "12457851-fd75-4f45-8e58-79d95c3e13a1"
        assert manifest.source_artifact_sha256 == "e8ff346c2c2a28078fc143c77bfa472a8875179602e97486984e73af9020bdc1"
        assert manifest.source_config_id == "reranked-llm-baseline-v1"

    def test_contains_exactly_the_six_documented_cases(self):
        manifest = load_regression_manifest(FROZEN_MANIFEST_PATH)
        case_ids = {c.case_id for c in manifest.cases}
        assert case_ids == {"AC-004", "AC-013", "AC-019", "AC-028", "AC-030", "AC-016"}

    def test_five_finding_failure_cases_do_not_require_correct_finding(self):
        manifest = load_regression_manifest(FROZEN_MANIFEST_PATH)
        finding_failures = [c for c in manifest.cases if c.category == "finding_failure"]
        assert len(finding_failures) == 5
        assert all(not c.require_correct_finding for c in finding_failures)
        assert all(not c.observed_correct_finding for c in finding_failures)

    def test_one_grounding_degradation_case_requires_correct_finding_preserved(self):
        manifest = load_regression_manifest(FROZEN_MANIFEST_PATH)
        grounding_cases = [c for c in manifest.cases if c.category == "grounding_degradation"]
        assert len(grounding_cases) == 1
        assert grounding_cases[0].case_id == "AC-016"
        assert grounding_cases[0].require_correct_finding is True
        assert grounding_cases[0].observed_correct_finding is True

    def test_missing_file_raises_oserror(self, tmp_path):
        with pytest.raises(OSError):
            load_regression_manifest(tmp_path / "does-not-exist.json")

    def test_malformed_file_raises_validation_error(self, tmp_path):
        bad_path = tmp_path / "bad.json"
        bad_path.write_text(json.dumps({"manifest_version": "x"}))
        with pytest.raises(ValidationError):
            load_regression_manifest(bad_path)


class TestFrozenManifestContainsNoSecrets:
    def test_no_known_secret_patterns(self):
        raw = FROZEN_MANIFEST_PATH.read_text()
        assert "OPENAI_API_KEY" not in raw
        assert "sk-" not in raw
