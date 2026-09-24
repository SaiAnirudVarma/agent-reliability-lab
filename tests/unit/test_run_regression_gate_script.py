"""In-process tests for scripts/run_regression_gate.py -- proving the
CLI returns exit code 0 on a passing candidate and non-zero for each
regression/failure category, entirely offline (fabricated JSON fixture
files in tmp_path; no provider, no dataset load, no real artifact
mutation).

Also includes one live check against the REAL, preserved official
Phase 8 artifact and the REAL frozen manifest -- a self-consistency
proof that the manifest the gate reads matches the run it was derived
from.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_regression_gate.py"
FROZEN_MANIFEST_PATH = REPO_ROOT / "regression" / "regression_manifest_v1.json"
REAL_OFFICIAL_ARTIFACT = (
    REPO_ROOT / "results" / "reranked_llm_experiments"
    / "synthetic-v2__reranked-llm-baseline-v1__12457851-fd75-4f45-8e58-79d95c3e13a1.json"
)

# Loaded once as an independent module object via importlib -- mirrors
# every other script test in this project (e.g.
# tests/unit/test_run_eval_experiments.py) -- never as a subprocess.
_spec = importlib.util.spec_from_file_location("run_regression_gate_test_module", SCRIPT_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
main = _module.main


def _manifest_dict(**overrides) -> dict:
    defaults = dict(
        manifest_version="test-manifest-v1", dataset_version="synthetic-v2",
        source_run_id="run-1", source_artifact_sha256="a" * 64, source_config_id="test-config",
        cases=[
            dict(
                case_id="AC-001", category="finding_failure",
                observed_expected_finding="PASS", observed_should_abstain=False,
                observed_actual_finding="PASS", observed_correct_finding=True,
                require_correct_finding=False, require_abstention_correct=None,
                min_schema_valid=True, min_citation_validity=1.0, min_required_evidence_recall=1.0,
                notes="test case",
            ),
        ],
    )
    defaults.update(overrides)
    return defaults


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data))
    return path


def _candidate_result(
    case_id="AC-001", correct_finding=True, schema_valid=True, citation_validity=1.0,
    required_evidence_recall=1.0, actual_finding="PASS",
) -> dict:
    actual = None
    if schema_valid:
        actual = dict(
            case_id=case_id, finding=actual_finding, confidence=0.9, reasoning_summary="r",
            evidence=[], missing_information=[], abstain=(actual_finding == "INSUFFICIENT_EVIDENCE"),
        )
    return dict(
        case_id=case_id, run_id="candidate-run-1",
        expected=dict(expected_finding="PASS", required_evidence_ids=[], should_abstain=False, rationale="r"),
        actual=actual, correct_finding=correct_finding, schema_valid=schema_valid,
        citation_validity=citation_validity, required_evidence_recall=required_evidence_recall,
        abstention_correct=None,
    )


def _candidate_dict(dataset_version="synthetic-v2", results=None) -> dict:
    return dict(dataset_version=dataset_version, results=results if results is not None else [_candidate_result()])


class TestPassingCandidateFixture:
    def test_exit_code_zero_when_every_floor_holds(self, tmp_path, capsys):
        manifest_path = _write_json(tmp_path / "manifest.json", _manifest_dict())
        candidate_path = _write_json(tmp_path / "candidate.json", _candidate_dict())

        exit_code = main(["--manifest", str(manifest_path), "--candidate", str(candidate_path)])

        assert exit_code == 0
        assert "GATE: PASS" in capsys.readouterr().out


class TestWrongFindingRegressionFixture:
    def test_exit_code_nonzero_when_required_correct_finding_regresses(self, tmp_path, capsys):
        manifest_path = _write_json(
            tmp_path / "manifest.json",
            _manifest_dict(cases=[{**_manifest_dict()["cases"][0], "require_correct_finding": True}]),
        )
        candidate_path = _write_json(
            tmp_path / "candidate.json",
            _candidate_dict(results=[_candidate_result(correct_finding=False, actual_finding="EXCEPTION")]),
        )

        exit_code = main(["--manifest", str(manifest_path), "--candidate", str(candidate_path)])

        stdout = capsys.readouterr().out
        assert exit_code != 0
        assert "GATE: FAIL" in stdout
        assert "correct_finding regressed" in stdout


class TestCitationGroundingRegressionFixture:
    def test_exit_code_nonzero_when_required_evidence_recall_drops_below_floor(self, tmp_path, capsys):
        manifest_path = _write_json(tmp_path / "manifest.json", _manifest_dict())
        candidate_path = _write_json(
            tmp_path / "candidate.json",
            _candidate_dict(results=[_candidate_result(required_evidence_recall=0.0)]),
        )

        exit_code = main(["--manifest", str(manifest_path), "--candidate", str(candidate_path)])

        stdout = capsys.readouterr().out
        assert exit_code != 0
        assert "GATE: FAIL" in stdout
        assert "required_evidence_recall regressed" in stdout


class TestMalformedOrMissingCaseFixture:
    def test_exit_code_nonzero_when_manifest_case_absent_from_candidate(self, tmp_path, capsys):
        manifest_path = _write_json(tmp_path / "manifest.json", _manifest_dict())
        candidate_path = _write_json(tmp_path / "candidate.json", _candidate_dict(results=[]))  # AC-001 missing

        exit_code = main(["--manifest", str(manifest_path), "--candidate", str(candidate_path)])

        stdout = capsys.readouterr().out
        assert exit_code != 0
        assert "MISSING" in stdout

    def test_exit_code_nonzero_when_candidate_case_is_schema_invalid(self, tmp_path, capsys):
        manifest_path = _write_json(tmp_path / "manifest.json", _manifest_dict())
        candidate_path = _write_json(
            tmp_path / "candidate.json",
            _candidate_dict(results=[_candidate_result(schema_valid=False, correct_finding=False, citation_validity=0.0, required_evidence_recall=None)]),
        )

        exit_code = main(["--manifest", str(manifest_path), "--candidate", str(candidate_path)])

        stdout = capsys.readouterr().out
        assert exit_code != 0
        assert "schema_valid regressed" in stdout

    def test_exit_code_nonzero_when_candidate_file_has_no_results_field(self, tmp_path, capsys):
        manifest_path = _write_json(tmp_path / "manifest.json", _manifest_dict())
        candidate_path = _write_json(tmp_path / "candidate.json", {"dataset_version": "synthetic-v2"})

        exit_code = main(["--manifest", str(manifest_path), "--candidate", str(candidate_path)])

        assert exit_code != 0
        assert "results" in capsys.readouterr().err

    def test_exit_code_nonzero_when_candidate_file_is_missing(self, tmp_path, capsys):
        manifest_path = _write_json(tmp_path / "manifest.json", _manifest_dict())

        exit_code = main(["--manifest", str(manifest_path), "--candidate", str(tmp_path / "does-not-exist.json")])

        assert exit_code != 0


class TestProvenanceMismatchFixture:
    def test_exit_code_nonzero_when_candidate_dataset_version_does_not_match_manifest(self, tmp_path, capsys):
        manifest_path = _write_json(tmp_path / "manifest.json", _manifest_dict(dataset_version="synthetic-v2"))
        candidate_path = _write_json(tmp_path / "candidate.json", _candidate_dict(dataset_version="synthetic-v1"))

        exit_code = main(["--manifest", str(manifest_path), "--candidate", str(candidate_path)])

        stdout = capsys.readouterr().out
        assert exit_code != 0
        assert "Provenance OK: False" in stdout


class TestMalformedManifestOrMissingManifestFile:
    def test_exit_code_nonzero_when_manifest_file_is_missing(self, tmp_path):
        candidate_path = _write_json(tmp_path / "candidate.json", _candidate_dict())
        exit_code = main(["--manifest", str(tmp_path / "does-not-exist.json"), "--candidate", str(candidate_path)])
        assert exit_code != 0

    def test_exit_code_nonzero_when_manifest_is_malformed(self, tmp_path):
        manifest_path = _write_json(tmp_path / "manifest.json", {"manifest_version": "x"})
        candidate_path = _write_json(tmp_path / "candidate.json", _candidate_dict())
        exit_code = main(["--manifest", str(manifest_path), "--candidate", str(candidate_path)])
        assert exit_code != 0


class TestLiveSelfConsistencyAgainstTheRealOfficialArtifact:
    """The frozen manifest was DERIVED from this exact preserved
    artifact, so checking the gate against it must pass -- this is a
    read-only sanity check, never a re-run of any experiment."""

    def test_gate_passes_against_the_artifact_it_was_derived_from(self, capsys):
        assert REAL_OFFICIAL_ARTIFACT.exists(), "the preserved Phase 8 artifact must exist for this test"

        exit_code = main(["--manifest", str(FROZEN_MANIFEST_PATH), "--candidate", str(REAL_OFFICIAL_ARTIFACT)])

        stdout = capsys.readouterr().out
        assert exit_code == 0
        assert "GATE: PASS" in stdout
        assert "6/6 passed" in stdout
