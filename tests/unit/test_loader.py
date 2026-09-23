"""Unit tests for app.datasets.loader.

Covers the real shipped dataset (all 10 cases, counts, ground-truth
invariants) plus the loader's error handling against small malformed
fixtures written to a tmp_path: duplicate IDs, dangling references,
malformed periods, contradictory abstention labels, and required evidence
missing from the evidence pool.
"""

import json
from pathlib import Path

import pytest

from app.datasets.loader import Dataset, DatasetError, load_dataset
from app.models.contracts import Finding

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATASET_DIR = REPO_ROOT / "datasets"


# ---------------------------------------------------------------------------
# Minimal valid fixture builders, used as a base that individual tests mutate
# to introduce exactly one problem.
# ---------------------------------------------------------------------------


def _base_control() -> dict:
    return {
        "control_id": "CTRL-TEST",
        "name": "Test Control",
        "description": "A control used only in tests.",
        "requirement_text": "Do the thing on time.",
        "sla_hours": 24,
    }


def _base_evidence(evidence_id: str = "EV-TEST-1") -> dict:
    return {
        "evidence_id": evidence_id,
        "document_id": "test_doc",
        "title": "Test Evidence",
        "period": "2025-Q1",
        "content": "Some evidence content.",
        "structured_fields": {},
    }


def _base_case(evidence_pool=None, required_evidence_ids=None) -> dict:
    return {
        "case_id": "AC-999",
        "control_id": "CTRL-TEST",
        "scenario_description": "A test scenario.",
        "evidence_pool": evidence_pool if evidence_pool is not None else ["EV-TEST-1"],
        "period": "2025-Q1",
        "expected_outcome": {
            "expected_finding": "PASS",
            "required_evidence_ids": (
                required_evidence_ids if required_evidence_ids is not None else ["EV-TEST-1"]
            ),
            "should_abstain": False,
            "rationale": "Because the test says so.",
        },
        "failure_mode_tag": "test_tag",
    }


def _write_dataset(tmp_path: Path, controls: list, evidence: list, cases: list) -> Path:
    (tmp_path / "controls.json").write_text(json.dumps(controls))
    (tmp_path / "evidence.json").write_text(json.dumps(evidence))
    (tmp_path / "eval_cases.json").write_text(json.dumps(cases))
    # A minimal version manifest covering every case_id actually written, so
    # load_dataset's default version="synthetic-v1" resolves for these
    # fixtures too. Tests that intend to fail EARLIER (malformed JSON,
    # duplicate IDs, dangling references) still do, since load_dataset
    # checks those before ever consulting versions.json.
    case_ids = [case["case_id"] for case in cases]
    (tmp_path / "versions.json").write_text(
        json.dumps({"synthetic-v1": {"description": "test fixture", "case_ids": case_ids}})
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Real shipped dataset
# ---------------------------------------------------------------------------


class TestRealDataset:
    def test_loads_successfully(self):
        dataset = load_dataset(REAL_DATASET_DIR)
        assert isinstance(dataset, Dataset)

    def test_expected_counts(self):
        dataset = load_dataset(REAL_DATASET_DIR)
        assert len(dataset.controls) == 5
        assert len(dataset.evidence) == 23
        assert len(dataset.cases) == 10

    def test_all_ten_cases_present(self):
        dataset = load_dataset(REAL_DATASET_DIR)
        expected_ids = {f"AC-{i:03d}" for i in range(1, 11)}
        assert set(dataset.cases_by_id.keys()) == expected_ids

    def test_finding_distribution(self):
        dataset = load_dataset(REAL_DATASET_DIR)
        findings = [case.expected_outcome.expected_finding for case in dataset.cases]
        assert findings.count(Finding.PASS) == 3
        assert findings.count(Finding.EXCEPTION) == 4
        assert findings.count(Finding.INSUFFICIENT_EVIDENCE) == 3

    def test_evidence_for_case_resolves_full_records(self):
        dataset = load_dataset(REAL_DATASET_DIR)
        case = dataset.cases_by_id["AC-004"]
        resolved = dataset.evidence_for_case(case)
        assert [e.evidence_id for e in resolved] == case.evidence_pool

    def test_ground_truth_never_appears_in_evidence(self):
        """ExpectedOutcome fields must not leak into any Evidence record's content."""
        dataset = load_dataset(REAL_DATASET_DIR)
        for case in dataset.cases:
            rationale_words = set(case.expected_outcome.rationale.lower().split())
            for evidence in dataset.evidence_for_case(case):
                assert evidence.content != case.expected_outcome.rationale
                # The literal finding label must never appear verbatim in the
                # evidence text presented to the agent.
                assert case.expected_outcome.expected_finding.value not in evidence.content

    def test_abstain_matches_finding_for_every_shipped_case(self):
        """Re-verifies, at the dataset level, an invariant Pydantic already
        enforces per-record — documents that the shipped data actually
        satisfies it, independent of the schema mechanism that guarantees it.
        """
        dataset = load_dataset(REAL_DATASET_DIR)
        for case in dataset.cases:
            expected_abstain = case.expected_outcome.expected_finding == Finding.INSUFFICIENT_EVIDENCE
            assert case.expected_outcome.should_abstain == expected_abstain

    def test_required_evidence_is_subset_of_pool_for_every_shipped_case(self):
        """Same intent as the previous test but for the evidence-subset rule:
        the Pydantic model on EvaluationCase already enforces this at
        construction time. This test instead walks the real, loaded dataset
        and asserts the invariant holds for every case we actually ship,
        so a future edit to eval_cases.json that somehow bypasses
        construction-time validation (e.g. a refactor of the loader) is
        still caught here.
        """
        dataset = load_dataset(REAL_DATASET_DIR)
        for case in dataset.cases:
            required = set(case.expected_outcome.required_evidence_ids)
            pool = set(case.evidence_pool)
            assert required <= pool, f"{case.case_id}: required evidence not in pool: {required - pool}"


# ---------------------------------------------------------------------------
# Error handling against malformed fixtures
# ---------------------------------------------------------------------------


class TestLoaderErrorHandling:
    def test_missing_dataset_file(self, tmp_path):
        (tmp_path / "controls.json").write_text("[]")
        # evidence.json / eval_cases.json intentionally absent
        with pytest.raises(DatasetError, match="not found"):
            load_dataset(tmp_path)

    def test_malformed_json_syntax(self, tmp_path):
        (tmp_path / "controls.json").write_text("{not valid json")
        (tmp_path / "evidence.json").write_text("[]")
        (tmp_path / "eval_cases.json").write_text("[]")
        with pytest.raises(DatasetError, match="not valid JSON"):
            load_dataset(tmp_path)

    def test_top_level_must_be_array(self, tmp_path):
        (tmp_path / "controls.json").write_text(json.dumps({"not": "a list"}))
        (tmp_path / "evidence.json").write_text("[]")
        (tmp_path / "eval_cases.json").write_text("[]")
        with pytest.raises(DatasetError, match="JSON array"):
            load_dataset(tmp_path)

    def test_duplicate_control_id_detected(self, tmp_path):
        controls = [_base_control(), _base_control()]
        _write_dataset(tmp_path, controls, [_base_evidence()], [_base_case()])
        with pytest.raises(DatasetError, match="Duplicate control id"):
            load_dataset(tmp_path)

    def test_duplicate_evidence_id_detected(self, tmp_path):
        evidence = [_base_evidence(), _base_evidence()]
        _write_dataset(tmp_path, [_base_control()], evidence, [_base_case()])
        with pytest.raises(DatasetError, match="Duplicate evidence id"):
            load_dataset(tmp_path)

    def test_duplicate_case_id_detected(self, tmp_path):
        cases = [_base_case(), _base_case()]
        _write_dataset(tmp_path, [_base_control()], [_base_evidence()], cases)
        with pytest.raises(DatasetError, match="Duplicate case id"):
            load_dataset(tmp_path)

    def test_missing_control_reference_detected(self, tmp_path):
        case = _base_case()
        case["control_id"] = "CTRL-DOES-NOT-EXIST"
        _write_dataset(tmp_path, [_base_control()], [_base_evidence()], [case])
        with pytest.raises(DatasetError, match="unknown control_id"):
            load_dataset(tmp_path)

    def test_missing_evidence_reference_detected(self, tmp_path):
        case = _base_case(
            evidence_pool=["EV-DOES-NOT-EXIST"], required_evidence_ids=["EV-DOES-NOT-EXIST"]
        )
        _write_dataset(tmp_path, [_base_control()], [_base_evidence()], [case])
        with pytest.raises(DatasetError, match="unknown evidence id"):
            load_dataset(tmp_path)

    def test_malformed_period_propagates_as_dataset_error(self, tmp_path):
        evidence = _base_evidence()
        evidence["period"] = "Q1-2025"  # wrong format, should be 2025-Q1
        _write_dataset(tmp_path, [_base_control()], [evidence], [_base_case()])
        with pytest.raises(DatasetError, match="evidence.json"):
            load_dataset(tmp_path)

    def test_contradictory_abstention_label_propagates_as_dataset_error(self, tmp_path):
        case = _base_case()
        case["expected_outcome"]["expected_finding"] = "INSUFFICIENT_EVIDENCE"
        case["expected_outcome"]["should_abstain"] = False  # contradiction
        _write_dataset(tmp_path, [_base_control()], [_base_evidence()], [case])
        with pytest.raises(DatasetError, match="eval_cases.json"):
            load_dataset(tmp_path)

    def test_required_evidence_not_in_pool_propagates_as_dataset_error(self, tmp_path):
        case = _base_case(evidence_pool=["EV-TEST-1"], required_evidence_ids=["EV-OTHER"])
        _write_dataset(
            tmp_path,
            [_base_control()],
            [_base_evidence(), _base_evidence("EV-OTHER-UNUSED")],
            [case],
        )
        with pytest.raises(DatasetError, match="eval_cases.json"):
            load_dataset(tmp_path)

    def test_error_message_identifies_the_bad_record(self, tmp_path):
        """Validation errors should name which record and file failed, not
        just 'something went wrong'."""
        evidence = _base_evidence()
        evidence["content"] = ""  # violates min_length=1
        _write_dataset(tmp_path, [_base_control()], [evidence], [_base_case()])
        with pytest.raises(DatasetError) as exc_info:
            load_dataset(tmp_path)
        message = str(exc_info.value)
        assert "evidence.json" in message
        assert "EV-TEST-1" in message

    def test_valid_minimal_dataset_loads(self, tmp_path):
        _write_dataset(tmp_path, [_base_control()], [_base_evidence()], [_base_case()])
        dataset = load_dataset(tmp_path)
        assert dataset.cases_by_id["AC-999"].control_id == "CTRL-TEST"


# ---------------------------------------------------------------------------
# Fingerprint hardening (Phase 6 correction, item F)
# ---------------------------------------------------------------------------


def _write_multi_version_dataset(
    tmp_path: Path, controls: list, evidence: list, cases: list, versions: dict
) -> Path:
    """Like _write_dataset, but with a caller-supplied versions.json mapping
    (multiple named versions over the same shared files), for tests that
    need to prove one version's fingerprint is unaffected by another
    version's resources."""

    (tmp_path / "controls.json").write_text(json.dumps(controls))
    (tmp_path / "evidence.json").write_text(json.dumps(evidence))
    (tmp_path / "eval_cases.json").write_text(json.dumps(cases))
    (tmp_path / "versions.json").write_text(json.dumps(versions))
    return tmp_path


class TestFingerprintHardening:
    def test_identical_content_yields_identical_fingerprint(self, tmp_path):
        _write_dataset(tmp_path, [_base_control()], [_base_evidence()], [_base_case()])
        first = load_dataset(tmp_path)
        second = load_dataset(tmp_path)
        assert first.fingerprint == second.fingerprint

    def test_identical_content_yields_identical_fingerprint_on_synthetic_v2(self):
        """Same invariant as the v1 test in TestSyntheticV1Preserved, proven
        directly for synthetic-v2 too."""
        first = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        second = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        assert first.fingerprint == second.fingerprint

    def test_changing_benchmark_relevant_content_changes_fingerprint(self, tmp_path):
        evidence = _base_evidence()
        _write_dataset(tmp_path, [_base_control()], [evidence], [_base_case()])
        before = load_dataset(tmp_path).fingerprint

        changed_evidence = _base_evidence()
        changed_evidence["content"] = "Some DIFFERENT evidence content."
        _write_dataset(tmp_path, [_base_control()], [changed_evidence], [_base_case()])
        after = load_dataset(tmp_path).fingerprint

        assert before != after

    def test_changing_a_case_rationale_changes_fingerprint(self, tmp_path):
        """Ground truth (ExpectedOutcome) is part of what a dataset version
        reproducibly means, not just the evidence/control text -- a
        rationale edit must move the fingerprint too."""
        _write_dataset(tmp_path, [_base_control()], [_base_evidence()], [_base_case()])
        before = load_dataset(tmp_path).fingerprint

        case = _base_case()
        case["expected_outcome"]["rationale"] = "A different rationale entirely."
        _write_dataset(tmp_path, [_base_control()], [_base_evidence()], [case])
        after = load_dataset(tmp_path).fingerprint

        assert before != after

    def test_v1_fingerprint_stable_when_v2_only_resources_change(self, tmp_path):
        """The exact invariant that makes dataset versioning safe: adding or
        editing a control/evidence/case that ONLY a later version
        references must never move an earlier version's fingerprint."""
        control = _base_control()
        evidence = _base_evidence()
        case = _base_case()
        versions_only_v1 = {"synthetic-v1": {"case_ids": ["AC-999"]}}
        _write_multi_version_dataset(tmp_path, [control], [evidence], [case], versions_only_v1)
        v1_before = load_dataset(tmp_path, version="synthetic-v1").fingerprint

        v2_only_control = _base_control()
        v2_only_control["control_id"] = "CTRL-TEST-V2-ONLY"
        v2_only_evidence = _base_evidence("EV-TEST-V2-ONLY")
        v2_only_case = _base_case()
        v2_only_case["case_id"] = "AC-998"
        v2_only_case["control_id"] = "CTRL-TEST-V2-ONLY"
        v2_only_case["evidence_pool"] = ["EV-TEST-V2-ONLY"]
        v2_only_case["expected_outcome"]["required_evidence_ids"] = ["EV-TEST-V2-ONLY"]

        versions_both = {
            "synthetic-v1": {"case_ids": ["AC-999"]},
            "synthetic-v2": {"case_ids": ["AC-999", "AC-998"]},
        }
        _write_multi_version_dataset(
            tmp_path,
            [control, v2_only_control],
            [evidence, v2_only_evidence],
            [case, v2_only_case],
            versions_both,
        )
        v1_after = load_dataset(tmp_path, version="synthetic-v1").fingerprint
        v2_fingerprint = load_dataset(tmp_path, version="synthetic-v2").fingerprint

        assert v1_before == v1_after
        assert v2_fingerprint != v1_after

    def test_v1_and_v2_fingerprints_differ_on_real_dataset(self):
        v1 = load_dataset(REAL_DATASET_DIR, version="synthetic-v1")
        v2 = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        assert v1.fingerprint != v2.fingerprint
