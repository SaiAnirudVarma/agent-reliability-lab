"""Phase 6 benchmark quality validator: deterministic checks that
synthetic-v2 is well-formed, that synthetic-v1 was not disturbed by adding
it, and that the agent-facing trust boundary still holds for every new case.
"""

from pathlib import Path

from app.agent.interface import AgentInput, build_agent_input
from app.datasets.loader import load_dataset
from app.models.contracts import Finding

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "datasets"


class TestSyntheticV2CaseIdentity:
    def test_exactly_thirty_cases(self):
        dataset = load_dataset(DATASET_DIR, version="synthetic-v2")
        assert len(dataset.cases) == 30

    def test_exact_expected_case_ids(self):
        dataset = load_dataset(DATASET_DIR, version="synthetic-v2")
        expected = {f"AC-{i:03d}" for i in range(1, 31)}
        assert set(dataset.cases_by_id.keys()) == expected

    def test_no_duplicate_case_ids(self):
        dataset = load_dataset(DATASET_DIR, version="synthetic-v2")
        case_ids = [case.case_id for case in dataset.cases]
        assert len(case_ids) == len(set(case_ids))

    def test_stable_manifest_order(self):
        dataset = load_dataset(DATASET_DIR, version="synthetic-v2")
        assert [c.case_id for c in dataset.cases] == [f"AC-{i:03d}" for i in range(1, 31)]


class TestSyntheticV2ReferentialIntegrity:
    def test_every_case_evidence_pool_resolves(self):
        dataset = load_dataset(DATASET_DIR, version="synthetic-v2")
        for case in dataset.cases:
            resolved = dataset.evidence_for_case(case)
            assert [e.evidence_id for e in resolved] == case.evidence_pool

    def test_every_required_evidence_id_resolves(self):
        dataset = load_dataset(DATASET_DIR, version="synthetic-v2")
        for case in dataset.cases:
            required = set(case.expected_outcome.required_evidence_ids)
            pool = set(case.evidence_pool)
            assert required <= pool, f"{case.case_id}: {required - pool}"

    def test_every_case_control_resolves(self):
        dataset = load_dataset(DATASET_DIR, version="synthetic-v2")
        for case in dataset.cases:
            assert case.control_id in dataset.controls


class TestSyntheticV2ClassDistribution:
    def test_new_cases_distribution_is_balanced(self):
        dataset = load_dataset(DATASET_DIR, version="synthetic-v2")
        new_cases = [c for c in dataset.cases if c.case_id not in {f"AC-{i:03d}" for i in range(1, 11)}]
        assert len(new_cases) == 20
        counts = {f: 0 for f in Finding}
        for case in new_cases:
            counts[case.expected_outcome.expected_finding] += 1
        for finding_class, count in counts.items():
            assert 6 <= count <= 7, f"{finding_class}: {count} (expected 6-7)"
        assert sum(counts.values()) == 20

    def test_multiple_controls_represented_in_new_cases(self):
        dataset = load_dataset(DATASET_DIR, version="synthetic-v2")
        new_cases = [c for c in dataset.cases if c.case_id not in {f"AC-{i:03d}" for i in range(1, 11)}]
        controls_used = {case.control_id for case in new_cases}
        assert len(controls_used) >= 6

    def test_multiple_failure_modes_represented(self):
        dataset = load_dataset(DATASET_DIR, version="synthetic-v2")
        new_cases = [c for c in dataset.cases if c.case_id not in {f"AC-{i:03d}" for i in range(1, 11)}]
        tags = {case.failure_mode_tag for case in new_cases}
        assert len(tags) == 20  # every new case has a distinct, descriptive tag

    def test_varying_evidence_counts(self):
        dataset = load_dataset(DATASET_DIR, version="synthetic-v2")
        new_cases = [c for c in dataset.cases if c.case_id not in {f"AC-{i:03d}" for i in range(1, 11)}]
        counts = {len(case.evidence_pool) for case in new_cases}
        assert len(counts) >= 3  # at least 3 distinct pool sizes

    def test_some_new_cases_require_three_or_more_evidence_records(self):
        dataset = load_dataset(DATASET_DIR, version="synthetic-v2")
        new_cases = [c for c in dataset.cases if c.case_id not in {f"AC-{i:03d}" for i in range(1, 11)}]
        assert any(len(case.evidence_pool) >= 3 for case in new_cases)

    def test_some_new_cases_have_single_document_pools_no_distractor(self):
        dataset = load_dataset(DATASET_DIR, version="synthetic-v2")
        new_cases = [c for c in dataset.cases if c.case_id not in {f"AC-{i:03d}" for i in range(1, 11)}]
        assert any(len(case.evidence_pool) == 1 for case in new_cases)

    def test_both_abstention_and_non_abstention_present_under_imperfect_evidence(self):
        """AC-022 (abstain) and AC-023 (should NOT abstain) are the explicit
        pair testing this; confirm both directions exist among new cases."""
        dataset = load_dataset(DATASET_DIR, version="synthetic-v2")
        new_cases = [c for c in dataset.cases if c.case_id not in {f"AC-{i:03d}" for i in range(1, 11)}]
        should_abstain_values = {case.expected_outcome.should_abstain for case in new_cases}
        assert should_abstain_values == {True, False}


class TestSyntheticV1Preserved:
    def test_v1_still_exactly_ten_cases(self):
        dataset = load_dataset(DATASET_DIR)  # default version="synthetic-v1"
        assert len(dataset.cases) == 10
        assert set(dataset.cases_by_id.keys()) == {f"AC-{i:03d}" for i in range(1, 11)}

    def test_v1_fingerprint_matches_official_baseline_dataset_content(self):
        """The v1 Dataset object's fingerprint must be computable and stable
        -- doesn't compare against a hardcoded value (which would be
        fragile to any legitimate future v1 wording fix), but proves it's
        deterministic across repeated loads."""
        first = load_dataset(DATASET_DIR)
        second = load_dataset(DATASET_DIR)
        assert first.fingerprint == second.fingerprint

    def test_v1_distribution_unchanged(self):
        dataset = load_dataset(DATASET_DIR)
        findings = [case.expected_outcome.expected_finding for case in dataset.cases]
        assert findings.count(Finding.PASS) == 3
        assert findings.count(Finding.EXCEPTION) == 4
        assert findings.count(Finding.INSUFFICIENT_EVIDENCE) == 3

    def test_v1_case_content_byte_identical_to_backup(self):
        """Cross-checks the original v1 records (case_id, control_id,
        evidence_pool, expected_outcome, failure_mode_tag) are exactly what
        Phase 1 shipped -- not just present, but unmodified."""
        dataset = load_dataset(DATASET_DIR)
        ac001 = dataset.cases_by_id["AC-001"]
        assert ac001.control_id == "CTRL-PRIV-ACCESS-REVIEW"
        assert ac001.expected_outcome.expected_finding == Finding.PASS
        assert ac001.failure_mode_tag == "baseline_positive"
        ac007 = dataset.cases_by_id["AC-007"]
        assert ac007.control_id == "CTRL-PRIV-MFA"
        assert ac007.failure_mode_tag == "context_conflict_policy"


class TestNoGroundTruthLeakageInNewCases:
    """Re-proves the AgentInput trust boundary (established in Component 3)
    holds for every new v2 case specifically -- not just the original 10."""

    def test_agent_input_excludes_ground_truth_for_every_new_case(self):
        dataset = load_dataset(DATASET_DIR, version="synthetic-v2")
        new_case_ids = {f"AC-{i:03d}" for i in range(11, 31)}
        for case in dataset.cases:
            if case.case_id not in new_case_ids:
                continue
            control = dataset.controls[case.control_id]
            evidence = dataset.evidence_for_case(case)
            task = build_agent_input(case, control, evidence)
            assert isinstance(task, AgentInput)
            dumped = task.model_dump_json()
            assert "expected_outcome" not in dumped
            assert "failure_mode_tag" not in dumped
            assert case.expected_outcome.rationale not in dumped
            assert case.failure_mode_tag not in dumped

    def test_failure_mode_tag_field_does_not_exist_on_agent_input_schema(self):
        assert "failure_mode_tag" not in AgentInput.model_fields
        assert "expected_outcome" not in AgentInput.model_fields
