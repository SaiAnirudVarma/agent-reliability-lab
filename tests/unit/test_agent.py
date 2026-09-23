"""Unit tests for the agent input boundary and the deterministic baseline.

Covers: the AgentInput trust boundary (ground truth cannot leak through it),
AgentRunner protocol conformance, the generic rule families exercised both
against the real dataset and against small synthetic fixtures built to
isolate one behavior at a time, citation discipline (only evidence actually
used gets referenced), and two independent safeguards — one static, one
behavioral — against case-ID-keyed decision logic.
"""

import inspect
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

import app.agent.deterministic_baseline as baseline_module
from app.agent.deterministic_baseline import (
    DeterministicBaselineAgent,
    _detect_unavailability,
    _evaluate_disposition,
    _evaluate_effective_policy,
    _evaluate_sla_timeliness,
    _split_by_period,
)
from app.agent.interface import AgentInput, AgentRunner, build_agent_input
from app.datasets.loader import load_dataset
from app.models.contracts import AgentFinding, Control, Evidence, Finding

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATASET_DIR = REPO_ROOT / "datasets"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(REAL_DATASET_DIR)


@pytest.fixture(scope="module")
def agent():
    return DeterministicBaselineAgent()


def real_task(dataset, case_id):
    case = dataset.cases_by_id[case_id]
    control = dataset.controls[case.control_id]
    evidence = dataset.evidence_for_case(case)
    return case, build_agent_input(case, control, evidence)


# ---------------------------------------------------------------------------
# 1-3: the AgentInput trust boundary
# ---------------------------------------------------------------------------


class TestAgentInputTrustBoundary:
    def test_agent_input_schema_has_no_expected_outcome_field(self):
        assert "expected_outcome" not in AgentInput.model_fields

    def test_agent_input_schema_has_no_failure_mode_tag_field(self):
        assert "failure_mode_tag" not in AgentInput.model_fields

    def test_agent_input_schema_has_no_required_evidence_ids_field(self):
        assert "required_evidence_ids" not in AgentInput.model_fields

    def test_agent_input_rejects_smuggled_ground_truth(self):
        """Even a caller that tries to attach ground truth is blocked by the
        schema itself (extra="forbid"), not merely by convention."""
        control = Control(
            control_id="CTRL-X", name="x", description="x", requirement_text="x"
        )
        with pytest.raises(ValidationError):
            AgentInput(
                case_id="AC-001",
                control=control,
                scenario_description="x",
                period="2025-Q1",
                evidence=[],
                expected_outcome={"expected_finding": "PASS"},
            )

    def test_build_agent_input_from_real_case_has_no_ground_truth(self, dataset):
        case, task = real_task(dataset, "AC-004")
        dumped_json = task.model_dump_json()
        assert "expected_outcome" not in dumped_json
        assert "failure_mode_tag" not in dumped_json
        assert case.expected_outcome.rationale not in dumped_json
        assert case.failure_mode_tag not in dumped_json

    def test_build_agent_input_rejects_control_mismatch(self, dataset):
        case = dataset.cases_by_id["AC-004"]
        wrong_control = dataset.controls["CTRL-PRIV-MFA"]
        evidence = dataset.evidence_for_case(case)
        with pytest.raises(ValueError, match="control_id mismatch"):
            build_agent_input(case, wrong_control, evidence)

    def test_build_agent_input_rejects_evidence_mismatch(self, dataset):
        case = dataset.cases_by_id["AC-004"]
        control = dataset.controls[case.control_id]
        wrong_evidence = dataset.evidence_for_case(dataset.cases_by_id["AC-005"])
        with pytest.raises(ValueError, match="evidence mismatch"):
            build_agent_input(case, control, wrong_evidence)


# ---------------------------------------------------------------------------
# 4: AgentRunner protocol conformance
# ---------------------------------------------------------------------------


class TestAgentRunnerProtocol:
    def test_deterministic_baseline_satisfies_agent_runner(self):
        agent = DeterministicBaselineAgent()
        assert isinstance(agent, AgentRunner)

    def test_agent_config_id_is_the_documented_value(self):
        agent = DeterministicBaselineAgent()
        assert agent.agent_config_id == "deterministic-baseline-v1"


# ---------------------------------------------------------------------------
# 5: AC-004 and AC-005 solved through the same generic SLA path
# ---------------------------------------------------------------------------


class TestSlaTimelinessIsGeneric:
    @pytest.mark.parametrize(
        "case_id,expected_elapsed_hours,expected_status",
        [("AC-004", 40.5, "exceeded_sla"), ("AC-005", 18.0, "within_sla")],
    )
    def test_elapsed_hours_computed_by_the_same_function(
        self, dataset, case_id, expected_elapsed_hours, expected_status
    ):
        case = dataset.cases_by_id[case_id]
        control = dataset.controls[case.control_id]
        evidence = dataset.evidence_for_case(case)
        in_period, _ = _split_by_period(case.period, evidence)
        result = _evaluate_sla_timeliness(control, in_period)
        assert result.status == expected_status
        assert result.elapsed_hours == pytest.approx(expected_elapsed_hours, abs=0.01)

    def test_full_run_reaches_the_ground_truth_finding_for_both(self, dataset, agent):
        for case_id in ("AC-004", "AC-005"):
            case, task = real_task(dataset, case_id)
            finding = agent.run(task)
            assert finding.finding == case.expected_outcome.expected_finding

    def test_sla_evaluator_binds_trigger_to_completion_by_subject(self, dataset):
        """AC-004's pool includes EV-403, a same-period termination notice for
        a DIFFERENT employee. The evaluator must not pair EV-403's trigger
        timestamp with EV-402's completion timestamp just because both are
        in-period."""
        case = dataset.cases_by_id["AC-004"]
        control = dataset.controls[case.control_id]
        evidence = dataset.evidence_for_case(case)
        in_period, _ = _split_by_period(case.period, evidence)
        result = _evaluate_sla_timeliness(control, in_period)
        cited_ids = {e.evidence_id for e, _ in result.evidence_used}
        assert cited_ids == {"EV-401", "EV-402"}
        assert "EV-403" not in cited_ids


# ---------------------------------------------------------------------------
# 6: missing required facts cause abstention
# ---------------------------------------------------------------------------


class TestAbstentionOnMissingFacts:
    def test_trigger_without_completion_abstains(self, agent):
        control = Control(
            control_id="CTRL-X",
            name="x",
            description="x",
            requirement_text="x",
            sla_hours=24,
        )
        evidence = [
            Evidence(
                evidence_id="EV-A",
                document_id="doc_a",
                title="Trigger Only",
                period="2025-Q1",
                content="Something happened.",
                structured_fields={"trigger_timestamp": "2025-01-01T00:00:00Z"},
            )
        ]
        task = AgentInput(
            case_id="TEST-1",
            control=control,
            scenario_description="Determine timeliness.",
            period="2025-Q1",
            evidence=evidence,
        )
        finding = agent.run(task)
        assert finding.finding == Finding.INSUFFICIENT_EVIDENCE
        assert finding.abstain is True
        assert finding.missing_information

    def test_no_relevant_evidence_at_all_abstains(self, agent):
        control = Control(control_id="CTRL-X", name="x", description="x", requirement_text="x")
        evidence = [
            Evidence(
                evidence_id="EV-A",
                document_id="doc_a",
                title="Unrelated Memo",
                period="2025-Q1",
                content="An unrelated internal memo.",
                structured_fields={},
            )
        ]
        task = AgentInput(
            case_id="TEST-2",
            control=control,
            scenario_description="Determine compliance.",
            period="2025-Q1",
            evidence=evidence,
        )
        finding = agent.run(task)
        assert finding.finding == Finding.INSUFFICIENT_EVIDENCE
        assert finding.evidence == []


# ---------------------------------------------------------------------------
# 7: wrong-period evidence is not treated as valid current evidence
# ---------------------------------------------------------------------------


class TestWrongPeriodEvidenceIsExcluded:
    def test_clean_out_of_period_evidence_does_not_produce_a_pass(self, agent):
        control = Control(control_id="CTRL-X", name="x", description="x", requirement_text="x")
        out_of_period_clean_evidence = Evidence(
            evidence_id="EV-OLD",
            document_id="doc_old",
            title="Prior Quarter Record",
            period="2025-Q1",
            content="Everything was fine last quarter.",
            structured_fields={"exceptions_found": 0},
        )
        task = AgentInput(
            case_id="TEST-3",
            control=control,
            scenario_description="Determine compliance for the current period.",
            period="2025-Q2",
            evidence=[out_of_period_clean_evidence],
        )
        finding = agent.run(task)
        assert finding.finding == Finding.INSUFFICIENT_EVIDENCE
        assert finding.evidence == []

    def test_real_ac009_excludes_wrong_period_record(self, dataset, agent):
        case, task = real_task(dataset, "AC-009")
        finding = agent.run(task)
        cited_ids = {ref.document_id for ref in finding.evidence}
        assert "firewall_rule_recert_record_2025q1" not in cited_ids
        assert "firewall_rule_recert_record_2025q2" in cited_ids


# ---------------------------------------------------------------------------
# 8: observable tool/access failure leads to abstention
# ---------------------------------------------------------------------------


class TestToolAccessFailureAbstention:
    def test_error_status_evidence_triggers_abstention(self, agent):
        control = Control(control_id="CTRL-X", name="x", description="x", requirement_text="x")
        error_evidence = Evidence(
            evidence_id="EV-ERR",
            document_id="doc_err",
            title="Tool Error Log",
            period="2025-Q2",
            content="The governance tool returned an error.",
            structured_fields={"status": "error", "http_status": 500},
        )
        task = AgentInput(
            case_id="TEST-4",
            control=control,
            scenario_description="Determine compliance.",
            period="2025-Q2",
            evidence=[error_evidence],
        )
        finding = agent.run(task)
        assert finding.finding == Finding.INSUFFICIENT_EVIDENCE
        assert finding.evidence[0].document_id == "doc_err"

    def test_real_ac010_abstains_on_tool_failure(self, dataset, agent):
        case, task = real_task(dataset, "AC-010")
        finding = agent.run(task)
        assert finding.finding == Finding.INSUFFICIENT_EVIDENCE
        assert finding.abstain is True
        cited_ids = {ref.document_id for ref in finding.evidence}
        assert cited_ids == {"db_governance_tool_error_log_2025q2"}


# ---------------------------------------------------------------------------
# 9: superseded policy does not override the currently effective policy
# ---------------------------------------------------------------------------


class TestEffectivePolicyResolution:
    def test_real_ac007_ignores_superseded_policy(self, dataset, agent):
        case, task = real_task(dataset, "AC-007")
        finding = agent.run(task)
        cited_ids = {ref.document_id for ref in finding.evidence}
        assert "mfa_enforcement_policy_v1" not in cited_ids
        assert "mfa_enforcement_policy_v2" in cited_ids
        assert finding.finding == Finding.EXCEPTION

    def test_effective_policy_evaluator_picks_the_window_containing_the_event(self):
        old_policy = Evidence(
            evidence_id="EV-P1",
            document_id="policy_v1",
            title="Policy v1",
            period="2025-Q1",
            content="Old policy.",
            structured_fields={
                "policy_version": "1.0",
                "effective_start": "2020-01-01",
                "effective_end": "2024-12-31",
                "grace_period_days": 30,
            },
        )
        new_policy = Evidence(
            evidence_id="EV-P2",
            document_id="policy_v2",
            title="Policy v2",
            period="2025-Q1",
            content="New policy.",
            structured_fields={
                "policy_version": "2.0",
                "effective_start": "2025-01-01",
                "effective_end": None,
                "grace_period_days": 0,
            },
        )
        fact = Evidence(
            evidence_id="EV-F1",
            document_id="provisioning_fact",
            title="Provisioning Fact",
            period="2025-Q1",
            content="Account provisioned.",
            structured_fields={
                "provisioned_date": "2025-03-01",
                "mfa_enabled_at_provisioning": False,
            },
        )
        result = _evaluate_effective_policy([old_policy, new_policy, fact])
        assert result is not None
        assert result.finding == Finding.EXCEPTION
        cited_ids = {e.evidence_id for e, _ in result.evidence_used}
        assert cited_ids == {"EV-P2", "EV-F1"}
        assert "EV-P1" not in cited_ids


# ---------------------------------------------------------------------------
# 10: multi-document evidence can be combined
# ---------------------------------------------------------------------------


class TestEvidenceSynthesis:
    def test_real_ac008_requires_both_documents(self, dataset):
        case = dataset.cases_by_id["AC-008"]
        control = dataset.controls[case.control_id]
        evidence = dataset.evidence_for_case(case)
        in_period, _ = _split_by_period(case.period, evidence)

        timing_only = [e for e in in_period if e.evidence_id == "EV-801"]
        disposition_only = [e for e in in_period if e.evidence_id == "EV-802"]

        assert _evaluate_disposition(timing_only).status == "not_applicable"
        assert _evaluate_sla_timeliness(control, disposition_only).status == "not_applicable"

        # Only together do they yield a decisive disposition + timeliness result.
        assert _evaluate_disposition(in_period).status == "clean"
        assert _evaluate_sla_timeliness(control, in_period).status == "within_sla"

    def test_full_run_combines_both_documents_into_pass(self, dataset, agent):
        case, task = real_task(dataset, "AC-008")
        finding = agent.run(task)
        assert finding.finding == Finding.PASS
        cited_ids = {ref.document_id for ref in finding.evidence}
        assert cited_ids == {
            "priv_access_review_kickoff_confirmation_2024q4",
            "priv_access_extract_disposition_2024q4",
        }


# ---------------------------------------------------------------------------
# 11: evidence references contain only evidence actually used
# ---------------------------------------------------------------------------


class TestCitationDiscipline:
    @pytest.mark.parametrize(
        "case_id,distractor_document_ids",
        [
            ("AC-001", {"new_hire_provisioning_log_2025q1"}),
            ("AC-002", {"vendor_access_request_form_v3"}),
            ("AC-004", {"hr_termination_notice_10091"}),
            ("AC-005", {"hr_termination_notice_10091"}),
            ("AC-006", {"org_chart_update_2025q4"}),
            ("AC-007", {"mfa_enforcement_policy_v1"}),
            ("AC-008", {"facilities_badge_access_log_2024q4"}),
            ("AC-009", {"firewall_rule_recert_record_2025q1"}),
            ("AC-010", {"db_privileged_role_export_2025q1", "db_privileged_role_export_request_2025q2"}),
        ],
    )
    def test_distractors_are_never_cited(self, dataset, agent, case_id, distractor_document_ids):
        _, task = real_task(dataset, case_id)
        finding = agent.run(task)
        cited_ids = {ref.document_id for ref in finding.evidence}
        assert cited_ids.isdisjoint(distractor_document_ids)

    def test_every_cited_document_was_actually_in_the_input(self, dataset, agent):
        for case in dataset.cases:
            _, task = real_task(dataset, case.case_id)
            finding = agent.run(task)
            available_ids = {e.document_id for e in task.evidence}
            cited_ids = {ref.document_id for ref in finding.evidence}
            assert cited_ids <= available_ids


# ---------------------------------------------------------------------------
# 12: AgentFinding passes Component 1 schema validation
# ---------------------------------------------------------------------------


class TestOutputContractCompliance:
    def test_every_case_produces_a_valid_agent_finding(self, dataset, agent):
        for case in dataset.cases:
            _, task = real_task(dataset, case.case_id)
            finding = agent.run(task)
            assert isinstance(finding, AgentFinding)
            restored = AgentFinding.model_validate_json(finding.model_dump_json())
            assert restored == finding


# ---------------------------------------------------------------------------
# 13: no case-ID-based decision logic
#
# Static inspection cannot mathematically prove the absence of hidden
# case-ID dispatch (a sufficiently obfuscated implementation could hash or
# encode it), so this is deliberately two complementary, weaker-but-honest
# checks: a source-text scan for the literal pattern this rule forbids, and
# a behavioral check that renaming a case's ID does not change its outcome.
# ---------------------------------------------------------------------------


class TestNoCaseIdLogic:
    def test_source_contains_no_case_id_literals_or_comparisons(self):
        source = inspect.getsource(baseline_module)
        assert not re.search(r"AC-\d{3}", source), "found a literal case_id in the rule engine"
        assert "case_id ==" not in source
        assert "case.case_id" not in source
        assert "task.case_id ==" not in source

    def test_relabeled_case_id_does_not_change_the_outcome(self, dataset, agent):
        """Same facts as AC-004, under a case_id the dataset has never used."""
        case = dataset.cases_by_id["AC-004"]
        control = dataset.controls[case.control_id]
        evidence = dataset.evidence_for_case(case)
        real_task_ = build_agent_input(case, control, evidence)
        relabeled_task = AgentInput(
            case_id="ZZ-NOT-A-REAL-CASE-999",
            control=real_task_.control,
            scenario_description=real_task_.scenario_description,
            period=real_task_.period,
            evidence=real_task_.evidence,
        )
        assert agent.run(real_task_).finding == agent.run(relabeled_task).finding == Finding.EXCEPTION


# ---------------------------------------------------------------------------
# Full-dataset accuracy (informational — the honest report is in the write-up)
# ---------------------------------------------------------------------------


class TestFullDatasetRun:
    def test_baseline_matches_ground_truth_on_every_case(self, dataset, agent):
        mismatches = []
        for case in dataset.cases:
            _, task = real_task(dataset, case.case_id)
            finding = agent.run(task)
            if finding.finding != case.expected_outcome.expected_finding:
                mismatches.append((case.case_id, finding.finding, case.expected_outcome.expected_finding))
        assert mismatches == [], f"baseline disagreed with ground truth on: {mismatches}"
