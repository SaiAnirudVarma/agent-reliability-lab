"""Unit tests for app.models.contracts.

Covers: happy-path construction of every model, the cross-field validation
rules that encode business logic (abstain/finding consistency, evidence
subset checks, provenance rules), format constraints, and JSON round-trip
serialization of the agent output contract.
"""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.models.contracts import (
    AgentFinding,
    AgentMode,
    Control,
    Evidence,
    EvidenceReference,
    EvaluationCase,
    EvaluationResult,
    ExecutionTrace,
    ExpectedOutcome,
    Finding,
    LLMConfig,
    MetricSummary,
    ModelProvider,
    RunReport,
)


def make_expected_outcome(**overrides) -> ExpectedOutcome:
    defaults = dict(
        expected_finding=Finding.PASS,
        required_evidence_ids=["ev-1"],
        should_abstain=False,
        rationale="Evidence shows timely completion.",
    )
    defaults.update(overrides)
    return ExpectedOutcome(**defaults)


def make_case(**overrides) -> EvaluationCase:
    defaults = dict(
        case_id="AC-001",
        control_id="CTRL-ACCESS-REVIEW",
        scenario_description="Complete, timely access review evidence.",
        evidence_pool=["ev-1", "ev-2"],
        period="2025-Q2",
        expected_outcome=make_expected_outcome(),
        failure_mode_tag="baseline_positive",
    )
    defaults.update(overrides)
    return EvaluationCase(**defaults)


def make_finding(**overrides) -> AgentFinding:
    defaults = dict(
        case_id="AC-001",
        finding=Finding.PASS,
        confidence=0.95,
        reasoning_summary="Access review was completed on time with full evidence.",
        evidence=[EvidenceReference(document_id="ev-1", reference="review-log")],
        missing_information=[],
        abstain=False,
    )
    defaults.update(overrides)
    return AgentFinding(**defaults)


def make_trace(**overrides) -> ExecutionTrace:
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    defaults = dict(
        trace_id="trace-1",
        case_id="AC-001",
        run_id="run-1",
        agent_mode=AgentMode.MOCK,
        provider=None,
        model_name=None,
        llm_config=None,
        agent_config_id="deterministic-baseline-v1",
        evidence_presented=["ev-1", "ev-2"],
        evidence_referenced=["ev-1"],
        started_at=started,
        ended_at=started + timedelta(milliseconds=42),
        latency_ms=42.0,
        output=make_finding(),
        errors=[],
        input_tokens=None,
        output_tokens=None,
        estimated_cost=None,
    )
    defaults.update(overrides)
    return ExecutionTrace(**defaults)


def make_llm_config(**overrides) -> LLMConfig:
    defaults = dict(prompt_version="llm-baseline-v1", temperature=0.0, max_output_tokens=800)
    defaults.update(overrides)
    return LLMConfig(**defaults)


def make_result(**overrides) -> EvaluationResult:
    defaults = dict(
        case_id="AC-001",
        run_id="run-1",
        expected=make_expected_outcome(),
        actual=make_finding(),
        correct_finding=True,
        schema_valid=True,
        citation_validity=1.0,
        abstention_correct=None,
    )
    defaults.update(overrides)
    return EvaluationResult(**defaults)


def make_metrics(**overrides) -> MetricSummary:
    defaults = dict(
        run_id="run-1",
        total_cases=10,
        finding_accuracy=0.9,
        macro_f1=0.88,
        exception_precision=0.8,
        exception_recall=0.75,
        citation_validity=1.0,
        schema_validity=1.0,
        abstention_precision=1.0,
        abstention_recall=1.0,
        abstention_f1=1.0,
        generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return MetricSummary(**defaults)


class TestControl:
    def test_valid_control(self):
        control = Control(
            control_id="CTRL-ACCESS-REVIEW",
            name="Quarterly Access Review",
            description="Reviewers must certify access quarterly.",
            requirement_text="Access must be reviewed every quarter.",
            sla_hours=24,
        )
        assert control.sla_hours == 24

    def test_sla_hours_must_be_positive(self):
        with pytest.raises(ValidationError):
            Control(
                control_id="CTRL-1",
                name="x",
                description="x",
                requirement_text="x",
                sla_hours=0,
            )

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            Control(
                control_id="CTRL-1",
                name="x",
                description="x",
                requirement_text="x",
                unexpected_field="nope",
            )


class TestEvidence:
    def test_valid_evidence(self):
        ev = Evidence(
            evidence_id="ev-1",
            document_id="access_log_q2",
            title="Q2 Access Log",
            period="2025-Q2",
            content="Employee 104 access reviewed on 2025-05-01.",
            structured_fields={"review_date": "2025-05-01"},
        )
        assert ev.structured_fields["review_date"] == "2025-05-01"

    def test_invalid_period_format_rejected(self):
        with pytest.raises(ValidationError):
            Evidence(
                evidence_id="ev-1",
                document_id="doc",
                title="t",
                period="Q2-2025",
                content="content",
            )

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError):
            Evidence(
                evidence_id="ev-1",
                document_id="doc",
                title="t",
                period="2025-Q2",
                content="",
            )


class TestExpectedOutcome:
    def test_valid_outcome(self):
        outcome = make_expected_outcome()
        assert outcome.should_abstain is False

    def test_insufficient_evidence_requires_abstain_true(self):
        with pytest.raises(ValidationError):
            make_expected_outcome(
                expected_finding=Finding.INSUFFICIENT_EVIDENCE,
                should_abstain=False,
            )

    def test_pass_requires_abstain_false(self):
        with pytest.raises(ValidationError):
            make_expected_outcome(expected_finding=Finding.PASS, should_abstain=True)


class TestEvaluationCase:
    def test_valid_case(self):
        case = make_case()
        assert case.case_id == "AC-001"

    def test_case_id_pattern_enforced(self):
        with pytest.raises(ValidationError):
            make_case(case_id="AC-1")

    def test_duplicate_evidence_pool_rejected(self):
        with pytest.raises(ValidationError):
            make_case(evidence_pool=["ev-1", "ev-1"])

    def test_required_evidence_must_be_subset_of_pool(self):
        with pytest.raises(ValidationError):
            make_case(
                evidence_pool=["ev-2"],
                expected_outcome=make_expected_outcome(required_evidence_ids=["ev-1"]),
            )

    def test_empty_evidence_pool_rejected(self):
        with pytest.raises(ValidationError):
            make_case(evidence_pool=[])


class TestAgentFinding:
    def test_valid_finding(self):
        finding = make_finding()
        assert finding.finding == Finding.PASS

    def test_insufficient_evidence_requires_abstain_true(self):
        with pytest.raises(ValidationError):
            make_finding(finding=Finding.INSUFFICIENT_EVIDENCE, abstain=False)

    def test_exception_requires_abstain_false(self):
        with pytest.raises(ValidationError):
            make_finding(finding=Finding.EXCEPTION, abstain=True)

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            make_finding(confidence=1.5)

    def test_reasoning_summary_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            make_finding(reasoning_summary="")

    def test_reasoning_summary_length_capped(self):
        with pytest.raises(ValidationError):
            make_finding(reasoning_summary="x" * 601)

    def test_matches_documented_output_contract_shape(self):
        finding = AgentFinding(
            case_id="AC-008-002",
            finding=Finding.EXCEPTION,
            confidence=0.93,
            reasoning_summary="Concise, non-chain-of-thought explanation.",
            evidence=[EvidenceReference(document_id="access_log_q2", reference="employee-104")],
            missing_information=[],
            abstain=False,
        )
        payload = finding.model_dump(mode="json")
        assert payload["finding"] == "EXCEPTION"
        assert payload["evidence"] == [
            {"document_id": "access_log_q2", "reference": "employee-104"}
        ]

    def test_json_round_trip(self):
        finding = make_finding()
        restored = AgentFinding.model_validate_json(finding.model_dump_json())
        assert restored == finding


class TestLLMConfig:
    def test_valid_config(self):
        config = make_llm_config()
        assert config.prompt_version == "llm-baseline-v1"
        assert config.temperature == 0.0
        assert config.max_output_tokens == 800

    def test_empty_prompt_version_rejected(self):
        with pytest.raises(ValidationError):
            make_llm_config(prompt_version="")

    def test_zero_max_output_tokens_rejected(self):
        with pytest.raises(ValidationError):
            make_llm_config(max_output_tokens=0)

    def test_negative_max_output_tokens_rejected(self):
        with pytest.raises(ValidationError):
            make_llm_config(max_output_tokens=-1)

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            LLMConfig(prompt_version="v1", temperature=0.0, max_output_tokens=800, extra_field="nope")

    def test_json_round_trip(self):
        config = make_llm_config()
        restored = LLMConfig.model_validate_json(config.model_dump_json())
        assert restored == config


class TestExecutionTrace:
    def test_valid_mock_trace(self):
        trace = make_trace()
        assert trace.agent_mode == AgentMode.MOCK

    def test_valid_llm_trace(self):
        trace = make_trace(
            agent_mode=AgentMode.LLM,
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-sonnet-5",
            llm_config=make_llm_config(),
            agent_config_id="llm-agent-v1",
        )
        assert trace.provider == ModelProvider.ANTHROPIC
        assert trace.llm_config.prompt_version == "llm-baseline-v1"

    def test_llm_mode_requires_provider_and_model(self):
        with pytest.raises(ValidationError):
            make_trace(
                agent_mode=AgentMode.LLM, provider=None, model_name=None, llm_config=make_llm_config()
            )

    def test_llm_mode_requires_llm_config(self):
        with pytest.raises(ValidationError):
            make_trace(
                agent_mode=AgentMode.LLM,
                provider=ModelProvider.OPENAI,
                model_name="gpt-4o",
                llm_config=None,
            )

    def test_mock_mode_rejects_llm_config(self):
        with pytest.raises(ValidationError):
            make_trace(agent_mode=AgentMode.MOCK, llm_config=make_llm_config())

    def test_mock_mode_rejects_served_model_name(self):
        with pytest.raises(ValidationError):
            make_trace(agent_mode=AgentMode.MOCK, served_model_name="some-model")

    def test_llm_mode_allows_served_model_name_to_differ_from_requested(self):
        trace = make_trace(
            agent_mode=AgentMode.LLM,
            provider=ModelProvider.OPENAI,
            model_name="gpt-5.4-mini-2026-03-17",
            served_model_name="gpt-5.4-mini-2026-03-17-exact-snapshot",
            llm_config=make_llm_config(),
        )
        assert trace.model_name == "gpt-5.4-mini-2026-03-17"
        assert trace.served_model_name == "gpt-5.4-mini-2026-03-17-exact-snapshot"

    def test_llm_mode_does_not_require_served_model_name(self):
        """served_model_name is best-effort even in LLM mode -- a provider
        may legitimately be unable to confirm it (e.g. an error path)."""
        trace = make_trace(
            agent_mode=AgentMode.LLM,
            provider=ModelProvider.OPENAI,
            model_name="gpt-5.4-mini-2026-03-17",
            served_model_name=None,
            llm_config=make_llm_config(),
        )
        assert trace.served_model_name is None

    def test_llm_mode_requires_both_not_just_one(self):
        with pytest.raises(ValidationError):
            make_trace(
                agent_mode=AgentMode.LLM,
                provider=ModelProvider.OPENAI,
                model_name=None,
            )

    def test_mock_mode_rejects_provider(self):
        with pytest.raises(ValidationError):
            make_trace(agent_mode=AgentMode.MOCK, provider=ModelProvider.OPENAI)

    def test_mock_mode_rejects_model_name(self):
        with pytest.raises(ValidationError):
            make_trace(agent_mode=AgentMode.MOCK, model_name="gpt-4o")

    def test_ended_before_started_rejected(self):
        started = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValidationError):
            make_trace(started_at=started, ended_at=started - timedelta(seconds=1))

    def test_negative_latency_rejected(self):
        with pytest.raises(ValidationError):
            make_trace(latency_ms=-1.0)


class TestEvaluationResult:
    def test_valid_result(self):
        result = make_result()
        assert result.correct_finding is True

    def test_citation_validity_bounds_enforced(self):
        with pytest.raises(ValidationError):
            make_result(citation_validity=1.2)


class TestMetricSummary:
    def test_valid_metrics(self):
        metrics = make_metrics()
        assert metrics.total_cases == 10

    def test_negative_total_cases_rejected(self):
        with pytest.raises(ValidationError):
            make_metrics(total_cases=-1)

    def test_metric_out_of_bounds_rejected(self):
        with pytest.raises(ValidationError):
            make_metrics(macro_f1=1.1)


class TestRunReport:
    def test_valid_mock_report(self):
        report = RunReport(
            run_id="run-1",
            agent_mode=AgentMode.MOCK,
            provider=None,
            model_name=None,
            results=[make_result()],
            traces=[make_trace()],
            metrics=make_metrics(),
            generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        assert report.agent_mode == AgentMode.MOCK

    def test_llm_report_requires_provider_and_model(self):
        with pytest.raises(ValidationError):
            RunReport(
                run_id="run-1",
                agent_mode=AgentMode.LLM,
                provider=None,
                model_name=None,
                results=[make_result()],
                traces=[make_trace()],
                metrics=make_metrics(),
                generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )

    def test_mock_report_rejects_provider(self):
        with pytest.raises(ValidationError):
            RunReport(
                run_id="run-1",
                agent_mode=AgentMode.MOCK,
                provider=ModelProvider.OPENAI,
                model_name=None,
                results=[make_result()],
                traces=[make_trace()],
                metrics=make_metrics(),
                generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )

    def test_valid_llm_report_carries_llm_config(self):
        report = RunReport(
            run_id="run-1",
            agent_mode=AgentMode.LLM,
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o-2024-08-06",
            llm_config=make_llm_config(),
            results=[make_result()],
            traces=[
                make_trace(
                    agent_mode=AgentMode.LLM,
                    provider=ModelProvider.OPENAI,
                    model_name="gpt-4o-2024-08-06",
                    llm_config=make_llm_config(),
                )
            ],
            metrics=make_metrics(),
            generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        assert report.llm_config.temperature == 0.0
        assert report.llm_config.max_output_tokens == 800

    def test_llm_report_requires_llm_config(self):
        with pytest.raises(ValidationError):
            RunReport(
                run_id="run-1",
                agent_mode=AgentMode.LLM,
                provider=ModelProvider.OPENAI,
                model_name="gpt-4o",
                llm_config=None,
                results=[make_result()],
                traces=[make_trace()],
                metrics=make_metrics(),
                generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )

    def test_mock_report_rejects_llm_config(self):
        with pytest.raises(ValidationError):
            RunReport(
                run_id="run-1",
                agent_mode=AgentMode.MOCK,
                llm_config=make_llm_config(),
                results=[make_result()],
                traces=[make_trace()],
                metrics=make_metrics(),
                generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )

    def test_json_round_trip(self):
        report = RunReport(
            run_id="run-1",
            agent_mode=AgentMode.MOCK,
            results=[make_result()],
            traces=[make_trace()],
            metrics=make_metrics(),
            generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        restored = RunReport.model_validate_json(report.model_dump_json())
        assert restored == report

    def test_json_round_trip_with_llm_config(self):
        report = RunReport(
            run_id="run-1",
            agent_mode=AgentMode.LLM,
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o-2024-08-06",
            llm_config=make_llm_config(),
            results=[make_result()],
            traces=[
                make_trace(
                    agent_mode=AgentMode.LLM,
                    provider=ModelProvider.OPENAI,
                    model_name="gpt-4o-2024-08-06",
                    llm_config=make_llm_config(),
                )
            ],
            metrics=make_metrics(),
            generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        restored = RunReport.model_validate_json(report.model_dump_json())
        assert restored == report
