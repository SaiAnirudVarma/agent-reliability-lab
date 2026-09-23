"""Unit tests for app.observability.tracing.run_traced.

Uses a minimal stub AgentRunner (not DeterministicBaselineAgent) for most
tests specifically to prove the tracer is generic over the AgentRunner
protocol -- it never inspects which concrete agent it is wrapping.
"""

from pathlib import Path

import pytest

from app.agent.interface import AgentInput, AgentOutputError, build_agent_input
from app.datasets.loader import load_dataset
from app.models.contracts import (
    AgentFinding,
    AgentMode,
    Control,
    Evidence,
    EvidenceReference,
    ExecutionTrace,
    Finding,
    LLMConfig,
    ModelProvider,
)
from app.observability.tracing import run_traced


def _make_llm_config(**overrides) -> LLMConfig:
    defaults = dict(prompt_version="llm-baseline-v1", temperature=0.0, max_output_tokens=800)
    defaults.update(overrides)
    return LLMConfig(**defaults)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATASET_DIR = REPO_ROOT / "datasets"


class _StubAgent:
    """A minimal AgentRunner used only to prove the tracer is generic."""

    agent_config_id = "stub-agent-v1"

    def __init__(self, finding: AgentFinding):
        self._finding = finding

    def run(self, task: AgentInput) -> AgentFinding:
        return self._finding


def _make_task() -> AgentInput:
    control = Control(control_id="CTRL-X", name="x", description="x", requirement_text="x")
    evidence = [
        Evidence(
            evidence_id="EV-1",
            document_id="doc_1",
            title="Doc 1",
            period="2025-Q1",
            content="content",
            structured_fields={},
        ),
        Evidence(
            evidence_id="EV-2",
            document_id="doc_2",
            title="Doc 2",
            period="2025-Q1",
            content="content",
            structured_fields={},
        ),
    ]
    return AgentInput(
        case_id="TEST-1", control=control, scenario_description="x", period="2025-Q1", evidence=evidence
    )


def _make_finding(evidence_refs: list[EvidenceReference]) -> AgentFinding:
    return AgentFinding(
        case_id="TEST-1",
        finding=Finding.PASS,
        confidence=0.9,
        reasoning_summary="Because.",
        evidence=evidence_refs,
        missing_information=[],
        abstain=False,
    )


class TestTracingSchema:
    def test_execution_trace_schema_has_no_ground_truth_fields(self):
        assert "expected_outcome" not in ExecutionTrace.model_fields
        assert "failure_mode_tag" not in ExecutionTrace.model_fields
        assert "rationale" not in ExecutionTrace.model_fields


class TestRunTracedIsGeneric:
    def test_trace_captures_identity_fields(self):
        task = _make_task()
        agent = _StubAgent(_make_finding([EvidenceReference(document_id="doc_1", reference="x")]))
        trace = run_traced(
            agent, task, run_id="run-123", agent_mode=AgentMode.MOCK, provider=None, model_name=None
        )
        assert trace.run_id == "run-123"
        assert trace.case_id == "TEST-1"
        assert trace.agent_config_id == "stub-agent-v1"
        assert trace.agent_mode == AgentMode.MOCK
        assert trace.provider is None
        assert trace.model_name is None

    def test_timestamps_ordered_and_latency_nonnegative(self):
        task = _make_task()
        agent = _StubAgent(_make_finding([]))
        trace = run_traced(
            agent, task, run_id="run-1", agent_mode=AgentMode.MOCK, provider=None, model_name=None
        )
        assert trace.ended_at >= trace.started_at
        assert trace.latency_ms >= 0.0

    def test_mock_run_has_none_token_and_cost_fields(self):
        task = _make_task()
        agent = _StubAgent(_make_finding([]))
        trace = run_traced(
            agent, task, run_id="run-1", agent_mode=AgentMode.MOCK, provider=None, model_name=None
        )
        assert trace.input_tokens is None
        assert trace.output_tokens is None
        assert trace.estimated_cost is None

    def test_evidence_presented_matches_task_evidence(self):
        task = _make_task()
        agent = _StubAgent(_make_finding([]))
        trace = run_traced(
            agent, task, run_id="run-1", agent_mode=AgentMode.MOCK, provider=None, model_name=None
        )
        assert trace.evidence_presented == ["doc_1", "doc_2"]

    def test_evidence_referenced_derived_from_actual_finding_not_pool(self):
        task = _make_task()
        # Agent cites only doc_1, even though doc_2 was also presented.
        agent = _StubAgent(_make_finding([EvidenceReference(document_id="doc_1", reference="x")]))
        trace = run_traced(
            agent, task, run_id="run-1", agent_mode=AgentMode.MOCK, provider=None, model_name=None
        )
        assert trace.evidence_referenced == ["doc_1"]

    def test_evidence_referenced_preserves_duplicates_and_invalid_citations_raw(self):
        """The tracer records what the agent claimed, unfiltered -- judging
        validity is the evaluator's job, not the tracer's."""
        task = _make_task()
        agent = _StubAgent(
            _make_finding(
                [
                    EvidenceReference(document_id="doc_1", reference="a"),
                    EvidenceReference(document_id="doc_1", reference="b"),
                    EvidenceReference(document_id="doc_not_in_pool", reference="c"),
                ]
            )
        )
        trace = run_traced(
            agent, task, run_id="run-1", agent_mode=AgentMode.MOCK, provider=None, model_name=None
        )
        assert trace.evidence_referenced == ["doc_1", "doc_1", "doc_not_in_pool"]

    def test_llm_mode_carries_provider_and_model(self):
        task = _make_task()
        agent = _StubAgent(_make_finding([]))
        trace = run_traced(
            agent,
            task,
            run_id="run-1",
            agent_mode=AgentMode.LLM,
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-sonnet-5",
            llm_config=_make_llm_config(),
        )
        assert trace.provider == ModelProvider.ANTHROPIC
        assert trace.model_name == "claude-sonnet-5"

    def test_llm_mode_carries_llm_config(self):
        task = _make_task()
        agent = _StubAgent(_make_finding([]))
        config = _make_llm_config(prompt_version="llm-baseline-v1", temperature=0.2, max_output_tokens=500)
        trace = run_traced(
            agent, task, run_id="run-1", agent_mode=AgentMode.LLM,
            provider=ModelProvider.OPENAI, model_name="gpt-4o", llm_config=config,
        )
        assert trace.llm_config.prompt_version == "llm-baseline-v1"
        assert trace.llm_config.temperature == 0.2
        assert trace.llm_config.max_output_tokens == 500

    def test_mock_mode_leaves_llm_config_none(self):
        """Deterministic runs must never fabricate LLM configuration values
        (e.g. temperature=0) just to satisfy a schema."""
        task = _make_task()
        agent = _StubAgent(_make_finding([]))
        trace = run_traced(
            agent, task, run_id="run-1", agent_mode=AgentMode.MOCK, provider=None, model_name=None,
        )
        assert trace.llm_config is None

    def test_tracer_never_branches_on_agent_type(self):
        """Two structurally different AgentRunner implementations produce
        traces through the exact same code path with matching identity."""

        class _OtherStubAgent:
            agent_config_id = "other-stub-v1"

            def run(self, task: AgentInput) -> AgentFinding:
                return _make_finding([])

        task = _make_task()
        for agent in (_StubAgent(_make_finding([])), _OtherStubAgent()):
            trace = run_traced(
                agent, task, run_id="run-1", agent_mode=AgentMode.MOCK, provider=None, model_name=None
            )
            assert trace.agent_config_id == agent.agent_config_id

    def test_agent_output_error_produces_a_trace_with_none_output(self):
        """A malformed model response (AgentOutputError) is a model/agent
        failure, not an infrastructure crash: run_traced must still return a
        valid ExecutionTrace, with output=None and the failure recorded in
        errors, rather than letting the exception propagate."""

        class _FailingAgent:
            agent_config_id = "failing-agent-v1"

            def run(self, task: AgentInput) -> AgentFinding:
                raise AgentOutputError(
                    "could not parse response", raw_output_excerpt="not json"
                )

        task = _make_task()
        trace = run_traced(
            _FailingAgent(), task, run_id="run-1", agent_mode=AgentMode.MOCK, provider=None, model_name=None
        )
        assert trace.output is None
        assert trace.raw_output_excerpt == "not json"
        assert len(trace.errors) == 1
        assert "could not parse response" in trace.errors[0]

    def test_agent_output_error_evidence_referenced_is_empty(self):
        class _FailingAgent:
            agent_config_id = "failing-agent-v1"

            def run(self, task: AgentInput) -> AgentFinding:
                raise AgentOutputError("nope")

        task = _make_task()
        trace = run_traced(
            _FailingAgent(), task, run_id="run-1", agent_mode=AgentMode.MOCK, provider=None, model_name=None
        )
        assert trace.evidence_referenced == []
        assert trace.evidence_presented == ["doc_1", "doc_2"]  # still recorded -- the agent DID see it

    def test_non_agent_output_error_exceptions_still_propagate(self):
        """Anything other than AgentOutputError is a genuine infrastructure
        failure and must NOT be swallowed into a trace."""

        class _CrashingAgent:
            agent_config_id = "crashing-agent-v1"

            def run(self, task: AgentInput) -> AgentFinding:
                raise RuntimeError("network timeout")

        task = _make_task()
        with pytest.raises(RuntimeError, match="network timeout"):
            run_traced(
                _CrashingAgent(), task, run_id="run-1", agent_mode=AgentMode.MOCK,
                provider=None, model_name=None,
            )

    def test_optional_token_usage_side_channel_is_read_when_present(self):
        from app.agent.interface import ExecutionUsage

        class _UsageReportingAgent:
            agent_config_id = "usage-agent-v1"

            def __init__(self):
                self.last_token_usage = None

            def run(self, task: AgentInput) -> AgentFinding:
                self.last_token_usage = ExecutionUsage(
                    input_tokens=123, output_tokens=45, estimated_cost=0.01
                )
                return _make_finding([])

        task = _make_task()
        trace = run_traced(
            _UsageReportingAgent(), task, run_id="run-1", agent_mode=AgentMode.LLM,
            provider=ModelProvider.OPENAI, model_name="gpt-4o", llm_config=_make_llm_config(),
        )
        assert trace.input_tokens == 123
        assert trace.output_tokens == 45
        assert trace.estimated_cost == 0.01

    def test_agent_without_usage_attribute_leaves_token_fields_none(self):
        """DeterministicBaselineAgent-style agents that never define
        last_token_usage must not break the tracer's getattr probe."""
        task = _make_task()
        agent = _StubAgent(_make_finding([]))
        assert not hasattr(agent, "last_token_usage")
        trace = run_traced(
            agent, task, run_id="run-1", agent_mode=AgentMode.MOCK, provider=None, model_name=None
        )
        assert trace.input_tokens is None
        assert trace.output_tokens is None
        assert trace.estimated_cost is None

    def test_real_deterministic_baseline_produces_a_valid_trace(self):
        from app.agent.deterministic_baseline import DeterministicBaselineAgent

        dataset = load_dataset(REAL_DATASET_DIR)
        case = dataset.cases_by_id["AC-004"]
        control = dataset.controls[case.control_id]
        evidence = dataset.evidence_for_case(case)
        task = build_agent_input(case, control, evidence)
        trace = run_traced(
            DeterministicBaselineAgent(),
            task,
            run_id="run-1",
            agent_mode=AgentMode.MOCK,
            provider=None,
            model_name=None,
        )
        assert trace.agent_config_id == "deterministic-baseline-v1"
        assert trace.output.finding == Finding.EXCEPTION
