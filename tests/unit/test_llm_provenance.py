"""End-to-end provenance tests for Component 5's Change 2: a real
RunReport/ExecutionTrace produced by an LLM-mode run must make provider,
model_name, agent_config_id, prompt_version, temperature, and
max_output_tokens all discoverable from the serialized file alone -- and a
deterministic-mode run must never fabricate any of them.

Uses LLMAgent wired to FakeLLMProvider throughout -- no network calls.
"""

from pathlib import Path

from app.agent.deterministic_baseline import DeterministicBaselineAgent
from app.agent.llm_agent import LLMAgent, LLMFindingPayload, build_llm_config
from app.datasets.loader import load_dataset
from app.evaluation.runner import run_evaluation
from app.models.contracts import (
    AgentMode,
    EvidenceReference,
    Finding,
    ModelProvider,
    RunReport,
)
from app.providers.base import ProviderTokenUsage, StructuredCompletion
from tests.support.fake_llm_provider import FakeLLMProvider

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATASET_DIR = REPO_ROOT / "datasets"


def _fixed_completion(finding: Finding, served_model_name: str = "fake-gpt-for-test-2026-03-17") -> StructuredCompletion:
    payload = LLMFindingPayload(
        finding=finding,
        confidence=0.7,
        reasoning_summary="Fixed fake response for provenance testing.",
        evidence=[EvidenceReference(document_id="doc-x", reference="x")] if finding != Finding.INSUFFICIENT_EVIDENCE else [],
        missing_information=[],
        abstain=finding == Finding.INSUFFICIENT_EVIDENCE,
    )
    return StructuredCompletion(
        parsed=payload, raw_text=None, usage=ProviderTokenUsage(150, 60),
        provider=ModelProvider.OPENAI, model_name="fake-gpt-for-test",
        served_model_name=served_model_name,
    )


def _run_llm_evaluation() -> RunReport:
    dataset = load_dataset(REAL_DATASET_DIR)
    provider = FakeLLMProvider(lambda messages, response_model: _fixed_completion(Finding.PASS))
    agent = LLMAgent(provider=provider)
    return run_evaluation(
        dataset, agent, agent_mode=AgentMode.LLM,
        provider=ModelProvider.OPENAI, model_name="fake-gpt-for-test",
        llm_config=build_llm_config(),
    )


class TestLLMRunProvenance:
    def test_report_contains_provider(self):
        report = _run_llm_evaluation()
        assert report.provider == ModelProvider.OPENAI

    def test_report_contains_model_name(self):
        report = _run_llm_evaluation()
        assert report.model_name == "fake-gpt-for-test"

    def test_report_contains_agent_config_id_on_every_trace(self):
        report = _run_llm_evaluation()
        assert all(trace.agent_config_id == "llm-baseline-v1" for trace in report.traces)

    def test_report_contains_temperature(self):
        report = _run_llm_evaluation()
        assert report.llm_config.temperature == 0.0

    def test_report_contains_max_output_tokens(self):
        report = _run_llm_evaluation()
        assert report.llm_config.max_output_tokens == 800

    def test_report_contains_prompt_version(self):
        report = _run_llm_evaluation()
        assert report.llm_config.prompt_version == "llm-baseline-v1"

    def test_every_trace_carries_the_same_llm_config(self):
        report = _run_llm_evaluation()
        for trace in report.traces:
            assert trace.llm_config == report.llm_config
            assert trace.provider == ModelProvider.OPENAI
            assert trace.model_name == "fake-gpt-for-test"

    def test_configuration_recoverable_from_serialized_json_alone(self):
        """Simulates opening a historical result file months later: parse
        the JSON with no access to source code and confirm every piece of
        required provenance is present."""
        report = _run_llm_evaluation()
        raw = report.model_dump_json()
        restored = RunReport.model_validate_json(raw)
        assert restored.provider is not None
        assert restored.model_name is not None
        assert restored.llm_config is not None
        assert restored.llm_config.prompt_version is not None
        assert restored.llm_config.temperature is not None
        assert restored.llm_config.max_output_tokens is not None
        assert all(t.agent_config_id for t in restored.traces)

    def test_json_round_trip_end_to_end(self):
        report = _run_llm_evaluation()
        restored = RunReport.model_validate_json(report.model_dump_json())
        assert restored == report


class TestRequestedVsServedModelProvenance:
    """Proves served_model_name propagates all the way from the provider's
    StructuredCompletion, through LLMAgent.last_served_model_name, through
    run_traced's getattr probe, into the persisted ExecutionTrace -- and
    that it can diverge from the requested model_name without either being
    lost.
    """

    def test_served_model_name_reaches_the_trace(self):
        dataset = load_dataset(REAL_DATASET_DIR)
        provider = FakeLLMProvider(
            lambda messages, response_model: _fixed_completion(
                Finding.PASS, served_model_name="gpt-5.4-mini-2026-03-17-EXACT-SNAPSHOT"
            )
        )
        agent = LLMAgent(provider=provider)
        report = run_evaluation(
            dataset, agent, agent_mode=AgentMode.LLM,
            provider=ModelProvider.OPENAI, model_name="gpt-5.4-mini-2026-03-17",
            llm_config=build_llm_config(),
        )
        for trace in report.traces:
            assert trace.model_name == "gpt-5.4-mini-2026-03-17"  # requested, unchanged
            assert trace.served_model_name == "gpt-5.4-mini-2026-03-17-EXACT-SNAPSHOT"  # served
            assert trace.model_name != trace.served_model_name  # both survive, distinguishable

    def test_agent_last_served_model_name_matches_provider_completion(self):
        provider = FakeLLMProvider(
            lambda messages, response_model: _fixed_completion(Finding.PASS, served_model_name="served-xyz")
        )
        agent = LLMAgent(provider=provider)
        dataset = load_dataset(REAL_DATASET_DIR)
        case = dataset.cases_by_id["AC-001"]
        control = dataset.controls[case.control_id]
        evidence = dataset.evidence_for_case(case)
        from app.agent.interface import build_agent_input

        task = build_agent_input(case, control, evidence)
        agent.run(task)
        assert agent.last_served_model_name == "served-xyz"

    def test_served_model_name_survives_json_round_trip(self):
        dataset = load_dataset(REAL_DATASET_DIR)
        provider = FakeLLMProvider(
            lambda messages, response_model: _fixed_completion(Finding.PASS, served_model_name="served-abc")
        )
        agent = LLMAgent(provider=provider)
        report = run_evaluation(
            dataset, agent, agent_mode=AgentMode.LLM,
            provider=ModelProvider.OPENAI, model_name="requested-model",
            llm_config=build_llm_config(),
        )
        restored = RunReport.model_validate_json(report.model_dump_json())
        assert restored.traces[0].model_name == "requested-model"
        assert restored.traces[0].served_model_name == "served-abc"


class TestDeterministicRunDoesNotFabricateLLMProvenance:
    def test_deterministic_report_has_no_provider_model_or_llm_config(self):
        dataset = load_dataset(REAL_DATASET_DIR)
        report = run_evaluation(dataset, DeterministicBaselineAgent(), agent_mode=AgentMode.MOCK)
        assert report.provider is None
        assert report.model_name is None
        assert report.llm_config is None

    def test_deterministic_traces_have_no_provider_model_or_llm_config(self):
        dataset = load_dataset(REAL_DATASET_DIR)
        report = run_evaluation(dataset, DeterministicBaselineAgent(), agent_mode=AgentMode.MOCK)
        for trace in report.traces:
            assert trace.provider is None
            assert trace.model_name is None
            assert trace.llm_config is None

    def test_deterministic_report_never_writes_placeholder_temperature_or_tokens(self):
        """Not-applicable must be represented as None, never as a fabricated
        temperature=0 / max_output_tokens=0 just to satisfy a schema."""
        dataset = load_dataset(REAL_DATASET_DIR)
        report = run_evaluation(dataset, DeterministicBaselineAgent(), agent_mode=AgentMode.MOCK)
        raw = report.model_dump_json()
        assert '"llm_config":null' in raw.replace(" ", "")


class TestNoSecretsInSerializedProvenance:
    def test_no_api_key_looking_value_in_report(self):
        report = _run_llm_evaluation()
        raw = report.model_dump_json()
        assert "sk-" not in raw
        assert "Authorization" not in raw
        assert "Bearer" not in raw

    def test_llm_config_schema_has_no_credential_fields(self):
        from app.models.contracts import LLMConfig

        field_names = set(LLMConfig.model_fields.keys())
        assert field_names == {"prompt_version", "temperature", "max_output_tokens"}
        for forbidden in ("api_key", "key", "secret", "token", "credential", "authorization"):
            assert forbidden not in field_names
