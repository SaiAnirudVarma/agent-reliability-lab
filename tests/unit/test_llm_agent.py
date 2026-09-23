"""Unit tests for app.agent.llm_agent.LLMAgent, using FakeLLMProvider so no
real API calls or credentials are ever needed.
"""

from pathlib import Path

import pytest

from app.agent.interface import AgentInput, AgentOutputError, AgentRunner, build_agent_input
from app.agent.llm_agent import LLMAgent, LLMFindingPayload, build_messages
from app.agent.pricing import PricingConfig
from app.datasets.loader import load_dataset
from app.models.contracts import (
    AgentFinding,
    Control,
    Evidence,
    EvidenceReference,
    Finding,
    ModelProvider,
)
from app.providers.base import ProviderTokenUsage, StructuredCompletion
from tests.support.fake_llm_provider import FakeLLMProvider

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATASET_DIR = REPO_ROOT / "datasets"


def _make_task(case_id: str = "TEST-1") -> AgentInput:
    control = Control(
        control_id="CTRL-X",
        name="Test Control",
        description="A test control.",
        requirement_text="Do the thing.",
    )
    evidence = [
        Evidence(
            evidence_id="EV-1",
            document_id="doc_1",
            title="Doc 1",
            period="2025-Q1",
            content="Some content.",
            structured_fields={"foo": "bar"},
        )
    ]
    return AgentInput(
        case_id=case_id, control=control, scenario_description="Evaluate the thing.",
        period="2025-Q1", evidence=evidence,
    )


def _payload(**overrides) -> LLMFindingPayload:
    defaults = dict(
        finding=Finding.PASS,
        confidence=0.8,
        reasoning_summary="Looks fine.",
        evidence=[EvidenceReference(document_id="doc_1", reference="x")],
        missing_information=[],
        abstain=False,
    )
    defaults.update(overrides)
    return LLMFindingPayload(**defaults)


def _completion(parsed=None, raw_text=None, usage=None) -> StructuredCompletion:
    return StructuredCompletion(
        parsed=parsed, raw_text=raw_text, usage=usage,
        provider=ModelProvider.OPENAI, model_name="fake-model-v1",
    )


class TestLLMAgentSatisfiesAgentRunner:
    def test_isinstance_check(self):
        agent = LLMAgent(provider=FakeLLMProvider(_completion(parsed=_payload())))
        assert isinstance(agent, AgentRunner)

    def test_agent_config_id_is_the_documented_prompt_version(self):
        agent = LLMAgent(provider=FakeLLMProvider(_completion(parsed=_payload())))
        assert agent.agent_config_id == "llm-baseline-v1"


class TestPromptContainsNoGroundTruth:
    def test_prompt_has_no_case_id(self):
        task = _make_task(case_id="AC-004")
        messages = build_messages(task)
        combined = "\n".join(m.content for m in messages)
        assert "AC-004" not in combined

    def test_prompt_has_no_ac_number_pattern_at_all(self):
        """No dataset case ID pattern should ever appear in the prompt --
        this is our generic, un-overfit baseline prompt, not a set of
        per-case instructions."""
        task = _make_task(case_id="AC-007")
        for case_ref in ["AC-001", "AC-002", "AC-003", "AC-004", "AC-005",
                          "AC-006", "AC-007", "AC-008", "AC-009", "AC-010"]:
            combined = "\n".join(m.content for m in build_messages(task))
            assert case_ref not in combined

    def test_prompt_has_no_expected_outcome_fields(self):
        task = _make_task()
        combined = "\n".join(m.content for m in build_messages(task))
        for leak in ["expected_finding", "required_evidence_ids", "should_abstain", "failure_mode_tag"]:
            assert leak not in combined

    def test_prompt_built_from_real_case_contains_only_agent_input_content(self):
        dataset = load_dataset(REAL_DATASET_DIR)
        case = dataset.cases_by_id["AC-007"]
        control = dataset.controls[case.control_id]
        evidence = dataset.evidence_for_case(case)
        task = build_agent_input(case, control, evidence)
        combined = "\n".join(m.content for m in build_messages(task))
        # The case's own rationale (ground truth) must never appear.
        assert case.expected_outcome.rationale not in combined
        assert case.failure_mode_tag not in combined
        # But the scenario description (legitimate input) should.
        assert task.scenario_description in combined


class TestSuccessfulStructuredOutput:
    def test_valid_payload_becomes_agent_finding(self):
        task = _make_task()
        payload = _payload(finding=Finding.EXCEPTION, abstain=False, confidence=0.75)
        provider = FakeLLMProvider(_completion(parsed=payload, usage=ProviderTokenUsage(120, 40)))
        agent = LLMAgent(provider=provider)

        finding = agent.run(task)

        assert isinstance(finding, AgentFinding)
        assert finding.case_id == "TEST-1"  # injected, not model-supplied
        assert finding.finding == Finding.EXCEPTION
        assert finding.confidence == 0.75

    def test_token_usage_captured_when_available(self):
        task = _make_task()
        provider = FakeLLMProvider(_completion(parsed=_payload(), usage=ProviderTokenUsage(200, 55)))
        agent = LLMAgent(provider=provider)
        agent.run(task)
        assert agent.last_token_usage.input_tokens == 200
        assert agent.last_token_usage.output_tokens == 55

    def test_token_usage_none_when_provider_reports_none(self):
        task = _make_task()
        provider = FakeLLMProvider(_completion(parsed=_payload(), usage=None))
        agent = LLMAgent(provider=provider)
        agent.run(task)
        assert agent.last_token_usage.input_tokens is None
        assert agent.last_token_usage.output_tokens is None

    def test_no_cost_without_pricing_config(self):
        task = _make_task()
        provider = FakeLLMProvider(_completion(parsed=_payload(), usage=ProviderTokenUsage(200, 55)))
        agent = LLMAgent(provider=provider, pricing=None)
        agent.run(task)
        assert agent.last_token_usage.estimated_cost is None

    def test_real_cost_with_explicit_pricing_config(self):
        task = _make_task()
        provider = FakeLLMProvider(_completion(parsed=_payload(), usage=ProviderTokenUsage(1000, 1000)))
        pricing = PricingConfig(
            model_name="fake-model-v1",
            input_price_per_1k_tokens=1.0,
            output_price_per_1k_tokens=2.0,
            pricing_as_of="2026-01-01",
        )
        agent = LLMAgent(provider=provider, pricing=pricing)
        agent.run(task)
        assert agent.last_token_usage.estimated_cost == pytest.approx(3.0)


class TestMalformedOutput:
    def test_unparseable_response_raises_agent_output_error(self):
        task = _make_task()
        provider = FakeLLMProvider(
            _completion(parsed=None, raw_text="not valid json", usage=ProviderTokenUsage(90, 10))
        )
        agent = LLMAgent(provider=provider)
        with pytest.raises(AgentOutputError):
            agent.run(task)

    def test_token_usage_still_captured_on_parse_failure(self):
        """The provider call itself succeeded and cost real tokens even
        though the response couldn't be turned into a finding."""
        task = _make_task()
        provider = FakeLLMProvider(
            _completion(parsed=None, raw_text="not valid json", usage=ProviderTokenUsage(90, 10))
        )
        agent = LLMAgent(provider=provider)
        with pytest.raises(AgentOutputError):
            agent.run(task)
        assert agent.last_token_usage.input_tokens == 90
        assert agent.last_token_usage.output_tokens == 10

    def test_schema_conformant_but_cross_validator_violating_payload_raises(self):
        """LLMFindingPayload doesn't itself enforce the abstain/finding
        invariant -- a payload with finding=PASS, abstain=True parses fine
        as a payload, but must fail when LLMAgent builds the real
        AgentFinding, which does enforce it."""
        task = _make_task()
        payload = _payload(finding=Finding.PASS, abstain=True)
        provider = FakeLLMProvider(_completion(parsed=payload, usage=ProviderTokenUsage(50, 20)))
        agent = LLMAgent(provider=provider)
        with pytest.raises(AgentOutputError):
            agent.run(task)

    def test_raw_output_excerpt_is_exposed_on_the_exception(self):
        task = _make_task()
        provider = FakeLLMProvider(
            _completion(parsed=None, raw_text="garbage output", usage=None)
        )
        agent = LLMAgent(provider=provider)
        with pytest.raises(AgentOutputError) as exc_info:
            agent.run(task)
        assert exc_info.value.raw_output_excerpt == "garbage output"


class TestProviderProvenance:
    def test_provider_and_model_name_are_on_the_provider_not_guessed(self):
        provider = FakeLLMProvider(_completion(parsed=_payload()))
        assert provider.provider == ModelProvider.OPENAI
        assert provider.model_name == "fake-model-v1"

    def test_fake_provider_records_call_without_network(self):
        task = _make_task()
        provider = FakeLLMProvider(_completion(parsed=_payload()))
        agent = LLMAgent(provider=provider)
        agent.run(task)
        assert provider.call_count == 1
        assert provider.last_messages is not None
