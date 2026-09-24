"""Unit tests for app.agent.interface.build_retrieved_agent_input --
the Phase 7B retrieval-context AgentInput constructor, and the explicit
proof that it and build_agent_input enforce DIFFERENT evidence boundaries.
"""

from __future__ import annotations

import pytest

from app.agent.interface import AgentInput, build_agent_input, build_retrieved_agent_input
from app.models.contracts import Control, Evidence, EvaluationCase


def _control(control_id="CTRL-X") -> Control:
    return Control(control_id=control_id, name="Test Control", description="A test control.", requirement_text="Do the thing.")


def _evidence(evidence_id: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id, document_id=f"doc_{evidence_id}", title=f"Title {evidence_id}",
        period="2025-Q1", content=f"Content for {evidence_id}.", structured_fields={},
    )


def _case(evidence_pool=None) -> EvaluationCase:
    return EvaluationCase(
        case_id="AC-999", control_id="CTRL-X", scenario_description="Evaluate the thing.",
        evidence_pool=evidence_pool or ["EV-1", "EV-2"], period="2025-Q1",
        expected_outcome={
            "expected_finding": "PASS", "required_evidence_ids": [], "should_abstain": False,
            "rationale": "because",
        },
        failure_mode_tag="tag",
    )


class TestBuildRetrievedAgentInput:
    def test_constructs_with_evidence_outside_the_oracle_pool(self):
        case = _case(evidence_pool=["EV-1", "EV-2"])
        control = _control()
        retrieved = [_evidence("EV-9"), _evidence("EV-8"), _evidence("EV-7")]  # none in evidence_pool

        task = build_retrieved_agent_input(case, control, retrieved)

        assert isinstance(task, AgentInput)
        assert [e.evidence_id for e in task.evidence] == ["EV-9", "EV-8", "EV-7"]

    def test_preserves_given_order_exactly(self):
        case = _case()
        control = _control()
        retrieved = [_evidence("EV-3"), _evidence("EV-1"), _evidence("EV-2")]
        task = build_retrieved_agent_input(case, control, retrieved)
        assert [e.evidence_id for e in task.evidence] == ["EV-3", "EV-1", "EV-2"]

    def test_reads_case_id_scenario_and_period_from_case(self):
        case = _case()
        control = _control()
        task = build_retrieved_agent_input(case, control, [_evidence("EV-1")])
        assert task.case_id == case.case_id
        assert task.scenario_description == case.scenario_description
        assert task.period == case.period

    def test_control_id_mismatch_raises(self):
        case = _case()
        wrong_control = _control(control_id="CTRL-OTHER")
        with pytest.raises(ValueError, match="control_id mismatch"):
            build_retrieved_agent_input(case, wrong_control, [_evidence("EV-1")])

    def test_duplicate_evidence_id_rejected(self):
        case = _case()
        control = _control()
        with pytest.raises(ValueError, match="duplicate evidence_id"):
            build_retrieved_agent_input(case, control, [_evidence("EV-1"), _evidence("EV-1")])

    def test_empty_evidence_list_is_accepted(self):
        """Zero retrieved evidence is a legitimate (if poor) retrieval
        outcome, not a construction error -- AgentInput.evidence has no
        min_length requirement."""
        case = _case()
        control = _control()
        task = build_retrieved_agent_input(case, control, [])
        assert task.evidence == []

    def test_never_reads_expected_outcome_or_failure_mode_tag(self):
        case_a = _case()
        case_b = _case()
        case_b.expected_outcome.rationale  # sanity: still readable on the case object itself
        control = _control()
        task_a = build_retrieved_agent_input(case_a, control, [_evidence("EV-1")])
        dumped = task_a.model_dump_json()
        assert "expected_outcome" not in dumped
        assert "failure_mode_tag" not in dumped
        assert case_a.expected_outcome.rationale not in dumped


class TestOracleAndRetrievalConstructorsEnforceDifferentBoundaries:
    """The explicit test the review called for: prove build_agent_input
    and build_retrieved_agent_input enforce DIFFERENT evidence boundaries
    against the exact same case/control/evidence-set inputs."""

    def test_build_agent_input_rejects_a_non_oracle_evidence_set(self):
        case = _case(evidence_pool=["EV-1", "EV-2"])
        control = _control()
        non_oracle_evidence = [_evidence("EV-9"), _evidence("EV-8")]
        with pytest.raises(ValueError, match="evidence mismatch"):
            build_agent_input(case, control, non_oracle_evidence)

    def test_build_retrieved_agent_input_accepts_the_same_non_oracle_evidence_set(self):
        case = _case(evidence_pool=["EV-1", "EV-2"])
        control = _control()
        non_oracle_evidence = [_evidence("EV-9"), _evidence("EV-8")]
        task = build_retrieved_agent_input(case, control, non_oracle_evidence)
        assert {e.evidence_id for e in task.evidence} == {"EV-9", "EV-8"}

    def test_build_agent_input_still_accepts_the_exact_oracle_pool(self):
        """build_agent_input's own behavior is completely unchanged --
        confirming it was never modified or weakened by this addition."""
        case = _case(evidence_pool=["EV-1", "EV-2"])
        control = _control()
        oracle_evidence = [_evidence("EV-1"), _evidence("EV-2")]
        task = build_agent_input(case, control, oracle_evidence)
        assert {e.evidence_id for e in task.evidence} == {"EV-1", "EV-2"}

    def test_build_retrieved_agent_input_also_accepts_the_exact_oracle_pool(self):
        """Retrieval happening to select exactly the oracle pool is a
        legitimate (if unlikely) outcome, not something the retrieval
        constructor should special-case or reject."""
        case = _case(evidence_pool=["EV-1", "EV-2"])
        control = _control()
        oracle_evidence = [_evidence("EV-1"), _evidence("EV-2")]
        task = build_retrieved_agent_input(case, control, oracle_evidence)
        assert {e.evidence_id for e in task.evidence} == {"EV-1", "EV-2"}
