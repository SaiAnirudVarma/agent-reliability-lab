"""The agent-facing input boundary and the AgentRunner interface.

``AgentInput`` is the ONLY view of an ``EvaluationCase`` any ``AgentRunner``
implementation is allowed to see. It structurally excludes every
ground-truth field on ``EvaluationCase`` — ``ExpectedOutcome``,
``required_evidence_ids``, ``rationale``, ``failure_mode_tag`` — not by
programmer discipline, but because those fields have no place in this
model's schema at all: with ``extra="forbid"``, even a caller that tried to
smuggle ground truth into an ``AgentInput`` would hit a validation error.

A later component (the evaluator) is allowed to see the full
``EvaluationCase``, including its ``ExpectedOutcome``. No ``AgentRunner``
implementation ever is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from app.models.contracts import AgentFinding, Control, Evidence, EvaluationCase


class AgentInput(BaseModel):
    """Everything — and only what — a real agent would legitimately receive.

    ``control`` and ``evidence`` are the actual resolved records (not IDs),
    since a real agent reasons over control requirements and evidence
    content/facts, not over dataset primary keys.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    control: Control
    scenario_description: str = Field(min_length=1)
    period: str = Field(pattern=r"^\d{4}-Q[1-4]$")
    evidence: list[Evidence]


def build_agent_input(case: EvaluationCase, control: Control, evidence: list[Evidence]) -> AgentInput:
    """Project an ``EvaluationCase`` down to the safe agent-facing subset.

    Deliberately reads only ``case.case_id``, ``case.scenario_description``
    and ``case.period`` — never ``case.expected_outcome`` or
    ``case.failure_mode_tag``. ``control`` and ``evidence`` are supplied
    already resolved by the caller (typically via
    ``Dataset.evidence_for_case``), so this function has no dependency on
    the dataset loader and cannot itself go looking for ground truth.
    """

    if control.control_id != case.control_id:
        raise ValueError(
            f"control_id mismatch: case {case.case_id} references "
            f"{case.control_id!r} but was given control {control.control_id!r}"
        )
    evidence_ids = {e.evidence_id for e in evidence}
    if evidence_ids != set(case.evidence_pool):
        raise ValueError(
            f"evidence mismatch for {case.case_id}: expected evidence_pool "
            f"{sorted(case.evidence_pool)}, got {sorted(evidence_ids)}"
        )

    return AgentInput(
        case_id=case.case_id,
        control=control,
        scenario_description=case.scenario_description,
        period=case.period,
        evidence=evidence,
    )


class AgentOutputError(Exception):
    """Raised by an AgentRunner that received a response from its underlying
    model/provider but could not turn it into a valid AgentFinding (e.g.
    non-JSON text, JSON missing a required field, or a value that fails one
    of AgentFinding's own cross-field validators).

    This is a MODEL/agent failure, not an infrastructure failure: the
    tracer (app.observability.tracing.run_traced) catches exactly this
    exception type and records a trace with ``output=None`` and
    ``schema_valid=False`` rather than aborting the run. Any other
    exception (authentication, network, misconfiguration, a bug) is left
    uncaught and propagates as a genuine infrastructure failure -- an
    AgentRunner implementation must only raise AgentOutputError around the
    specific step of interpreting a response it already received, never
    around the call that fetches the response in the first place.

    ``raw_output_excerpt`` should contain only what's needed to diagnose
    the parse failure -- never secrets, never hidden provider reasoning
    beyond the provider's own visible completion text -- and should be
    bounded in length (ExecutionTrace.raw_output_excerpt caps it further).
    """

    def __init__(self, message: str, raw_output_excerpt: Optional[str] = None):
        super().__init__(message)
        self.raw_output_excerpt = raw_output_excerpt


@dataclass(frozen=True)
class ExecutionUsage:
    """Real, observed token/cost accounting for one agent execution.

    Distinct from any provider-level usage type: this is what the *agent*
    reports back to the tracer, after applying (or declining to apply, if
    unconfigured) its own cost estimation on top of raw provider token
    counts. ``estimated_cost`` must be ``None`` whenever pricing isn't
    reliably configured -- never a fabricated or guessed number.
    """

    input_tokens: Optional[int]
    output_tokens: Optional[int]
    estimated_cost: Optional[float]


@runtime_checkable
class AgentRunner(Protocol):
    """The interface every agent implementation — mock or real — must satisfy.

    Intentionally minimal: one identity attribute, one method. Future
    implementations (``LLMAgent``, ``LangGraphAgentRunner``,
    ``ContextAwareAgent``) all satisfy this same interface without it
    needing to change.

    An implementation MAY raise ``AgentOutputError`` instead of returning,
    to report a model/agent-level failure (see that class's docstring).

    An implementation MAY additionally expose a ``last_token_usage:
    Optional[ExecutionUsage]`` attribute, set during the most recent call to
    ``run`` (whether it returned or raised ``AgentOutputError``), which
    ``run_traced`` reads via ``getattr(agent, "last_token_usage", None)``
    immediately afterward. This is deliberately NOT part of the Protocol's
    required surface: most agents (e.g. the deterministic baseline) have no
    real token usage to report and simply never define the attribute, and
    requiring every future AgentRunner to implement usage-reporting would
    burden non-LLM agents with a method they have nothing meaningful to
    return from.
    """

    agent_config_id: str

    def run(self, task: AgentInput) -> AgentFinding: ...
