"""Execution tracing for any AgentRunner.

``run_traced`` is the only place in the codebase that constructs an
``ExecutionTrace``. It is written entirely against the ``AgentRunner``
protocol and ``AgentInput`` — it never inspects which concrete agent it is
wrapping (no ``isinstance`` checks, no per-implementation branches), and it
never infers ``agent_mode``/``provider``/``model_name``: the caller supplies
them explicitly, consistent with the project's rule that execution
provenance is recorded, never guessed.

Only observable execution metadata is captured here. There is no field for,
and this function never touches, an agent's internal reasoning beyond the
``reasoning_summary`` already present on the ``AgentFinding`` it returns.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.agent.interface import AgentInput, AgentOutputError, AgentRunner
from app.models.contracts import ExecutionTrace, AgentMode, LLMConfig, ModelProvider


def run_traced(
    agent: AgentRunner,
    task: AgentInput,
    run_id: str,
    agent_mode: AgentMode,
    provider: ModelProvider | None,
    model_name: str | None,
    llm_config: LLMConfig | None = None,
) -> ExecutionTrace:
    """Execute ``agent.run(task)`` once and capture its execution provenance.

    ``provider``/``model_name``/``llm_config`` are, like ``agent_mode``,
    supplied explicitly by the caller and never inferred from ``agent`` --
    the same "declared configuration, not introspected" rule already applied
    to provider/model_name extends naturally to generation parameters
    (temperature, max output tokens, prompt version), since those are also
    pre-declared configuration choices, not something observed from
    execution the way token usage is.

    ``evidence_presented`` and ``evidence_referenced`` are both recorded as
    ``document_id`` strings — the same identifier space an agent's
    ``AgentFinding.evidence`` citations use — so the two lists are directly
    comparable. ``evidence_referenced`` is copied verbatim from the finding,
    including duplicates or citations to documents that were never
    presented: judging whether a citation is *valid* is the evaluator's job
    (Component 4B), not the tracer's — this function only records what
    happened.

    A total INFRASTRUCTURE failure (any exception other than
    ``AgentOutputError`` raised out of ``agent.run``) is deliberately NOT
    caught here — it propagates to the caller unchanged, since there is no
    finding of any kind to build a trace around. ``AgentOutputError``
    specifically (a MODEL/agent failure: a response was received but could
    not be turned into a valid ``AgentFinding``) IS caught, and produces a
    trace with ``output=None``, ``errors`` describing what went wrong, and
    ``raw_output_excerpt`` carrying whatever bounded, secret-free excerpt
    the agent supplied — so a malformed model response never aborts the run.

    Token/cost accounting is read from an OPTIONAL, informally-typed
    ``agent.last_token_usage`` attribute (an ``ExecutionUsage`` or ``None``),
    probed via ``getattr`` so agents with nothing to report (like the
    deterministic baseline) simply never define it. See ``AgentRunner``'s
    docstring for why this is a duck-typed capability rather than a required
    Protocol method. ``served_model_name`` (the model a provider's response
    actually reported, vs. ``model_name`` which is requested/declared) is
    read the same way, via ``agent.last_served_model_name`` -- it's an
    OBSERVED fact from execution, not a pre-declared configuration choice,
    so it belongs with token usage rather than with provider/model_name.
    """

    trace_id = str(uuid4())
    started_at = datetime.now(timezone.utc)

    try:
        output = agent.run(task)
        errors: list[str] = []
        raw_output_excerpt = None
    except AgentOutputError as exc:
        output = None
        errors = [f"{type(exc).__name__}: {exc}"]
        raw_output_excerpt = exc.raw_output_excerpt

    ended_at = datetime.now(timezone.utc)
    latency_ms = (ended_at - started_at).total_seconds() * 1000

    usage = getattr(agent, "last_token_usage", None)
    input_tokens = usage.input_tokens if usage is not None else None
    output_tokens = usage.output_tokens if usage is not None else None
    estimated_cost = usage.estimated_cost if usage is not None else None
    served_model_name = getattr(agent, "last_served_model_name", None)

    evidence_presented = [evidence.document_id for evidence in task.evidence]
    evidence_referenced = (
        [reference.document_id for reference in output.evidence] if output is not None else []
    )

    return ExecutionTrace(
        trace_id=trace_id,
        case_id=task.case_id,
        run_id=run_id,
        agent_mode=agent_mode,
        provider=provider,
        model_name=model_name,
        served_model_name=served_model_name,
        llm_config=llm_config,
        agent_config_id=agent.agent_config_id,
        evidence_presented=evidence_presented,
        evidence_referenced=evidence_referenced,
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=latency_ms,
        output=output,
        errors=errors,
        raw_output_excerpt=raw_output_excerpt,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost=estimated_cost,
    )
