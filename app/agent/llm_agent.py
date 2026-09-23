"""The first stochastic LLMAgent, satisfying the same AgentRunner interface
as DeterministicBaselineAgent.

This is the Phase 1 LLM BASELINE: one generic prompt, unmodified after
seeing results, with no case-specific instructions and no knowledge of
expected outcomes. It receives only what AgentInput exposes — the same
control, scenario description, period, and evidence the deterministic
baseline sees — and nothing else.

PROMPT_VERSION identifies this exact prompt + temperature + max-output-tokens
configuration. Any change to the prompt text, TEMPERATURE, or
MAX_OUTPUT_TOKENS below requires bumping PROMPT_VERSION, since — together
with ``build_llm_config()`` below — it is what this system records as
"which configuration produced this run" beyond provider/model_name (see
LLMConfig and ExecutionTrace.agent_config_id).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agent.interface import AgentInput, AgentOutputError, ExecutionUsage
from app.agent.pricing import PricingConfig, estimate_cost
from app.models.contracts import AgentFinding, Evidence, EvidenceReference, Finding, LLMConfig
from app.providers.base import LLMProvider, ProviderMessage

PROMPT_VERSION = "llm-baseline-v1"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 800


def build_llm_config() -> LLMConfig:
    """The LLMConfig describing this module's current, untouched prompt
    configuration. The single source of truth for "what does an LLM trace's
    llm_config look like" -- callers (the CLI, tests) should use this rather
    than re-typing PROMPT_VERSION/TEMPERATURE/MAX_OUTPUT_TOKENS themselves,
    so the two can never drift out of sync.
    """

    return LLMConfig(
        prompt_version=PROMPT_VERSION,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

SYSTEM_PROMPT = """\
You are an internal controls-testing assistant. You will be given one \
control's requirement, a reporting period, and a fixed set of evidence \
documents. Determine whether the control was satisfied for that period, \
based only on the evidence provided.

Rules:
1. Use only the control description and the evidence documents you are \
given. Do not use outside knowledge, and do not invent facts, dates, or \
values that are not present in the evidence.
2. Evidence may include documents from a different reporting period, or \
documents unrelated to this control. A document being present does not \
mean it is relevant — check each document's stated period and content \
against the control's requirement before relying on it.
3. When two or more evidence documents describe conflicting or superseded \
versions of a policy or requirement, determine which one was actually in \
effect for the relevant date, and evaluate compliance against that one only.
4. If the evidence needed to reach a confident conclusion is missing, \
incomplete, contradictory, or was reported unavailable (e.g. a tool or \
system error), respond with INSUFFICIENT_EVIDENCE rather than guessing.
5. Respond with exactly one finding: PASS (the requirement was met), \
EXCEPTION (the requirement was violated), or INSUFFICIENT_EVIDENCE (you \
cannot reach a defensible conclusion from the evidence provided).
6. In the evidence field, list only the document IDs you actually relied \
on to reach your conclusion. Do not list documents you did not use, and \
do not cite a document you were not given.
7. reasoning_summary must be a short, plain-language explanation of your \
conclusion (a few sentences) suitable to show a human reviewer. Do not \
include step-by-step deliberation, alternative hypotheses you considered \
and rejected, or any other internal reasoning — state only the conclusion \
and its concise justification.
8. Set abstain to true if and only if your finding is INSUFFICIENT_EVIDENCE.
"""


class LLMFindingPayload(BaseModel):
    """The subset of AgentFinding fields the model is asked to produce.

    ``case_id`` is deliberately excluded: it is already known before the
    call, so asking the model to echo it back would only invite a spurious
    validation failure (e.g. a mistyped ID) unrelated to the model's actual
    judgment. LLMAgent injects the real case_id after parsing.

    This model intentionally does NOT duplicate AgentFinding's
    abstain/finding cross-field invariant — AgentFinding remains the single
    source of truth for that rule. A payload that is internally
    inconsistent (e.g. finding=PASS with abstain=True) parses fine here and
    is only caught, correctly, when LLMAgent constructs the real
    AgentFinding from it.
    """

    model_config = ConfigDict(extra="forbid")

    finding: Finding
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str = Field(min_length=1, max_length=600)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    abstain: bool


def _render_evidence(evidence: Evidence) -> str:
    lines = [
        f"Document ID: {evidence.document_id}",
        f"Title: {evidence.title}",
        f"Period: {evidence.period}",
        f"Content: {evidence.content}",
    ]
    if evidence.structured_fields:
        fields = ", ".join(f"{key}={value}" for key, value in evidence.structured_fields.items())
        lines.append(f"Structured fields: {fields}")
    return "\n".join(lines)


def _build_user_message(task: AgentInput) -> str:
    control = task.control
    lines = [
        f"Control: {control.name} ({control.control_id})",
        f"Control description: {control.description}",
        f"Control requirement: {control.requirement_text}",
    ]
    if control.sla_hours is not None:
        lines.append(f"Control SLA: {control.sla_hours} hours")
    lines += [
        f"Reporting period under review: {task.period}",
        f"Scenario: {task.scenario_description}",
        "",
        f"Evidence documents ({len(task.evidence)} total):",
    ]
    for evidence in task.evidence:
        lines.append("---")
        lines.append(_render_evidence(evidence))
    return "\n".join(lines)


def build_messages(task: AgentInput) -> list[ProviderMessage]:
    """Build the two-message prompt for one case. Contains no case_id, no
    dataset case identifiers, and nothing derived from ExpectedOutcome —
    only what AgentInput itself exposes."""

    return [
        ProviderMessage(role="system", content=SYSTEM_PROMPT),
        ProviderMessage(role="user", content=_build_user_message(task)),
    ]


class LLMAgent:
    """An AgentRunner backed by a hosted LLM via the LLMProvider seam.

    Reasons only over ``task`` (via ``build_messages``) — it never receives
    ExpectedOutcome, required_evidence_ids, or failure_mode_tag, because
    AgentInput structurally cannot carry them.
    """

    agent_config_id = PROMPT_VERSION

    def __init__(self, provider: LLMProvider, pricing: Optional[PricingConfig] = None):
        self._provider = provider
        self._pricing = pricing
        self.last_token_usage: Optional[ExecutionUsage] = None
        self.last_served_model_name: Optional[str] = None

    def run(self, task: AgentInput) -> AgentFinding:
        messages = build_messages(task)
        completion = self._provider.complete_structured(messages, LLMFindingPayload)

        input_tokens = completion.usage.input_tokens if completion.usage is not None else None
        output_tokens = completion.usage.output_tokens if completion.usage is not None else None
        cost = estimate_cost(input_tokens, output_tokens, self._pricing)
        # Recorded regardless of what happens next: the provider call itself
        # already consumed these tokens even if parsing below fails.
        self.last_token_usage = ExecutionUsage(input_tokens, output_tokens, cost)
        # Like last_token_usage, this is an OBSERVED fact about execution
        # (what the provider's response actually reported), not a
        # pre-declared configuration choice -- so it's exposed the same way,
        # for run_traced to read via getattr rather than as an explicit
        # caller-supplied parameter.
        self.last_served_model_name = completion.served_model_name

        if completion.parsed is None:
            raw_excerpt = completion.raw_text[:2000] if completion.raw_text else None
            raise AgentOutputError(
                "the model's response could not be parsed into the expected schema",
                raw_output_excerpt=raw_excerpt,
            )

        payload = completion.parsed
        try:
            return AgentFinding(
                case_id=task.case_id,
                finding=payload.finding,
                confidence=payload.confidence,
                reasoning_summary=payload.reasoning_summary,
                evidence=payload.evidence,
                missing_information=payload.missing_information,
                abstain=payload.abstain,
            )
        except ValidationError as exc:
            raise AgentOutputError(
                f"the model's response was schema-conformant JSON but failed AgentFinding "
                f"validation: {exc}",
                raw_output_excerpt=payload.model_dump_json()[:2000],
            ) from exc
