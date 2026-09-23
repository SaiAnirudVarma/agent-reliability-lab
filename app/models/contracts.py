"""Core Pydantic data contracts for the Agent Reliability Lab.

These models are the single source of truth for every shape that crosses a
boundary in the system: synthetic dataset records, the agent's structured
output, execution provenance, and evaluation results. Keeping them strict
(``extra="forbid"``) means a malformed dataset file or a malformed agent
response fails loudly at the boundary instead of silently propagating.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Finding(str, Enum):
    """The three possible audit conclusions an agent may reach."""

    PASS = "PASS"
    EXCEPTION = "EXCEPTION"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class AgentMode(str, Enum):
    """Execution mode, selected explicitly via the AGENT_MODE env var.

    Never inferred from which API keys happen to be present.
    """

    MOCK = "mock"
    LLM = "llm"


class ModelProvider(str, Enum):
    """LLM providers supported by the provider abstraction (Phase 2+)."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class LLMConfig(BaseModel):
    """The generation configuration used to produce an LLM agent's output.

    Present if and only if ``agent_mode == LLM`` — see ``_check_provenance``.
    Together with ``provider``/``model_name`` (which stay at the top level of
    ``ExecutionTrace``/``RunReport``, since they describe the hosted model
    being called rather than a generation parameter specific to how it was
    called), this makes an LLM result's exact configuration recoverable from
    the serialized file alone, months later, without reading source code.

    Deliberately a single all-or-nothing bundle rather than three separate
    top-level ``Optional`` fields: that makes "this run had generation
    config at all" one atomic fact instead of three fields that could in
    principle drift out of sync (e.g. a ``temperature`` set while
    ``prompt_version`` was left ``None``).

    Contains only generation parameters — never API keys, environment
    variables, or other provider configuration.
    """

    model_config = ConfigDict(extra="forbid")

    prompt_version: str = Field(min_length=1)
    temperature: float
    max_output_tokens: int = Field(gt=0)


def _check_provenance(
    mode: AgentMode,
    provider: Optional[ModelProvider],
    model_name: Optional[str],
    llm_config: Optional[LLMConfig],
    served_model_name: Optional[str] = None,
) -> None:
    """Shared provenance rule for anything that records agent_mode + config.

    LLM mode must carry a concrete provider/model/llm_config; mock mode must
    not carry any of them, since it never called a hosted model and
    fabricating placeholder values (e.g. temperature=0) would misrepresent
    that as a real configuration choice.

    ``served_model_name`` (what a provider's response actually reported,
    vs. ``model_name``, what was requested) is NOT required even in LLM
    mode -- a provider may legitimately be unable to confirm it (e.g. on an
    error/refusal path) -- but mock mode must never carry one, same as
    every other LLM-specific field.
    """

    if mode == AgentMode.LLM:
        if provider is None or model_name is None:
            raise ValueError(
                "agent_mode=llm requires both provider and model_name to be set"
            )
        if llm_config is None:
            raise ValueError("agent_mode=llm requires llm_config to be set")
    else:
        if provider is not None or model_name is not None:
            raise ValueError("agent_mode=mock must not set provider or model_name")
        if llm_config is not None:
            raise ValueError("agent_mode=mock must not set llm_config")
        if served_model_name is not None:
            raise ValueError("agent_mode=mock must not set served_model_name")


class Control(BaseModel):
    """A reusable control requirement (e.g. an access-review policy).

    Controls are independent of any specific case: the same control can be
    exercised by many EvaluationCases against different evidence.
    """

    model_config = ConfigDict(extra="forbid")

    control_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    requirement_text: str = Field(min_length=1)
    sla_hours: Optional[int] = Field(default=None, gt=0)


class Evidence(BaseModel):
    """A single piece of documentary evidence available to the agent.

    ``content`` is the human-readable text an LLM agent reads. ``structured_fields``
    holds machine-readable facts (e.g. timestamps) so the deterministic
    baseline agent can reason generically without parsing prose.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    period: str = Field(pattern=r"^\d{4}-Q[1-4]$")
    content: str = Field(min_length=1)
    structured_fields: dict[str, Any] = Field(default_factory=dict)


class EvidenceReference(BaseModel):
    """A pointer from an agent finding back to a specific piece of evidence."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    reference: str = Field(min_length=1)


class ExpectedOutcome(BaseModel):
    """Ground truth for an EvaluationCase, used only by the evaluator.

    Never shown to the agent under test.
    """

    model_config = ConfigDict(extra="forbid")

    expected_finding: Finding
    required_evidence_ids: list[str] = Field(default_factory=list)
    should_abstain: bool
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _abstain_matches_finding(self) -> "ExpectedOutcome":
        expected_abstain = self.expected_finding == Finding.INSUFFICIENT_EVIDENCE
        if self.should_abstain != expected_abstain:
            raise ValueError(
                "should_abstain must be True iff expected_finding is INSUFFICIENT_EVIDENCE"
            )
        return self


class EvaluationCase(BaseModel):
    """A synthetic control-testing scenario with a fixed evidence pool."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(pattern=r"^AC-\d{3}$")
    control_id: str = Field(min_length=1)
    scenario_description: str = Field(min_length=1)
    evidence_pool: list[str] = Field(min_length=1)
    period: str = Field(pattern=r"^\d{4}-Q[1-4]$")
    expected_outcome: ExpectedOutcome
    failure_mode_tag: str = Field(min_length=1)

    @field_validator("evidence_pool")
    @classmethod
    def _unique_evidence_ids(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("evidence_pool must not contain duplicate evidence IDs")
        return value

    @model_validator(mode="after")
    def _required_evidence_subset_of_pool(self) -> "EvaluationCase":
        missing = set(self.expected_outcome.required_evidence_ids) - set(
            self.evidence_pool
        )
        if missing:
            raise ValueError(
                "required_evidence_ids references evidence not in evidence_pool: "
                f"{sorted(missing)}"
            )
        return self


class AgentFinding(BaseModel):
    """The audit agent's structured output contract.

    ``reasoning_summary`` must be a concise, user-facing explanation only —
    never a hidden chain-of-thought dump. The length cap enforces conciseness.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    finding: Finding
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str = Field(min_length=1, max_length=600)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    abstain: bool

    @model_validator(mode="after")
    def _abstain_matches_finding(self) -> "AgentFinding":
        expected_abstain = self.finding == Finding.INSUFFICIENT_EVIDENCE
        if self.abstain != expected_abstain:
            raise ValueError("abstain must be True iff finding is INSUFFICIENT_EVIDENCE")
        return self


class ExecutionTrace(BaseModel):
    """Execution provenance for one agent run against one case.

    Provenance is explicit and never inferred: ``agent_mode``, ``provider``
    and ``model_name`` are recorded verbatim from the configuration that
    produced this run, so results stay reproducible and auditable after the
    fact.

    ``output`` is ``None`` exactly when the agent raised ``AgentOutputError``
    (app.agent.interface) instead of returning a finding -- i.e. it received
    a response but could not turn it into a valid ``AgentFinding``. This is a
    model/agent failure, not an infrastructure failure: the trace still
    exists, ``errors`` carries a human-readable description, and
    ``raw_output_excerpt`` optionally carries a bounded, secret-free
    excerpt of what was actually returned, to make the failure debuggable.
    It is only ever populated on a parse failure -- a successful run never
    duplicates the same content into both ``output`` and this field.

    ``llm_config`` carries the generation configuration (prompt version,
    temperature, max output tokens) and is present if and only if
    ``agent_mode == LLM`` -- see ``LLMConfig``.

    ``model_name`` is the REQUESTED model identity -- what the run was
    configured to call. ``served_model_name`` is what the provider's
    response reported as the model that actually served the request (e.g.
    a dated snapshot behind a rolling alias), when a provider can confirm
    one. It is best-effort even in LLM mode (a provider may not always be
    able to report it, e.g. on an error path) and always ``None`` in mock
    mode, same as every other LLM-specific field.
    """

    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    agent_mode: AgentMode
    provider: Optional[ModelProvider] = None
    model_name: Optional[str] = None
    served_model_name: Optional[str] = None
    llm_config: Optional[LLMConfig] = None
    agent_config_id: str = Field(min_length=1)
    evidence_presented: list[str] = Field(default_factory=list)
    evidence_referenced: list[str] = Field(default_factory=list)
    started_at: datetime
    ended_at: datetime
    latency_ms: float = Field(ge=0.0)
    output: Optional[AgentFinding] = None
    errors: list[str] = Field(default_factory=list)
    raw_output_excerpt: Optional[str] = Field(default=None, max_length=2000)
    input_tokens: Optional[int] = Field(default=None, ge=0)
    output_tokens: Optional[int] = Field(default=None, ge=0)
    estimated_cost: Optional[float] = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _timestamps_ordered(self) -> "ExecutionTrace":
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must not be before started_at")
        return self

    @model_validator(mode="after")
    def _provenance_matches_mode(self) -> "ExecutionTrace":
        _check_provenance(
            self.agent_mode, self.provider, self.model_name, self.llm_config, self.served_model_name
        )
        return self


class EvaluationResult(BaseModel):
    """Deterministic scoring of one AgentFinding against its ExpectedOutcome.

    ``citation_validity`` measures only whether cited evidence *exists* and
    was presented to the agent — it is NOT a semantic-support check. That is
    a distinct, deferred metric named ``citation_support`` (see roadmap); do
    not conflate the two when reading or extending this model.

    ``required_evidence_recall`` answers a different question from
    ``citation_validity``: "did the agent cite the evidence that was
    genuinely necessary for the ground-truth decision?" (vs. "were the
    agent's citations real?"). It is ``None`` when
    ``ExpectedOutcome.required_evidence_ids`` is empty for this case — i.e.
    genuinely not applicable, not a measured zero — OR when this
    ``EvaluationResult`` predates the metric's introduction (an old artifact
    loaded for backward compatibility). These two ``None`` causes are not
    distinguishable from this field alone; see
    ``app.evaluation.evaluator.compute_required_evidence_recall``.

    ``actual`` is ``None`` when the agent failed to produce a schema-valid
    finding at all (``schema_valid=False``). In that case ``correct_finding``
    is ``False`` (there is nothing to have gotten right), ``citation_validity``
    is ``0.0`` (a harder grounding failure than a real finding citing
    nothing), ``required_evidence_recall`` is ``0.0`` if the case had any
    required evidence (nothing was cited) or ``None`` if it had none, and
    ``abstention_correct`` is ``None`` -- genuinely undefined, since there is
    no ``abstain`` flag to compare against ground truth.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    expected: ExpectedOutcome
    actual: Optional[AgentFinding] = None
    correct_finding: bool
    schema_valid: bool
    citation_validity: float = Field(ge=0.0, le=1.0)
    required_evidence_recall: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    abstention_correct: Optional[bool] = None


class MetricSummary(BaseModel):
    """Aggregate metrics computed from actual EvaluationResults for one run.

    ``required_evidence_recall`` is ``Optional`` ONLY for backward
    compatibility with artifacts saved before this metric existed (e.g. the
    official synthetic-v1 LLM baseline) -- ``None`` there means exactly
    "not computed for this run," never "computed as zero." Every run
    produced after this metric's introduction always populates it with a
    real float (using the documented empty-denominator convention when zero
    cases are applicable), so for any current run ``None`` should not occur.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    total_cases: int = Field(ge=0)
    finding_accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    exception_precision: float = Field(ge=0.0, le=1.0)
    exception_recall: float = Field(ge=0.0, le=1.0)
    citation_validity: float = Field(ge=0.0, le=1.0)
    required_evidence_recall: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    schema_validity: float = Field(ge=0.0, le=1.0)
    abstention_precision: float = Field(ge=0.0, le=1.0)
    abstention_recall: float = Field(ge=0.0, le=1.0)
    abstention_f1: float = Field(ge=0.0, le=1.0)
    generated_at: datetime


class RunReport(BaseModel):
    """The full artifact serialized to results/baseline.json.

    ``agent_mode``/``provider``/``model_name``/``llm_config`` are
    denormalized here (in addition to each ExecutionTrace) as a convenience
    header describing the single configuration used for the whole run, so a
    reader can determine which provider, model, agent configuration, prompt
    version, temperature, and max output tokens produced this file without
    inspecting individual traces or reading source code.

    ``dataset_version``/``dataset_fingerprint`` are ``Optional`` ONLY for
    backward compatibility with artifacts saved before Phase 6 introduced
    dataset versioning (e.g. the official synthetic-v1 LLM baseline) --
    ``None`` there means "this run predates dataset versioning," not
    "unknown version." Every run produced after Phase 6 always populates
    both. ``dataset_fingerprint`` is a SHA-256 over exactly the controls,
    evidence, and cases that make up ``dataset_version`` (see
    ``app.datasets.loader`` for exactly what enters it), so two runs can be
    confirmed to have used byte-identical dataset content even if the
    dataset directory has since changed.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    agent_mode: AgentMode
    provider: Optional[ModelProvider] = None
    model_name: Optional[str] = None
    llm_config: Optional[LLMConfig] = None
    dataset_version: Optional[str] = None
    dataset_fingerprint: Optional[str] = None
    results: list[EvaluationResult]
    traces: list[ExecutionTrace]
    metrics: MetricSummary
    generated_at: datetime

    @model_validator(mode="after")
    def _provenance_matches_mode(self) -> "RunReport":
        _check_provenance(self.agent_mode, self.provider, self.model_name, self.llm_config)
        return self
