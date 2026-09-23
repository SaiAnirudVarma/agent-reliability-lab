"""A deterministic, rule-based baseline AgentRunner.

This is not an attempt to imitate an AI agent. Its purpose is to:

1. validate the AgentInput -> AgentFinding contract end to end,
2. establish a reproducible, zero-cost baseline,
3. surface reasoning or data-model problems before stochastic LLM behavior
   is introduced, and
4. give later LLM- and retrieval-based agents something concrete to compare
   against.

Every rule below reasons from ``Control`` fields and ``Evidence`` /
``Evidence.structured_fields`` values that are visible on ``AgentInput`` —
never from ``case_id``, and never from any ground-truth field. The same
handful of generic rule families is reused across every case that shares the
underlying reasoning pattern (e.g. one SLA-elapsed-time calculation handles
every timeliness case in the dataset, regardless of which control or which
case_id it appears in).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from app.agent.interface import AgentInput
from app.models.contracts import AgentFinding, Control, Evidence, EvidenceReference, Finding

AGENT_CONFIG_ID = "deterministic-baseline-v1"

# Confidence values below are heuristic labels chosen by the author to
# roughly rank "fully mechanical arithmetic/lookup" above "single-fact
# inference" above "abstention," and are NOT a statistically calibrated
# probability of correctness. Calibration (e.g. Expected Calibration Error)
# is only meaningful once we have stochastic model output to measure against
# — see the roadmap.
CONFIDENCE_DETERMINISTIC = 0.95
CONFIDENCE_SINGLE_SOURCE = 0.9
CONFIDENCE_ABSTAIN = 0.5
CONFIDENCE_NO_SIGNAL = 0.4

EvidenceFact = tuple[Evidence, str]


# ---------------------------------------------------------------------------
# Small internal result types (not part of any external contract)
# ---------------------------------------------------------------------------


@dataclass
class _TimelinessResult:
    status: str  # "within_sla" | "exceeded_sla" | "indeterminate" | "not_applicable"
    elapsed_hours: Optional[float]
    evidence_used: list[EvidenceFact] = field(default_factory=list)
    note: str = ""


@dataclass
class _DispositionResult:
    status: str  # "clean" | "flagged" | "not_applicable"
    evidence_used: list[EvidenceFact] = field(default_factory=list)


@dataclass
class _PolicyResult:
    finding: Finding
    evidence_used: list[EvidenceFact]
    reasoning: str


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _subject_key(evidence: Evidence) -> Optional[str]:
    """The identity an evidence record pertains to, if any (e.g. an employee
    or account), used to bind a trigger event to its matching completion
    event rather than to any completion event that happens to be in the pool.
    """

    return evidence.structured_fields.get("employee_id") or evidence.structured_fields.get(
        "account_id"
    )


def _split_by_period(period: str, evidence: list[Evidence]) -> tuple[list[Evidence], list[Evidence]]:
    """Partition evidence into what pertains to the case's own reporting
    period versus everything else. Only in-period evidence may be used to
    reach a conclusion about the current period — a clean, complete document
    from the wrong period is still the wrong period.
    """

    in_period = [e for e in evidence if e.period == period]
    out_of_period = [e for e in evidence if e.period != period]
    return in_period, out_of_period


# ---------------------------------------------------------------------------
# Evaluator: SLA / elapsed-time timeliness
#
# Applies to any control with sla_hours set, given evidence carrying a
# trigger_timestamp (the event that starts the clock) and a
# completion_timestamp (when the required action happened). The same
# function handles a quarterly review's "quarter end -> review completed"
# clock and a termination's "termination -> access removed" clock; nothing
# here is specific to either control.
# ---------------------------------------------------------------------------


def _evaluate_sla_timeliness(control: Control, in_period_evidence: list[Evidence]) -> _TimelinessResult:
    if control.sla_hours is None:
        return _TimelinessResult("not_applicable", None)

    trigger_candidates = [
        (e, e.structured_fields.get("trigger_timestamp"))
        for e in in_period_evidence
        if e.structured_fields.get("trigger_timestamp")
    ]
    completion_candidates = [
        (e, e.structured_fields.get("completion_timestamp"))
        for e in in_period_evidence
        if e.structured_fields.get("completion_timestamp")
    ]

    if not trigger_candidates:
        return _TimelinessResult("not_applicable", None)

    if not completion_candidates:
        return _TimelinessResult(
            "indeterminate",
            None,
            [(e, f"trigger_timestamp={ts}") for e, ts in trigger_candidates],
            note="a trigger event was found but no completion timestamp is available",
        )

    matched = [
        (trigger_evidence, trigger_ts, completion_evidence, completion_ts)
        for trigger_evidence, trigger_ts in trigger_candidates
        for completion_evidence, completion_ts in completion_candidates
        if _subject_key(trigger_evidence) is None
        or _subject_key(trigger_evidence) == _subject_key(completion_evidence)
    ]

    if not matched:
        return _TimelinessResult(
            "indeterminate",
            None,
            [(e, f"trigger_timestamp={ts}") for e, ts in trigger_candidates],
            note="no completion evidence could be matched to the same subject as the trigger event",
        )

    distinct_elapsed = {
        round((_parse_dt(ct) - _parse_dt(tt)).total_seconds() / 3600, 2) for _, tt, _, ct in matched
    }
    if len(distinct_elapsed) > 1:
        return _TimelinessResult(
            "indeterminate",
            None,
            [(te, f"trigger_timestamp={tt}") for te, tt, _, _ in matched],
            note="multiple, inconsistent trigger/completion pairs were found for this subject",
        )

    trigger_evidence, trigger_ts, completion_evidence, completion_ts = matched[0]
    elapsed_hours = (_parse_dt(completion_ts) - _parse_dt(trigger_ts)).total_seconds() / 3600

    if elapsed_hours < 0:
        return _TimelinessResult(
            "indeterminate",
            None,
            [(trigger_evidence, f"trigger_timestamp={trigger_ts}")],
            note="the completion timestamp precedes the trigger timestamp",
        )

    status = "within_sla" if elapsed_hours <= control.sla_hours else "exceeded_sla"
    return _TimelinessResult(
        status,
        elapsed_hours,
        [
            (trigger_evidence, f"trigger_timestamp={trigger_ts}"),
            (completion_evidence, f"completion_timestamp={completion_ts}"),
        ],
    )


# ---------------------------------------------------------------------------
# Evaluator: disposition / substantive-outcome signal
#
# Looks for whatever "was the reviewed thing found clean" signal the
# in-period evidence happens to carry: an explicit boolean, or a set of
# counts that must be derived (assessed/appropriate/flagged). The same
# function handles an access review's "all_confirmed_appropriate" flag, a
# termination's "access_fully_removed" flag, a firewall recert's
# "exceptions_found" count, and a case where scope/assessment counts are
# split across two separate evidence records that must be read together.
# ---------------------------------------------------------------------------


def _evaluate_disposition(in_period_evidence: list[Evidence]) -> _DispositionResult:
    for evidence in in_period_evidence:
        sf = evidence.structured_fields
        if sf.get("all_confirmed_appropriate") is not None:
            clean = bool(sf["all_confirmed_appropriate"])
            return _DispositionResult(
                "clean" if clean else "flagged",
                [(evidence, f"all_confirmed_appropriate={sf['all_confirmed_appropriate']}")],
            )
        if sf.get("access_fully_removed") is not None:
            clean = bool(sf["access_fully_removed"])
            return _DispositionResult(
                "clean" if clean else "flagged",
                [(evidence, f"access_fully_removed={sf['access_fully_removed']}")],
            )
        if sf.get("exceptions_found") is not None:
            clean = sf["exceptions_found"] == 0
            fact = f"exceptions_found={sf['exceptions_found']}"
            if sf.get("exception_rule_id"):
                fact += f", exception_rule_id={sf['exception_rule_id']}"
            return _DispositionResult("clean" if clean else "flagged", [(evidence, fact)])

    assessed = appropriate = flagged = None
    scope_evidence: list[EvidenceFact] = []
    for evidence in in_period_evidence:
        sf = evidence.structured_fields
        if sf.get("accounts_assessed") is not None:
            assessed = sf["accounts_assessed"]
            scope_evidence.append((evidence, f"accounts_assessed={assessed}"))
        if sf.get("accounts_appropriate") is not None:
            appropriate = sf["accounts_appropriate"]
            scope_evidence.append((evidence, f"accounts_appropriate={appropriate}"))
        if sf.get("accounts_flagged") is not None:
            flagged = sf["accounts_flagged"]
            scope_evidence.append((evidence, f"accounts_flagged={flagged}"))

    if assessed is not None and appropriate is not None and flagged is not None:
        clean = flagged == 0 and appropriate == assessed
        return _DispositionResult("clean" if clean else "flagged", scope_evidence)

    return _DispositionResult("not_applicable")


# ---------------------------------------------------------------------------
# Evaluator: observable tool/access unavailability
#
# A generic scan for evidence that documents its OWN retrieval/availability
# failure (an explicit error status, or an explicit "*_available: false"),
# as opposed to evidence that is simply silent on the topic.
# ---------------------------------------------------------------------------


def _detect_unavailability(in_period_evidence: list[Evidence]) -> list[EvidenceFact]:
    flagged: list[EvidenceFact] = []
    for evidence in in_period_evidence:
        sf = evidence.structured_fields
        if sf.get("status") == "error":
            detail = "status=error"
            if sf.get("http_status"):
                detail += f", http_status={sf['http_status']}"
            flagged.append((evidence, detail))
            continue
        for key, value in sf.items():
            if key.endswith("_available") and value is False:
                flagged.append((evidence, f"{key}=False"))
                break
    return flagged


# ---------------------------------------------------------------------------
# Evaluator: effective-policy resolution
#
# Generic to any case where (a) two or more evidence records describe policy
# versions with effective_start/effective_end windows and (b) exactly one
# other evidence record describes a compliance-relevant event with its own
# date. Resolves which policy's window actually contains the event date and
# evaluates compliance against THAT policy only — a superseded or
# not-yet-effective policy is never the one consulted, regardless of how
# prominently it appears in the evidence pool.
# ---------------------------------------------------------------------------


def _evaluate_effective_policy(all_evidence: list[Evidence]) -> Optional[_PolicyResult]:
    policy_candidates = [
        e
        for e in all_evidence
        if "policy_version" in e.structured_fields and "effective_start" in e.structured_fields
    ]
    if not policy_candidates:
        return None

    fact_candidates = [e for e in all_evidence if "provisioned_date" in e.structured_fields]
    if len(fact_candidates) != 1:
        return None
    fact = fact_candidates[0]
    event_date = date.fromisoformat(fact.structured_fields["provisioned_date"])

    governing = None
    for policy in policy_candidates:
        start = date.fromisoformat(policy.structured_fields["effective_start"])
        end_raw = policy.structured_fields.get("effective_end")
        end = date.fromisoformat(end_raw) if end_raw else None
        if start <= event_date and (end is None or event_date <= end):
            governing = policy
            break
    if governing is None:
        return None

    enabled_at_provisioning = fact.structured_fields.get("mfa_enabled_at_provisioning")
    if enabled_at_provisioning is None:
        return None

    version = governing.structured_fields.get("policy_version", "unknown")
    evidence_used: list[EvidenceFact] = [
        (
            governing,
            f"policy_version={version}, effective_start={governing.structured_fields['effective_start']}",
        ),
        (
            fact,
            f"provisioned_date={event_date.isoformat()}, "
            f"mfa_enabled_at_provisioning={enabled_at_provisioning}",
        ),
    ]

    if enabled_at_provisioning:
        return _PolicyResult(
            Finding.PASS,
            evidence_used,
            f"MFA was enabled at provisioning on {event_date.isoformat()}, satisfying policy "
            f"version {version}, which was in effect on that date.",
        )

    grace_days = governing.structured_fields.get("grace_period_days") or 0
    if grace_days <= 0:
        return _PolicyResult(
            Finding.EXCEPTION,
            evidence_used,
            f"Policy version {version}, in effect on the {event_date.isoformat()} provisioning "
            f"date, permits no grace period, but MFA was not enabled at provisioning.",
        )

    enabled_date_raw = fact.structured_fields.get("mfa_enabled_date")
    if not enabled_date_raw:
        return _PolicyResult(
            Finding.EXCEPTION,
            evidence_used,
            f"Policy version {version} allows a {grace_days}-day grace period, but no MFA "
            f"enablement date is recorded.",
        )

    enabled_date = date.fromisoformat(enabled_date_raw)
    delta_days = (enabled_date - event_date).days
    evidence_used[-1] = (fact, evidence_used[-1][1] + f", mfa_enabled_date={enabled_date_raw}")

    if delta_days <= grace_days:
        return _PolicyResult(
            Finding.PASS,
            evidence_used,
            f"MFA was enabled {delta_days} day(s) after provisioning, within the {grace_days}-day "
            f"grace period permitted by policy version {version}.",
        )
    return _PolicyResult(
        Finding.EXCEPTION,
        evidence_used,
        f"MFA was enabled {delta_days} day(s) after provisioning, exceeding the {grace_days}-day "
        f"grace period permitted by policy version {version}.",
    )


# ---------------------------------------------------------------------------
# Evidence reference construction
# ---------------------------------------------------------------------------


def _build_references(evidence_used: list[EvidenceFact]) -> list[EvidenceReference]:
    facts_by_document: dict[str, list[str]] = {}
    order: list[str] = []
    for evidence, fact in evidence_used:
        if evidence.document_id not in facts_by_document:
            facts_by_document[evidence.document_id] = []
            order.append(evidence.document_id)
        if fact not in facts_by_document[evidence.document_id]:
            facts_by_document[evidence.document_id].append(fact)
    return [
        EvidenceReference(document_id=doc_id, reference="; ".join(facts_by_document[doc_id]))
        for doc_id in order
    ]


# ---------------------------------------------------------------------------
# The agent itself
# ---------------------------------------------------------------------------


class DeterministicBaselineAgent:
    """A rule-based ``AgentRunner`` used as a reproducible comparison point.

    Reasons only over ``task.control`` and ``task.evidence`` (the fields
    ``AgentInput`` exposes). Contains no case-ID branching and no lookup
    table of expected outcomes — see the module docstring for the four
    generic rule families this composes.
    """

    agent_config_id = AGENT_CONFIG_ID

    def run(self, task: AgentInput) -> AgentFinding:
        in_period, _out_of_period = _split_by_period(task.period, task.evidence)

        policy_result = _evaluate_effective_policy(task.evidence)
        if policy_result is not None:
            return self._finding(
                task,
                policy_result.finding,
                policy_result.evidence_used,
                policy_result.reasoning,
                CONFIDENCE_SINGLE_SOURCE,
            )

        unavailable = _detect_unavailability(in_period)
        sla = _evaluate_sla_timeliness(task.control, in_period)
        disposition = _evaluate_disposition(in_period)

        if sla.status == "exceeded_sla":
            reasoning = (
                f"{sla.elapsed_hours:.1f} hours elapsed between the trigger event and completion, "
                f"exceeding the control's {task.control.sla_hours}-hour requirement."
            )
            return self._finding(
                task, Finding.EXCEPTION, sla.evidence_used, reasoning, CONFIDENCE_DETERMINISTIC
            )

        if sla.status == "indeterminate":
            reasoning = f"Timeliness could not be determined: {sla.note}."
            return self._finding(
                task,
                Finding.INSUFFICIENT_EVIDENCE,
                sla.evidence_used + unavailable,
                reasoning,
                CONFIDENCE_ABSTAIN,
                missing_information=[sla.note],
            )

        if sla.status == "within_sla":
            if disposition.status == "flagged":
                detail = "; ".join(fact for _, fact in disposition.evidence_used)
                reasoning = (
                    f"The action was completed within the {task.control.sla_hours}-hour "
                    f"requirement, but the recorded outcome shows an exception: {detail}."
                )
                return self._finding(
                    task,
                    Finding.EXCEPTION,
                    sla.evidence_used + disposition.evidence_used,
                    reasoning,
                    CONFIDENCE_DETERMINISTIC,
                )
            if disposition.status == "clean":
                reasoning = (
                    f"The action was completed within the {task.control.sla_hours}-hour "
                    f"requirement and the recorded outcome was clean."
                )
                return self._finding(
                    task,
                    Finding.PASS,
                    sla.evidence_used + disposition.evidence_used,
                    reasoning,
                    CONFIDENCE_DETERMINISTIC,
                )
            # Timely, but no evidence establishes the substantive outcome —
            # do not guess that "on time" implies "compliant content."
            reasoning = (
                "The action was completed within the required timeframe, but no evidence "
                "establishes the substantive outcome (disposition) of the completed review."
            )
            return self._finding(
                task,
                Finding.INSUFFICIENT_EVIDENCE,
                sla.evidence_used + unavailable,
                reasoning,
                CONFIDENCE_ABSTAIN,
                missing_information=["Evidence confirming the substantive outcome of the review"],
            )

        # sla.status == "not_applicable": this control/evidence combination
        # carries no trigger/completion timestamps at all — decide from
        # disposition alone if we have it.
        if disposition.status == "clean":
            detail = "; ".join(fact for _, fact in disposition.evidence_used)
            reasoning = f"The available current-period evidence recorded a clean outcome: {detail}."
            return self._finding(
                task, Finding.PASS, disposition.evidence_used, reasoning, CONFIDENCE_SINGLE_SOURCE
            )
        if disposition.status == "flagged":
            detail = "; ".join(fact for _, fact in disposition.evidence_used)
            reasoning = f"The available current-period evidence recorded an exception: {detail}."
            return self._finding(
                task, Finding.EXCEPTION, disposition.evidence_used, reasoning, CONFIDENCE_SINGLE_SOURCE
            )

        if unavailable:
            detail = "; ".join(f"{e.title} ({fact})" for e, fact in unavailable)
            reasoning = f"Required evidence for the current period could not be obtained: {detail}."
            return self._finding(
                task,
                Finding.INSUFFICIENT_EVIDENCE,
                unavailable,
                reasoning,
                CONFIDENCE_ABSTAIN,
                missing_information=[detail],
            )

        reasoning = (
            "No evidence relevant to this control's requirements is available for the current "
            "reporting period."
        )
        return self._finding(
            task,
            Finding.INSUFFICIENT_EVIDENCE,
            [],
            reasoning,
            CONFIDENCE_NO_SIGNAL,
            missing_information=["Evidence establishing this control's outcome for the current period"],
        )

    def _finding(
        self,
        task: AgentInput,
        finding: Finding,
        evidence_used: list[EvidenceFact],
        reasoning: str,
        confidence: float,
        missing_information: Optional[list[str]] = None,
    ) -> AgentFinding:
        return AgentFinding(
            case_id=task.case_id,
            finding=finding,
            confidence=confidence,
            reasoning_summary=reasoning,
            evidence=_build_references(evidence_used),
            missing_information=missing_information or [],
            abstain=finding == Finding.INSUFFICIENT_EVIDENCE,
        )
