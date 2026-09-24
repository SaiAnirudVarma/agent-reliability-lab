"""Unit tests for app.integration.agent_input: hydrating a case's Top-K
reranked candidates into full Evidence and building a ground-truth-free
AgentInput from them.
"""

from __future__ import annotations

from app.integration.agent_input import build_agent_input_from_reranking, select_reranked_evidence_for_agent
from app.models.contracts import Control, Evidence, EvaluationCase, ExpectedOutcome, Finding
from app.reranking.contracts import RerankedEvidence
from app.retrieval.corpus import EvidenceCorpus


def _evidence(evidence_id: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id, document_id=f"doc_{evidence_id}", title=f"Title {evidence_id}",
        period="2025-Q1", content=f"Content for {evidence_id}",
    )


def _corpus(evidence_ids: list[str]) -> EvidenceCorpus:
    return EvidenceCorpus(version="synthetic-v2", evidence=[_evidence(eid) for eid in evidence_ids])


def _control() -> Control:
    return Control(control_id="CTRL-X", name="N", description="D", requirement_text="R")


def _case(case_id: str, required_evidence_ids: list[str], failure_mode_tag: str = "some-secret-failure-mode") -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id, control_id="CTRL-X", scenario_description="A scenario", evidence_pool=["EV-1"],
        period="2025-Q1",
        expected_outcome=ExpectedOutcome(
            expected_finding=Finding.PASS, required_evidence_ids=required_evidence_ids, should_abstain=False,
            rationale="a secret rationale that must never reach the agent",
        ),
        failure_mode_tag=failure_mode_tag,
    )


def _reranked(evidence_id: str, reranked_rank: int) -> RerankedEvidence:
    return RerankedEvidence(
        evidence_id=evidence_id, document_id=f"doc_{evidence_id}", original_rank=reranked_rank,
        reranked_rank=reranked_rank, original_score=1.0, reranker_score=1.0, reranker_method="pass-through",
    )


class TestSelectRerankedEvidenceForAgent:
    def test_returns_evidence_in_reranked_rank_order_not_input_order(self):
        corpus = _corpus(["EV-1", "EV-2", "EV-3"])
        # Deliberately supplied out of rank order -- the function must sort.
        candidates = [_reranked("EV-3", 1), _reranked("EV-1", 2), _reranked("EV-2", 3)]
        evidence = select_reranked_evidence_for_agent(candidates, corpus)
        assert [e.evidence_id for e in evidence] == ["EV-3", "EV-1", "EV-2"]

    def test_hydrates_exactly_the_given_candidates_no_more_no_fewer(self):
        corpus = _corpus(["EV-1", "EV-2", "EV-3", "EV-4", "EV-5"])
        candidates = [_reranked(f"EV-{i}", i) for i in range(1, 4)]  # only 1..3
        evidence = select_reranked_evidence_for_agent(candidates, corpus)
        assert len(evidence) == 3
        assert {e.evidence_id for e in evidence} == {"EV-1", "EV-2", "EV-3"}


class TestBuildAgentInputFromReranking:
    def test_exactly_top_k_reaches_the_agent_in_rank_order(self):
        corpus = _corpus([f"EV-{i}" for i in range(1, 11)])
        case = _case("AC-001", required_evidence_ids=["EV-1"])
        top_5 = [_reranked(f"EV-{i}", i) for i in range(1, 6)]  # already truncated to top 5

        task = build_agent_input_from_reranking(case, _control(), top_5, corpus)

        assert len(task.evidence) == 5
        assert [e.evidence_id for e in task.evidence] == ["EV-1", "EV-2", "EV-3", "EV-4", "EV-5"]

    def test_no_rank_beyond_the_supplied_candidates_reaches_the_agent(self):
        """The function trusts its caller to have already truncated to
        the intended Top-K (see app.integration.source.build_top_k_candidates_by_case)
        -- this test proves it never reaches past what it was given to
        pull in extra, higher-ranked-elsewhere evidence."""

        corpus = _corpus([f"EV-{i}" for i in range(1, 11)])
        case = _case("AC-001", required_evidence_ids=["EV-1"])
        only_three = [_reranked(f"EV-{i}", i) for i in range(1, 4)]

        task = build_agent_input_from_reranking(case, _control(), only_three, corpus)

        assert len(task.evidence) == 3
        assert "EV-4" not in [e.evidence_id for e in task.evidence]

    def test_no_ground_truth_leakage_into_agent_input(self):
        corpus = _corpus([f"EV-{i}" for i in range(1, 6)])
        case = _case(
            "AC-001", required_evidence_ids=["EV-1"],
            failure_mode_tag="THIS-MUST-NEVER-APPEAR-IN-AGENT-INPUT",
        )
        top_5 = [_reranked(f"EV-{i}", i) for i in range(1, 6)]

        task = build_agent_input_from_reranking(case, _control(), top_5, corpus)
        dumped = task.model_dump_json()

        assert "expected_outcome" not in dumped
        assert "failure_mode_tag" not in dumped
        assert "required_evidence_ids" not in dumped
        assert "THIS-MUST-NEVER-APPEAR-IN-AGENT-INPUT" not in dumped
        assert "a secret rationale that must never reach the agent" not in dumped

    def test_agent_input_schema_structurally_cannot_carry_ground_truth(self):
        """AgentInput itself (extra='forbid') has no ground-truth-shaped
        field at all -- this is a structural guarantee, not merely a
        convention this module happens to follow."""

        from app.agent.interface import AgentInput

        forbidden_substrings = ("expected_outcome", "required_evidence", "failure_mode")
        for field_name in AgentInput.model_fields:
            lowered = field_name.lower()
            assert not any(bad in lowered for bad in forbidden_substrings), field_name
