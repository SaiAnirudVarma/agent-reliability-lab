"""Demonstrates the Phase 7B agent-integration boundary in real code (no
wiring, no LLM, no network): RetrievalResult -> selected Evidence[] ->
existing AgentInput -> existing AgentRunner, using the zero-cost
DeterministicBaselineAgent to prove the existing AgentRunner interface and
AgentInput model need no changes at all.
"""

from __future__ import annotations

from pathlib import Path

from app.agent.deterministic_baseline import DeterministicBaselineAgent
from app.agent.interface import AgentInput
from app.datasets.loader import load_dataset
from app.models.contracts import AgentFinding
from app.retrieval.contracts import build_retrieval_query
from app.retrieval.corpus import build_full_version_corpus
from app.retrieval.integration import select_evidence_for_agent
from app.retrieval.lexical_baseline import LexicalBaselineRetriever

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATASET_DIR = REPO_ROOT / "datasets"


class TestRetrievalResultFeedsExistingAgentInputUnmodified:
    def test_end_to_end_pipeline_with_no_agent_input_changes(self):
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus = build_full_version_corpus(dataset)
        retriever = LexicalBaselineRetriever()
        case = dataset.cases_by_id["AC-001"]
        control = dataset.controls[case.control_id]

        query = build_retrieval_query(case, control)
        retrieval_result = retriever.retrieve(query, corpus, top_k=5)

        selected_evidence = select_evidence_for_agent(retrieval_result, corpus)
        assert len(selected_evidence) == len(retrieval_result.candidates)

        # The existing AgentInput model, constructed directly -- no new
        # field, no schema change, no relaxed validator anywhere on
        # AgentInput itself.
        task = AgentInput(
            case_id=case.case_id, control=control, scenario_description=case.scenario_description,
            period=case.period, evidence=selected_evidence,
        )

        # The existing AgentRunner (DeterministicBaselineAgent, zero cost,
        # zero network) runs it exactly like any oracle-pool AgentInput.
        agent = DeterministicBaselineAgent()
        finding = agent.run(task)
        assert isinstance(finding, AgentFinding)
        assert finding.case_id == case.case_id

    def test_retrieval_selected_evidence_can_legitimately_differ_from_oracle_pool(self):
        """This is the entire point of retrieval: the selected set need not
        equal case.evidence_pool. build_agent_input's oracle-pool-equality
        check would reject this -- confirming AgentInput itself doesn't
        enforce that equality, only that one particular oracle-mode
        constructor does."""
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus = build_full_version_corpus(dataset)
        retriever = LexicalBaselineRetriever()
        case = dataset.cases_by_id["AC-001"]  # oracle pool has 2 evidence records
        control = dataset.controls[case.control_id]

        query = build_retrieval_query(case, control)
        # top_k larger than the oracle pool size -- retrieval legitimately
        # returns a different (larger) set than case.evidence_pool.
        retrieval_result = retriever.retrieve(query, corpus, top_k=10)
        selected_evidence = select_evidence_for_agent(retrieval_result, corpus)

        selected_ids = {e.evidence_id for e in selected_evidence}
        assert selected_ids != set(case.evidence_pool)

        # AgentInput itself accepts this without any change or relaxed check.
        task = AgentInput(
            case_id=case.case_id, control=control, scenario_description=case.scenario_description,
            period=case.period, evidence=selected_evidence,
        )
        assert {e.evidence_id for e in task.evidence} == selected_ids

    def test_build_agent_input_helper_is_unmodified_and_still_enforces_oracle_equality(self):
        """Confirms the EXISTING oracle-mode helper's behavior is untouched
        -- it must still reject a retrieval-selected set that differs from
        case.evidence_pool, which is exactly why Phase 7B needs its own,
        separate constructor rather than a change to this one (see
        app.retrieval.integration's module docstring)."""
        import pytest

        from app.agent.interface import build_agent_input

        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus = build_full_version_corpus(dataset)
        retriever = LexicalBaselineRetriever()
        case = dataset.cases_by_id["AC-001"]
        control = dataset.controls[case.control_id]
        query = build_retrieval_query(case, control)
        retrieval_result = retriever.retrieve(query, corpus, top_k=10)
        selected_evidence = select_evidence_for_agent(retrieval_result, corpus)

        with pytest.raises(ValueError, match="evidence mismatch"):
            build_agent_input(case, control, selected_evidence)
