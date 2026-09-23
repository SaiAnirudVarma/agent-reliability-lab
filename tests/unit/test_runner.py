"""Unit tests for app.evaluation.runner.run_evaluation."""

from pathlib import Path

from app.agent.deterministic_baseline import DeterministicBaselineAgent
from app.agent.interface import AgentInput
from app.datasets.loader import load_dataset
from app.evaluation.runner import run_evaluation
from app.models.contracts import AgentFinding, AgentMode, Finding

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATASET_DIR = REPO_ROOT / "datasets"


class TestRunEvaluation:
    def test_runs_all_ten_cases(self):
        dataset = load_dataset(REAL_DATASET_DIR)
        report = run_evaluation(dataset, DeterministicBaselineAgent(), agent_mode=AgentMode.MOCK)
        assert len(report.results) == 10
        assert len(report.traces) == 10

    def test_stable_case_order(self):
        dataset = load_dataset(REAL_DATASET_DIR)
        report = run_evaluation(dataset, DeterministicBaselineAgent(), agent_mode=AgentMode.MOCK)
        expected_order = [case.case_id for case in dataset.cases]
        assert [r.case_id for r in report.results] == expected_order
        assert [t.case_id for t in report.traces] == expected_order

    def test_single_run_id_shared_across_report_results_and_traces(self):
        dataset = load_dataset(REAL_DATASET_DIR)
        report = run_evaluation(dataset, DeterministicBaselineAgent(), agent_mode=AgentMode.MOCK)
        run_ids = {report.run_id} | {r.run_id for r in report.results} | {t.run_id for t in report.traces}
        assert run_ids == {report.run_id}
        assert report.metrics.run_id == report.run_id

    def test_two_runs_get_different_run_ids(self):
        dataset = load_dataset(REAL_DATASET_DIR)
        agent = DeterministicBaselineAgent()
        report_1 = run_evaluation(dataset, agent, agent_mode=AgentMode.MOCK)
        report_2 = run_evaluation(dataset, agent, agent_mode=AgentMode.MOCK)
        assert report_1.run_id != report_2.run_id

    def test_ground_truth_is_never_passed_to_the_agent(self, monkeypatch):
        """Instrument AgentRunner.run to prove it is only ever called with
        an AgentInput that carries no ground truth."""
        dataset = load_dataset(REAL_DATASET_DIR)
        agent = DeterministicBaselineAgent()
        seen_tasks: list[AgentInput] = []
        original_run = agent.run

        def spy_run(task: AgentInput) -> AgentFinding:
            seen_tasks.append(task)
            return original_run(task)

        monkeypatch.setattr(agent, "run", spy_run)
        run_evaluation(dataset, agent, agent_mode=AgentMode.MOCK)

        assert len(seen_tasks) == 10
        for task in seen_tasks:
            assert isinstance(task, AgentInput)
            dumped = task.model_dump_json()
            assert "expected_outcome" not in dumped
            assert "failure_mode_tag" not in dumped

    def test_a_wrong_prediction_does_not_abort_the_run(self):
        """A stub agent that always guesses PASS should still produce a
        complete RunReport for all 10 cases, not raise -- an incorrect
        prediction is an experiment result, not an infrastructure failure.
        """
        dataset = load_dataset(REAL_DATASET_DIR)

        class _AlwaysPassAgent:
            agent_config_id = "always-pass-v1"

            def run(self, task: AgentInput) -> AgentFinding:
                return AgentFinding(
                    case_id=task.case_id,
                    finding=Finding.PASS,
                    confidence=0.5,
                    reasoning_summary="Always guesses PASS.",
                    evidence=[],
                    missing_information=[],
                    abstain=False,
                )

        report = run_evaluation(dataset, _AlwaysPassAgent(), agent_mode=AgentMode.MOCK)
        assert len(report.results) == 10
        assert len(report.traces) == 10
        assert 0.0 < report.metrics.finding_accuracy < 1.0
