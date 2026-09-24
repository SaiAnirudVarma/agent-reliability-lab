"""Unit tests for app.reranking.runner.run_reranking_evaluation -- the
before/after reranking evaluation orchestration. Uses FakeReranker and
PassThroughReranker only; no real provider anywhere in this file.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.models.contracts import Control
from app.reranking.pacing import CallPacer
from app.reranking.pass_through import PassThroughReranker
from app.reranking.runner import RerankingReport, run_reranking_evaluation
from app.retrieval.contracts import RetrievalQuery, RetrievalResult, RetrievedEvidence
from app.retrieval.runner import RetrievalReport
from tests.support.fake_clock import FakeClock
from tests.support.fake_reranker import FakeReranker


def _control() -> Control:
    return Control(control_id="CTRL-X", name="N", description="D", requirement_text="R")


def _query(case_id: str) -> RetrievalQuery:
    return RetrievalQuery(case_id=case_id, control=_control(), scenario_description="S", period="2025-Q1")


def _candidate(evidence_id: str, rank: int) -> RetrievedEvidence:
    return RetrievedEvidence(evidence_id=evidence_id, document_id=f"doc_{evidence_id}", score=1.0 / rank, rank=rank, retrieval_method="x")


def _result(case_id: str, evidence_ids: list[str]) -> RetrievalResult:
    candidates = [_candidate(eid, i) for i, eid in enumerate(evidence_ids, start=1)]
    return RetrievalResult(case_id=case_id, query=_query(case_id), candidates=candidates, top_k=len(candidates), retriever_config_id="x")


def _report(*results: RetrievalResult) -> RetrievalReport:
    return RetrievalReport(
        run_id="source-run-1", git_commit_sha="a" * 40, dataset_version="synthetic-v2",
        dataset_fingerprint="b" * 64, corpus_fingerprint="c" * 64, retriever_config_id="retrieval-baseline-v1",
        top_k_values=[1, 3], case_metrics=[], results=list(results),
        aggregate_recall_at_k={1: 1.0, 3: 1.0}, aggregate_mrr=1.0, total_retrieval_latency_ms=1.0,
        generated_at=datetime.now(timezone.utc),
    )


class TestRunRerankingEvaluationWithPassThrough:
    def test_pass_through_produces_zero_delta(self):
        report = _report(_result("AC-1", ["MISS", "REQUIRED", "OTHER"]))
        required = {"AC-1": ["REQUIRED"]}
        result = run_reranking_evaluation(
            report, PassThroughReranker(), candidate_depth=3, output_depth=3, k_values=(1, 3),
            required_evidence_ids_by_case=required, source_retrieval_artifact_sha256="e" * 64,
            reranker_config_id="pass-through-v1",
        )
        case = result.case_reports[0]
        for k in (1, 3):
            assert case.recall_at_k_delta[k] == 0.0
        assert case.mrr_delta == 0.0

    def test_report_provenance_fields(self):
        report = _report(_result("AC-1", ["A"]))
        result = run_reranking_evaluation(
            report, PassThroughReranker(), candidate_depth=1, output_depth=1, k_values=(1,),
            required_evidence_ids_by_case={}, source_retrieval_artifact_sha256="e" * 64,
            reranker_config_id="pass-through-v1",
        )
        assert result.dataset_version == "synthetic-v2"
        assert result.dataset_fingerprint == "b" * 64
        assert result.corpus_fingerprint == "c" * 64
        assert result.source_retrieval_run_id == "source-run-1"
        assert result.source_retrieval_artifact_sha256 == "e" * 64
        assert result.source_retriever_config_id == "retrieval-baseline-v1"
        assert result.reranker_provider is None  # not supplied -- never fabricated
        assert result.requested_reranker_model is None
        assert result.served_reranker_model is None


class TestRunRerankingEvaluationWithFakeReranker:
    def test_fake_reranker_moves_required_item_up_and_recall_improves(self):
        report = _report(_result("AC-1", ["A", "B", "REQUIRED"]))  # REQUIRED at rank 3
        required = {"AC-1": ["REQUIRED"]}
        reranker = FakeReranker({"A": 1.0, "B": 2.0, "REQUIRED": 5.0})  # boost REQUIRED to rank 1

        result = run_reranking_evaluation(
            report, reranker, candidate_depth=3, output_depth=3, k_values=(1, 3),
            required_evidence_ids_by_case=required, source_retrieval_artifact_sha256="e" * 64,
            reranker_config_id="fake-reranker-v1",
        )
        case = result.case_reports[0]
        assert case.recall_at_k_before[1] == 0.0
        assert case.recall_at_k_after[1] == 1.0
        assert case.recall_at_k_delta[1] == 1.0
        assert case.mrr_before == pytest.approx(1 / 3)
        assert case.mrr_after == 1.0

    def test_required_evidence_rank_movement_recorded_per_case(self):
        report = _report(_result("AC-1", ["A", "REQUIRED"]))
        required = {"AC-1": ["REQUIRED"]}
        reranker = FakeReranker({"A": 1.0, "REQUIRED": 5.0})
        result = run_reranking_evaluation(
            report, reranker, candidate_depth=2, output_depth=2, k_values=(1,),
            required_evidence_ids_by_case=required, source_retrieval_artifact_sha256="e" * 64,
            reranker_config_id="fake-reranker-v1",
        )
        movement = result.case_reports[0].required_evidence_rank_movement[0]
        assert movement.evidence_id == "REQUIRED"
        assert movement.in_candidate_set is True
        assert movement.original_rank == 2
        assert movement.reranked_rank == 1
        assert movement.rank_delta == 1

    def test_case_with_no_required_evidence_has_none_metrics(self):
        report = _report(_result("AC-1", ["A", "B"]))
        reranker = FakeReranker({"A": 1.0, "B": 2.0})
        result = run_reranking_evaluation(
            report, reranker, candidate_depth=2, output_depth=2, k_values=(1,),
            required_evidence_ids_by_case={"AC-1": []}, source_retrieval_artifact_sha256="e" * 64,
            reranker_config_id="fake-reranker-v1",
        )
        case = result.case_reports[0]
        assert case.recall_at_k_before[1] is None
        assert case.recall_at_k_after[1] is None
        assert case.recall_at_k_delta[1] is None
        assert case.mrr_before is None
        assert case.mrr_after is None
        assert case.required_evidence_rank_movement == []

    def test_provenance_threaded_through_from_caller(self):
        report = _report(_result("AC-1", ["A"]))
        reranker = FakeReranker({"A": 1.0})
        result = run_reranking_evaluation(
            report, reranker, candidate_depth=1, output_depth=1, k_values=(1,),
            required_evidence_ids_by_case={}, source_retrieval_artifact_sha256="e" * 64,
            reranker_config_id="cohere-rerank-v4-pro-baseline-v1", reranker_provider="cohere",
            requested_reranker_model="rerank-v4.0-pro", served_reranker_model=None,
            git_commit_sha="deadbeef", run_id="rerank-run-1",
        )
        assert result.reranker_provider == "cohere"
        assert result.requested_reranker_model == "rerank-v4.0-pro"
        assert result.served_reranker_model is None
        assert result.git_commit_sha == "deadbeef"
        assert result.run_id == "rerank-run-1"

    def test_run_id_generated_when_not_supplied(self):
        report = _report(_result("AC-1", ["A"]))
        result = run_reranking_evaluation(
            report, PassThroughReranker(), candidate_depth=1, output_depth=1, k_values=(1,),
            required_evidence_ids_by_case={}, source_retrieval_artifact_sha256="e" * 64,
            reranker_config_id="pass-through-v1",
        )
        assert result.run_id  # a real generated value

    def test_all_cases_scored(self):
        report = _report(_result("AC-1", ["A"]), _result("AC-2", ["B"]))
        result = run_reranking_evaluation(
            report, PassThroughReranker(), candidate_depth=1, output_depth=1, k_values=(1,),
            required_evidence_ids_by_case={}, source_retrieval_artifact_sha256="e" * 64,
            reranker_config_id="pass-through-v1",
        )
        assert {c.case_id for c in result.case_reports} == {"AC-1", "AC-2"}
        assert len(result.rerank_results) == 2

    def test_reranker_never_receives_required_evidence_ids(self):
        """The reranker call itself only ever sees a RerankInput -- which
        structurally cannot carry required_evidence_ids -- even though
        the runner has that ground truth for scoring purposes."""
        report = _report(_result("AC-1", ["A", "REQUIRED"]))
        required = {"AC-1": ["REQUIRED"]}

        captured_inputs = []

        class _RecordingReranker:
            reranker_config_id = "recording-v1"

            def rerank(self, rerank_input, *, final_k=None):
                captured_inputs.append(rerank_input)
                return PassThroughReranker().rerank(rerank_input, final_k=final_k)

        run_reranking_evaluation(
            report, _RecordingReranker(), candidate_depth=2, output_depth=2, k_values=(1,),
            required_evidence_ids_by_case=required, source_retrieval_artifact_sha256="e" * 64,
            reranker_config_id="recording-v1",
        )
        assert len(captured_inputs) == 1
        dumped = captured_inputs[0].model_dump_json()
        assert "required_evidence_ids" not in dumped
        assert "REQUIRED" in dumped  # the evidence_id itself is legitimately visible as a candidate


class _CountingReranker:
    """Wraps PassThroughReranker, counting calls and optionally raising on
    a specific case_id -- used to prove pacing/progress/failure-stop
    semantics without any real provider."""

    reranker_config_id = "counting-v1"

    def __init__(self, fail_on_case_id: str | None = None):
        self.fail_on_case_id = fail_on_case_id
        self.call_count = 0
        self.called_case_ids: list[str] = []

    def rerank(self, rerank_input, *, final_k=None):
        self.call_count += 1
        self.called_case_ids.append(rerank_input.case_id)
        if self.fail_on_case_id is not None and rerank_input.case_id == self.fail_on_case_id:
            raise RuntimeError(f"simulated provider failure on {rerank_input.case_id}")
        return PassThroughReranker().rerank(rerank_input, final_k=final_k)


class TestPacingIntegration:
    def test_pacer_wait_called_once_per_case_before_the_provider_call(self):
        report = _report(_result("AC-1", ["A"]), _result("AC-2", ["B"]), _result("AC-3", ["C"]))
        clock = FakeClock(start=0.0)
        pacer = CallPacer(7.0, monotonic=clock.monotonic, sleep=clock.sleep)
        reranker = _CountingReranker()

        run_reranking_evaluation(
            report, reranker, candidate_depth=1, output_depth=1, k_values=(1,),
            required_evidence_ids_by_case={}, source_retrieval_artifact_sha256="e" * 64,
            reranker_config_id="counting-v1", pacer=pacer,
        )

        assert reranker.call_count == 3
        # First call: no sleep. Second and third: sleep to maintain the
        # 7-second minimum interval between call STARTS.
        assert clock.sleep_calls == [7.0, 7.0]
        assert clock.now_value == 14.0

    def test_no_pacer_means_no_sleep_at_all(self):
        report = _report(_result("AC-1", ["A"]), _result("AC-2", ["B"]))
        reranker = _CountingReranker()
        run_reranking_evaluation(
            report, reranker, candidate_depth=1, output_depth=1, k_values=(1,),
            required_evidence_ids_by_case={}, source_retrieval_artifact_sha256="e" * 64,
            reranker_config_id="counting-v1", pacer=None,
        )
        assert reranker.call_count == 2  # completed instantly, no pacing applied


class TestProgressCallbacksAndFailureStopsFutureCases:
    def test_progress_callbacks_fire_once_per_case_in_order(self):
        report = _report(_result("AC-1", ["A"]), _result("AC-2", ["B"]))
        starts = []
        completes = []
        run_reranking_evaluation(
            report, PassThroughReranker(), candidate_depth=1, output_depth=1, k_values=(1,),
            required_evidence_ids_by_case={}, source_retrieval_artifact_sha256="e" * 64,
            reranker_config_id="pass-through-v1",
            on_case_start=lambda i, n, cid: starts.append((i, n, cid)),
            on_case_complete=lambda i, n, cid: completes.append((i, n, cid)),
        )
        assert starts == [(1, 2, "AC-1"), (2, 2, "AC-2")]
        assert completes == [(1, 2, "AC-1"), (2, 2, "AC-2")]

    def test_exactly_one_provider_invocation_per_case_on_success(self):
        report = _report(_result("AC-1", ["A"]), _result("AC-2", ["B"]), _result("AC-3", ["C"]))
        reranker = _CountingReranker()
        run_reranking_evaluation(
            report, reranker, candidate_depth=1, output_depth=1, k_values=(1,),
            required_evidence_ids_by_case={}, source_retrieval_artifact_sha256="e" * 64,
            reranker_config_id="counting-v1",
        )
        assert reranker.call_count == 3
        assert reranker.called_case_ids == ["AC-1", "AC-2", "AC-3"]

    def test_failure_stops_future_cases_and_does_not_retry(self):
        report = _report(_result("AC-1", ["A"]), _result("AC-2", ["B"]), _result("AC-3", ["C"]))
        reranker = _CountingReranker(fail_on_case_id="AC-2")
        starts = []
        completes = []

        with pytest.raises(RuntimeError, match="AC-2"):
            run_reranking_evaluation(
                report, reranker, candidate_depth=1, output_depth=1, k_values=(1,),
                required_evidence_ids_by_case={}, source_retrieval_artifact_sha256="e" * 64,
                reranker_config_id="counting-v1",
                on_case_start=lambda i, n, cid: starts.append(cid),
                on_case_complete=lambda i, n, cid: completes.append(cid),
            )

        # Called AC-1 (succeeded) then AC-2 (failed) -- NEVER AC-3, and
        # AC-2 was never retried (exactly one call for it).
        assert reranker.called_case_ids == ["AC-1", "AC-2"]
        assert reranker.call_count == 2
        assert starts == ["AC-1", "AC-2"]
        assert completes == ["AC-1"]  # AC-2 started but never completed


class TestZeroSemanticDriftFromPacingAndProgressInstrumentation:
    """Proves pacing/progress instrumentation changes nothing about WHAT
    is sent/scored/recorded -- only WHEN calls happen."""

    def _run(self, report, reranker, **kwargs):
        return run_reranking_evaluation(
            report, reranker, candidate_depth=2, output_depth=2, k_values=(1, 2),
            required_evidence_ids_by_case={"AC-1": ["B"]}, source_retrieval_artifact_sha256="e" * 64,
            reranker_config_id="counting-v1",
            **kwargs,
        )

    def test_rerank_results_identical_with_and_without_pacer(self):
        report = _report(_result("AC-1", ["A", "B"]))

        result_without_pacer = self._run(report, _CountingReranker())

        clock = FakeClock()
        pacer = CallPacer(7.0, monotonic=clock.monotonic, sleep=clock.sleep)
        result_with_pacer = self._run(report, _CountingReranker(), pacer=pacer)

        # Same candidates, same order, same scores/ranks/provenance --
        # compare everything except the timing fields, which legitimately
        # differ (this test process's own wall-clock timing, unrelated to
        # the FAKE pacing clock above).
        without_dump = result_without_pacer.model_dump(exclude={"generated_at", "run_id"})
        with_dump = result_with_pacer.model_dump(exclude={"generated_at", "run_id"})
        for report_dict in (without_dump, with_dump):
            for case_report in report_dict["case_reports"]:
                case_report.pop("reranking_latency_ms")
            report_dict.pop("total_reranking_latency_ms")
        assert without_dump == with_dump

    def test_candidate_depth_output_depth_k_values_unaffected_by_pacer(self):
        report = _report(_result("AC-1", ["A", "B"]))
        clock = FakeClock()
        pacer = CallPacer(7.0, monotonic=clock.monotonic, sleep=clock.sleep)
        result = self._run(report, _CountingReranker(), pacer=pacer)
        assert result.candidate_depth == 2
        assert result.output_depth == 2
        assert result.evaluation_k_values == [1, 2]

    def test_source_and_dataset_provenance_unaffected_by_pacer(self):
        report = _report(_result("AC-1", ["A", "B"]))
        clock = FakeClock()
        pacer = CallPacer(7.0, monotonic=clock.monotonic, sleep=clock.sleep)
        result = self._run(report, _CountingReranker(), pacer=pacer)
        assert result.source_retrieval_run_id == "source-run-1"
        assert result.source_retrieval_artifact_sha256 == "e" * 64
        assert result.dataset_fingerprint == "b" * 64
        assert result.corpus_fingerprint == "c" * 64

    def test_candidate_ids_and_order_sent_to_reranker_unaffected_by_pacer(self):
        report = _report(_result("AC-1", ["A", "B"]))
        captured = []

        class _RecordingReranker:
            reranker_config_id = "recording-v1"

            def rerank(self, rerank_input, *, final_k=None):
                captured.append([c.evidence_id for c in rerank_input.candidates])
                return PassThroughReranker().rerank(rerank_input, final_k=final_k)

        clock = FakeClock()
        pacer = CallPacer(7.0, monotonic=clock.monotonic, sleep=clock.sleep)
        self._run(report, _RecordingReranker(), pacer=pacer)
        assert captured == [["A", "B"]]

    def test_execution_policy_id_recorded_but_does_not_affect_scoring(self):
        report = _report(_result("AC-1", ["A", "B"]))
        result_a = self._run(report, _CountingReranker(), execution_policy_id="cohere-trial-pacing-v1")
        result_b = self._run(report, _CountingReranker(), execution_policy_id=None)
        assert result_a.execution_policy_id == "cohere-trial-pacing-v1"
        assert result_b.execution_policy_id is None
        assert result_a.aggregate_recall_at_k_after == result_b.aggregate_recall_at_k_after
        assert result_a.aggregate_mrr_after == result_b.aggregate_mrr_after
