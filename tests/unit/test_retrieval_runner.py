"""Unit tests for app.retrieval.runner.run_retrieval_evaluation -- the
retrieval-only evaluation loop. No AgentRunner, no LLM, anywhere here.
"""

from __future__ import annotations

from pathlib import Path

from app.datasets.loader import load_dataset
from app.retrieval.corpus import build_full_version_corpus
from app.retrieval.corpus_index import build_corpus_embedding_index
from app.retrieval.embedding_provider import FakeEmbeddingProvider
from app.retrieval.lexical_baseline import LexicalBaselineRetriever
from app.retrieval.runner import DEFAULT_K_VALUES, RetrievalReport, run_retrieval_evaluation
from app.retrieval.serialization import evidence_embedding_text, retrieval_query_embedding_text
from app.retrieval.vector_retriever import VectorRetriever

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATASET_DIR = REPO_ROOT / "datasets"


class TestRunRetrievalEvaluationWithLexicalBaseline:
    """Exercises the runner end-to-end against the real, frozen
    synthetic-v2 dataset -- read-only, no dataset modification, no LLM."""

    def test_produces_a_result_and_case_metric_per_case(self):
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus = build_full_version_corpus(dataset)
        retriever = LexicalBaselineRetriever()

        report = run_retrieval_evaluation(dataset, retriever, corpus)

        assert isinstance(report, RetrievalReport)
        assert len(report.results) == 30
        assert len(report.case_metrics) == 30
        assert {r.case_id for r in report.results} == {c.case_id for c in report.case_metrics}

    def test_default_k_values_are_1_3_5_10(self):
        assert DEFAULT_K_VALUES == (1, 3, 5, 10)

    def test_case_metrics_have_all_default_k_values(self):
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus = build_full_version_corpus(dataset)
        report = run_retrieval_evaluation(dataset, LexicalBaselineRetriever(), corpus)
        for case_metric in report.case_metrics:
            assert set(case_metric.recall_at_k.keys()) == {1, 3, 5, 10}

    def test_aggregate_recall_at_k_has_all_default_k_values(self):
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus = build_full_version_corpus(dataset)
        report = run_retrieval_evaluation(dataset, LexicalBaselineRetriever(), corpus)
        assert set(report.aggregate_recall_at_k.keys()) == {1, 3, 5, 10}
        for value in report.aggregate_recall_at_k.values():
            assert 0.0 <= value <= 1.0

    def test_aggregate_mrr_in_valid_range(self):
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus = build_full_version_corpus(dataset)
        report = run_retrieval_evaluation(dataset, LexicalBaselineRetriever(), corpus)
        assert 0.0 <= report.aggregate_mrr <= 1.0

    def test_only_one_retrieve_call_per_case_regardless_of_k_count(self):
        """Recall@1/3/5/10 must come from ONE retrieve(top_k=10) call per
        case, not four."""
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus = build_full_version_corpus(dataset)

        call_count = {"n": 0}
        real_retriever = LexicalBaselineRetriever()

        class _CountingRetriever:
            retriever_config_id = "counting-wrapper"

            def retrieve(self, query, corpus, *, top_k):
                call_count["n"] += 1
                return real_retriever.retrieve(query, corpus, top_k=top_k)

        run_retrieval_evaluation(dataset, _CountingRetriever(), corpus)
        assert call_count["n"] == 30  # exactly one call per case, not 30*4

    def test_report_provenance_fields(self):
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus = build_full_version_corpus(dataset)
        report = run_retrieval_evaluation(dataset, LexicalBaselineRetriever(), corpus)
        assert report.dataset_version == "synthetic-v2"
        assert report.dataset_fingerprint == dataset.fingerprint
        assert report.corpus_fingerprint == corpus.fingerprint
        assert report.retriever_config_id == "lexical-baseline-v1"
        # lexical baseline is not embedding-backed -- none of these are
        # applicable, and none are fabricated as a result.
        assert report.embedding_provider is None
        assert report.requested_embedding_model is None
        assert report.served_embedding_model is None
        assert report.embedding_dimension is None
        assert report.corpus_embedding_latency_ms is None
        assert report.top_k_values == [1, 3, 5, 10]

    def test_git_commit_sha_is_none_when_not_supplied_and_threaded_when_supplied(self):
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus = build_full_version_corpus(dataset)
        report_without = run_retrieval_evaluation(dataset, LexicalBaselineRetriever(), corpus)
        assert report_without.git_commit_sha is None

        report_with = run_retrieval_evaluation(
            dataset, LexicalBaselineRetriever(), corpus, git_commit_sha="fake-sha-for-test",
        )
        assert report_with.git_commit_sha == "fake-sha-for-test"

    def test_run_id_generated_when_not_supplied_and_threaded_when_supplied(self):
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus = build_full_version_corpus(dataset)
        report_without = run_retrieval_evaluation(dataset, LexicalBaselineRetriever(), corpus)
        assert report_without.run_id  # a real, non-empty generated UUID string

        report_with = run_retrieval_evaluation(
            dataset, LexicalBaselineRetriever(), corpus, run_id="fixed-run-id-for-test",
        )
        assert report_with.run_id == "fixed-run-id-for-test"

    def test_total_retrieval_latency_is_nonnegative(self):
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus = build_full_version_corpus(dataset)
        report = run_retrieval_evaluation(dataset, LexicalBaselineRetriever(), corpus)
        assert report.total_retrieval_latency_ms >= 0.0

    def test_report_json_never_contains_ground_truth_field_names(self):
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus = build_full_version_corpus(dataset)
        report = run_retrieval_evaluation(dataset, LexicalBaselineRetriever(), corpus)
        dumped = report.model_dump_json()
        for forbidden in ("should_abstain", "rationale", "failure_mode_tag"):
            assert forbidden not in dumped

    def test_custom_k_values_respected(self):
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus = build_full_version_corpus(dataset)
        report = run_retrieval_evaluation(dataset, LexicalBaselineRetriever(), corpus, k_values=(2, 7))
        assert report.top_k_values == [2, 7]
        assert set(report.aggregate_recall_at_k.keys()) == {2, 7}

    def test_case_latency_is_nonnegative(self):
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus = build_full_version_corpus(dataset)
        report = run_retrieval_evaluation(dataset, LexicalBaselineRetriever(), corpus)
        for case_metric in report.case_metrics:
            assert case_metric.latency_ms >= 0.0

    def test_synthetic_v1_dataset_also_runs(self):
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v1")
        corpus = build_full_version_corpus(dataset)
        report = run_retrieval_evaluation(dataset, LexicalBaselineRetriever(), corpus)
        assert len(report.results) == 10


class TestRunRetrievalEvaluationWithVectorRetriever:
    def _build_dataset_corpus_and_index(self, served_model_name=None):
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus = build_full_version_corpus(dataset)

        from app.retrieval.contracts import build_retrieval_query

        vectors: dict[str, tuple[float, ...]] = {}
        for evidence in corpus.evidence:
            # A trivial, deterministic 1-D embedding derived only from
            # evidence_id length -- not tuned against any case, purely to
            # exercise the pipeline without a real embedding model.
            vectors[evidence_embedding_text(evidence)] = (float(len(evidence.evidence_id)),)
        for case in dataset.cases:
            control = dataset.controls[case.control_id]
            query = build_retrieval_query(case, control)
            vectors[retrieval_query_embedding_text(query)] = (float(len(case.case_id)),)

        provider = FakeEmbeddingProvider(
            vectors, model_name="fake-embed-runner-test", served_model_name=served_model_name,
        )
        index = build_corpus_embedding_index(corpus, provider)
        return dataset, corpus, provider, index

    def test_runs_end_to_end_with_full_embedding_provenance_recorded(self):
        dataset, corpus, provider, index = self._build_dataset_corpus_and_index()
        retriever = VectorRetriever(provider, index)

        report = run_retrieval_evaluation(dataset, retriever, corpus, embedding_index=index)

        assert len(report.results) == 30
        assert report.requested_embedding_model == "fake-embed-runner-test"
        assert report.served_embedding_model is None  # fake reports nothing by default -- never fabricated
        assert report.embedding_provider == index.provider
        assert report.embedding_dimension == index.embedding_dimension
        assert report.corpus_fingerprint == corpus.fingerprint
        assert report.retriever_config_id == retriever.retriever_config_id
        # Corpus embedded exactly once (at build_corpus_embedding_index
        # time above), then one embed_query per case -- never re-embedded
        # per case inside the runner.
        assert provider.embed_call_count == 1 + 30

    def test_served_embedding_model_recorded_when_provider_reports_one(self):
        dataset, corpus, provider, index = self._build_dataset_corpus_and_index(served_model_name="fake-served-v1")
        retriever = VectorRetriever(provider, index)
        report = run_retrieval_evaluation(dataset, retriever, corpus, embedding_index=index)
        assert report.served_embedding_model == "fake-served-v1"

    def test_corpus_embedding_latency_threaded_through_when_supplied(self):
        dataset, corpus, provider, index = self._build_dataset_corpus_and_index()
        retriever = VectorRetriever(provider, index)
        report = run_retrieval_evaluation(
            dataset, retriever, corpus, embedding_index=index, corpus_embedding_latency_ms=42.5,
        )
        assert report.corpus_embedding_latency_ms == 42.5

    def test_case_latency_excludes_corpus_embedding_cost(self):
        """Per-case latency must reflect only that case's retrieve() call
        -- not the (much larger, one-time) corpus embedding cost supplied
        separately via corpus_embedding_latency_ms."""
        dataset, corpus, provider, index = self._build_dataset_corpus_and_index()
        retriever = VectorRetriever(provider, index)
        report = run_retrieval_evaluation(
            dataset, retriever, corpus, embedding_index=index, corpus_embedding_latency_ms=10_000.0,
        )
        for case_metric in report.case_metrics:
            assert case_metric.latency_ms < 1_000.0  # nowhere near the 10s corpus-embedding figure
