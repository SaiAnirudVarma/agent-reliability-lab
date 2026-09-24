"""Unit tests for app.retrieval.vector_retriever.VectorRetriever, using
FakeEmbeddingProvider throughout -- zero network, exact engineered
cosine-similarity relationships so rankings are fully predictable.
"""

from __future__ import annotations

import pytest

from app.models.contracts import Control, Evidence
from app.retrieval.contracts import RetrievalQuery
from app.retrieval.corpus import EvidenceCorpus
from app.retrieval.corpus_index import build_corpus_embedding_index
from app.retrieval.embedding_provider import FakeEmbeddingProvider
from app.retrieval.serialization import evidence_embedding_text, retrieval_query_embedding_text
from app.retrieval.vector_retriever import VectorRetriever


def _control(**overrides) -> Control:
    defaults = dict(control_id="CTRL-X", name="N", description="D", requirement_text="R")
    defaults.update(overrides)
    return Control(**defaults)


def _evidence(evidence_id, title="T", content="C") -> Evidence:
    return Evidence(evidence_id=evidence_id, document_id=f"doc_{evidence_id}", title=title, period="2025-Q1", content=content, structured_fields={})


def _query(case_id="AC-999", control=None, scenario="S", period="2025-Q1") -> RetrievalQuery:
    return RetrievalQuery(case_id=case_id, control=control or _control(), scenario_description=scenario, period=period)


def _corpus(*records, version="test-v") -> EvidenceCorpus:
    return EvidenceCorpus(version=version, evidence=list(records))


class TestExactCosineOrdering:
    def test_ranks_by_descending_similarity_to_query(self):
        ev_close = _evidence("EV-close", title="Close", content="close content")
        ev_mid = _evidence("EV-mid", title="Mid", content="mid content")
        ev_far = _evidence("EV-far", title="Far", content="far content")
        query = _query()
        query_text = retrieval_query_embedding_text(query)

        vectors = {
            query_text: (1.0, 0.0),
            evidence_embedding_text(ev_close): (1.0, 0.0),   # sim 1.0
            evidence_embedding_text(ev_mid): (1.0, 1.0),      # sim ~0.707
            evidence_embedding_text(ev_far): (0.0, 1.0),      # sim 0.0
        }
        provider = FakeEmbeddingProvider(vectors)
        corpus = _corpus(ev_far, ev_close, ev_mid)  # scrambled input order
        index = build_corpus_embedding_index(corpus, provider)
        retriever = VectorRetriever(provider, index)

        result = retriever.retrieve(query, corpus, top_k=3)
        assert [c.evidence_id for c in result.candidates] == ["EV-close", "EV-mid", "EV-far"]
        assert result.candidates[0].score == pytest.approx(1.0)
        assert result.candidates[2].score == pytest.approx(0.0)

    def test_scores_are_actual_cosine_values_not_ranks(self):
        ev = _evidence("EV-1")
        query = _query()
        vectors = {retrieval_query_embedding_text(query): (1.0, 0.0), evidence_embedding_text(ev): (1.0, 1.0)}
        provider = FakeEmbeddingProvider(vectors)
        corpus = _corpus(ev)
        index = build_corpus_embedding_index(corpus, provider)
        retriever = VectorRetriever(provider, index)
        result = retriever.retrieve(query, corpus, top_k=1)
        assert result.candidates[0].score == pytest.approx(1 / 2 ** 0.5)


class TestStableTieBreaking:
    def test_equal_similarity_breaks_tie_by_evidence_id_ascending(self):
        ev_b, ev_a = _evidence("EV-B"), _evidence("EV-A")
        query = _query()
        vectors = {
            retrieval_query_embedding_text(query): (1.0, 0.0),
            evidence_embedding_text(ev_b): (1.0, 0.0),
            evidence_embedding_text(ev_a): (1.0, 0.0),
        }
        provider = FakeEmbeddingProvider(vectors)
        corpus = _corpus(ev_b, ev_a)  # EV-B listed first
        index = build_corpus_embedding_index(corpus, provider)
        retriever = VectorRetriever(provider, index)
        result = retriever.retrieve(query, corpus, top_k=2)
        assert [c.evidence_id for c in result.candidates] == ["EV-A", "EV-B"]


class TestZeroVectorHandling:
    def test_zero_vector_evidence_scores_zero_not_error(self):
        ev_zero = _evidence("EV-zero", title="Zero", content="zero content")
        ev_normal = _evidence("EV-normal", title="Normal", content="normal content")
        query = _query()
        vectors = {
            retrieval_query_embedding_text(query): (1.0, 0.0),
            evidence_embedding_text(ev_zero): (0.0, 0.0),
            evidence_embedding_text(ev_normal): (1.0, 0.0),
        }
        provider = FakeEmbeddingProvider(vectors)
        corpus = _corpus(ev_zero, ev_normal)
        index = build_corpus_embedding_index(corpus, provider)
        retriever = VectorRetriever(provider, index)
        result = retriever.retrieve(query, corpus, top_k=2)
        assert [c.evidence_id for c in result.candidates] == ["EV-normal", "EV-zero"]
        assert result.candidates[1].score == 0.0


class TestFullCorpusSearched:
    def test_every_corpus_record_is_a_scoring_candidate(self):
        records = [_evidence(f"EV-{i}", content=f"content {i}") for i in range(10)]
        query = _query()
        vectors = {retrieval_query_embedding_text(query): (1.0,)}
        for r in records:
            vectors[evidence_embedding_text(r)] = (float(hash(r.evidence_id) % 7 + 1),)
        provider = FakeEmbeddingProvider(vectors)
        corpus = _corpus(*records)
        index = build_corpus_embedding_index(corpus, provider)
        retriever = VectorRetriever(provider, index)
        result = retriever.retrieve(query, corpus, top_k=10)
        assert len(result.candidates) == 10
        assert {c.evidence_id for c in result.candidates} == {r.evidence_id for r in records}


class TestCaseIdHasNoSemanticInfluence:
    def test_only_case_id_differs_scores_identical(self):
        ev = _evidence("EV-1")
        control = _control()
        # both queries produce the SAME embedding text (case_id excluded) --
        # key the fake provider by that full serialized text, not the raw
        # scenario string, since that's what embed_query actually receives.
        query_a = RetrievalQuery(case_id="AC-001", control=control, scenario_description="S1", period="2025-Q1")
        query_b = RetrievalQuery(case_id="AC-999", control=control, scenario_description="S1", period="2025-Q1")
        vectors = {
            retrieval_query_embedding_text(query_a): (1.0, 0.0),
            evidence_embedding_text(ev): (1.0, 1.0),
        }
        provider = FakeEmbeddingProvider(vectors)
        corpus = _corpus(ev)
        index = build_corpus_embedding_index(corpus, provider)
        retriever = VectorRetriever(provider, index)

        result_a = retriever.retrieve(query_a, corpus, top_k=1)
        result_b = retriever.retrieve(query_b, corpus, top_k=1)
        assert result_a.candidates[0].score == result_b.candidates[0].score


class TestCorpusEmbeddedOnceQueriesIndependently:
    def test_document_embedding_not_recomputed_across_multiple_retrieve_calls(self):
        ev1 = _evidence("EV-1", title="One", content="one content")
        ev2 = _evidence("EV-2", title="Two", content="two content")
        control = _control()
        queries = [
            RetrievalQuery(case_id="AC-X", control=control, scenario_description=scenario, period="2025-Q1")
            for scenario in ("S-a", "S-b", "S-c")
        ]
        vectors = {
            evidence_embedding_text(ev1): (1.0, 0.0), evidence_embedding_text(ev2): (0.0, 1.0),
            retrieval_query_embedding_text(queries[0]): (1.0, 0.0),
            retrieval_query_embedding_text(queries[1]): (0.0, 1.0),
            retrieval_query_embedding_text(queries[2]): (1.0, 1.0),
        }
        provider = FakeEmbeddingProvider(vectors)
        corpus = _corpus(ev1, ev2)
        index = build_corpus_embedding_index(corpus, provider)  # 1 call so far
        retriever = VectorRetriever(provider, index)

        for query in queries:
            retriever.retrieve(query, corpus, top_k=2)

        # 1 (corpus build) + 3 (one embed_query per case) = 4 total calls,
        # never 1 + 3*2 (which would mean documents were re-embedded per query).
        assert provider.embed_call_count == 4


class TestTopKAndConfig:
    def test_top_k_limits_results(self):
        records = [_evidence(f"EV-{i}") for i in range(5)]
        query = _query()
        vectors = {retrieval_query_embedding_text(query): (1.0,)}
        for r in records:
            vectors[evidence_embedding_text(r)] = (1.0,)
        provider = FakeEmbeddingProvider(vectors)
        corpus = _corpus(*records)
        index = build_corpus_embedding_index(corpus, provider)
        retriever = VectorRetriever(provider, index)
        result = retriever.retrieve(query, corpus, top_k=2)
        assert len(result.candidates) == 2

    @pytest.mark.parametrize("bad_top_k", [0, -1])
    def test_invalid_top_k_raises(self, bad_top_k):
        ev = _evidence("EV-1")
        query = _query()
        provider = FakeEmbeddingProvider({retrieval_query_embedding_text(query): (1.0,), evidence_embedding_text(ev): (1.0,)})
        corpus = _corpus(ev)
        index = build_corpus_embedding_index(corpus, provider)
        retriever = VectorRetriever(provider, index)
        with pytest.raises(ValueError, match="top_k"):
            retriever.retrieve(query, corpus, top_k=bad_top_k)

    def test_retriever_config_id_defaults_to_model_name_derived(self):
        ev = _evidence("EV-1")
        provider = FakeEmbeddingProvider({"q": (1.0,), evidence_embedding_text(ev): (1.0,)}, model_name="fake-embed-v9")
        corpus = _corpus(ev)
        index = build_corpus_embedding_index(corpus, provider)
        retriever = VectorRetriever(provider, index)
        assert retriever.retriever_config_id == "vector-fake-embed-v9"

    def test_retriever_config_id_explicit_override(self):
        ev = _evidence("EV-1")
        provider = FakeEmbeddingProvider({"q": (1.0,), evidence_embedding_text(ev): (1.0,)})
        corpus = _corpus(ev)
        index = build_corpus_embedding_index(corpus, provider)
        retriever = VectorRetriever(provider, index, retriever_config_id="my-custom-id")
        assert retriever.retriever_config_id == "my-custom-id"

    def test_candidates_carry_the_retriever_config_id_as_method(self):
        ev = _evidence("EV-1")
        query = RetrievalQuery(case_id="AC-X", control=_control(), scenario_description="q", period="2025-Q1")
        provider = FakeEmbeddingProvider({
            retrieval_query_embedding_text(query): (1.0,), evidence_embedding_text(ev): (1.0,),
        })
        corpus = _corpus(ev)
        index = build_corpus_embedding_index(corpus, provider)
        retriever = VectorRetriever(provider, index)
        result = retriever.retrieve(query, corpus, top_k=1)
        assert result.candidates[0].retrieval_method == retriever.retriever_config_id

    def test_model_name_mismatch_between_provider_and_index_rejected_at_construction(self):
        ev = _evidence("EV-1")
        provider_a = FakeEmbeddingProvider({"x": (1.0,)}, model_name="model-a")
        provider_b = FakeEmbeddingProvider({evidence_embedding_text(ev): (1.0,)}, model_name="model-b")
        corpus = _corpus(ev)
        index = build_corpus_embedding_index(corpus, provider_b)  # index built with model-b
        with pytest.raises(ValueError, match="does not match"):
            VectorRetriever(provider_a, index)  # constructed with model-a

    def test_corpus_version_mismatch_rejected_at_retrieve_time(self):
        ev = _evidence("EV-1")
        provider = FakeEmbeddingProvider({"q": (1.0,), evidence_embedding_text(ev): (1.0,)})
        corpus = _corpus(ev, version="v-original")
        index = build_corpus_embedding_index(corpus, provider)
        retriever = VectorRetriever(provider, index)
        mismatched_corpus = _corpus(ev, version="v-different")
        query = RetrievalQuery(case_id="AC-X", control=_control(), scenario_description="q", period="2025-Q1")
        with pytest.raises(ValueError, match="does not match"):
            retriever.retrieve(query, mismatched_corpus, top_k=1)

    def test_same_version_string_but_changed_corpus_content_rejected(self):
        """The exact invariant item 3 calls for: version.__eq__ alone is
        NOT sufficient -- a corpus sharing the SAME version string but
        with genuinely different evidence content must still be rejected,
        via the fingerprint check."""
        ev_original = _evidence("EV-1", title="Original", content="original content")
        provider = FakeEmbeddingProvider({
            "q": (1.0,), evidence_embedding_text(ev_original): (1.0,),
        })
        corpus = _corpus(ev_original, version="same-version-label")
        index = build_corpus_embedding_index(corpus, provider)
        retriever = VectorRetriever(provider, index)

        ev_changed = _evidence("EV-1", title="Original", content="CHANGED content")  # same evidence_id, different content
        changed_corpus = _corpus(ev_changed, version="same-version-label")  # identical version string
        assert changed_corpus.version == corpus.version
        assert changed_corpus.fingerprint != corpus.fingerprint

        query = RetrievalQuery(case_id="AC-X", control=_control(), scenario_description="q", period="2025-Q1")
        with pytest.raises(ValueError, match="fingerprint"):
            retriever.retrieve(query, changed_corpus, top_k=1)


class TestSatisfiesRetrieverProtocol:
    def test_isinstance_check(self):
        from app.retrieval.interface import Retriever

        ev = _evidence("EV-1")
        provider = FakeEmbeddingProvider({"q": (1.0,), evidence_embedding_text(ev): (1.0,)})
        corpus = _corpus(ev)
        index = build_corpus_embedding_index(corpus, provider)
        retriever = VectorRetriever(provider, index)
        assert isinstance(retriever, Retriever)
