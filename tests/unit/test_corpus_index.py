"""Unit tests for app.retrieval.corpus_index.build_corpus_embedding_index --
the embed-once corpus cache, and its full provenance record.
"""

from __future__ import annotations

import pytest

from app.models.contracts import Evidence, ModelProvider
from app.retrieval.corpus import EvidenceCorpus
from app.retrieval.corpus_index import build_corpus_embedding_index
from app.retrieval.embedding_provider import EmbeddedVector, FakeEmbeddingProvider
from app.retrieval.serialization import evidence_embedding_text


def _evidence(evidence_id, title="T", content="C") -> Evidence:
    return Evidence(evidence_id=evidence_id, document_id=f"doc_{evidence_id}", title=title, period="2025-Q1", content=content, structured_fields={})


def _corpus(*records) -> EvidenceCorpus:
    return EvidenceCorpus(version="test-v", evidence=list(records))


class TestBuildCorpusEmbeddingIndex:
    def test_embeds_every_record_exactly_once_via_one_batch_call(self):
        ev1, ev2, ev3 = _evidence("EV-1", "T1", "C1"), _evidence("EV-2", "T2", "C2"), _evidence("EV-3", "T3", "C3")
        vectors_by_text = {
            evidence_embedding_text(ev1): (1.0, 0.0),
            evidence_embedding_text(ev2): (0.0, 1.0),
            evidence_embedding_text(ev3): (1.0, 1.0),
        }
        provider = FakeEmbeddingProvider(vectors_by_text)
        corpus = _corpus(ev1, ev2, ev3)

        index = build_corpus_embedding_index(corpus, provider)

        assert provider.embed_call_count == 1  # ONE embed_documents() call, not 3
        assert len(index) == 3
        assert index.vectors_by_evidence_id["EV-1"] == (1.0, 0.0)
        assert index.vectors_by_evidence_id["EV-2"] == (0.0, 1.0)
        assert index.vectors_by_evidence_id["EV-3"] == (1.0, 1.0)

    def test_index_carries_full_provenance(self):
        ev1 = _evidence("EV-1")
        provider = FakeEmbeddingProvider({evidence_embedding_text(ev1): (1.0,)}, model_name="fake-embed-x")
        corpus = _corpus(ev1)
        index = build_corpus_embedding_index(corpus, provider)
        assert index.version == "test-v"
        assert index.corpus_fingerprint == corpus.fingerprint
        assert index.provider == ModelProvider.OPENAI
        assert index.model_name == "fake-embed-x"
        assert index.served_model_name is None  # fake reports nothing by default
        assert index.embedding_dimension == 1

    def test_served_model_name_recorded_when_provider_reports_one(self):
        ev1 = _evidence("EV-1")
        provider = FakeEmbeddingProvider(
            {evidence_embedding_text(ev1): (1.0,)}, served_model_name="fake-served-v1",
        )
        index = build_corpus_embedding_index(_corpus(ev1), provider)
        assert index.served_model_name == "fake-served-v1"

    def test_inconsistent_served_model_names_within_one_batch_rejected(self):
        ev1, ev2 = _evidence("EV-1"), _evidence("EV-2")
        provider = FakeEmbeddingProvider({
            evidence_embedding_text(ev1): (1.0,), evidence_embedding_text(ev2): (2.0,),
        })
        corpus = _corpus(ev1, ev2)
        provider.embed_documents = lambda texts: [
            EmbeddedVector(vector=(1.0,), served_model_name="model-a"),
            EmbeddedVector(vector=(2.0,), served_model_name="model-b"),
        ]
        with pytest.raises(ValueError, match="inconsistent served model"):
            build_corpus_embedding_index(corpus, provider)

    def test_evidence_by_id_preserves_full_records(self):
        ev1 = _evidence("EV-1", title="Distinctive Title")
        provider = FakeEmbeddingProvider({evidence_embedding_text(ev1): (1.0,)})
        index = build_corpus_embedding_index(_corpus(ev1), provider)
        assert index.evidence_by_id["EV-1"].title == "Distinctive Title"

    def test_dimension_mismatch_from_provider_raises(self):
        """A provider misbehaving (returning wrong-length vectors for its
        own declared embedding_dimension) must be caught, not silently
        stored into the index."""
        ev1 = _evidence("EV-1")
        provider = FakeEmbeddingProvider({evidence_embedding_text(ev1): (1.0, 2.0)})
        # Force a provider that claims dimension 2 but we monkeypatch it to
        # actually return dimension 3 for this one record.
        provider.embed_documents = lambda texts: [EmbeddedVector(vector=(1.0, 2.0, 3.0)) for _ in texts]
        with pytest.raises(ValueError, match="dimension"):
            build_corpus_embedding_index(_corpus(ev1), provider)

    def test_vector_count_mismatch_from_provider_raises(self):
        ev1, ev2 = _evidence("EV-1"), _evidence("EV-2")
        provider = FakeEmbeddingProvider({
            evidence_embedding_text(ev1): (1.0,), evidence_embedding_text(ev2): (2.0,),
        })
        provider.embed_documents = lambda texts: [EmbeddedVector(vector=(1.0,))]  # 1 vector for 2 texts
        with pytest.raises(ValueError, match="returned 1 vectors"):
            build_corpus_embedding_index(_corpus(ev1, ev2), provider)

    def test_empty_corpus_produces_empty_index(self):
        provider = FakeEmbeddingProvider({"placeholder": (1.0,)})
        empty_corpus = EvidenceCorpus(version="test-v", evidence=[])
        index = build_corpus_embedding_index(empty_corpus, provider)
        assert len(index) == 0
        assert provider.embed_call_count == 1  # still called once, with an empty list
