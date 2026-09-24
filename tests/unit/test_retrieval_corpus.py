"""Unit tests for app.retrieval.corpus: the FULL VERSION CORPUS boundary.

The critical invariant under test: a retriever's candidate corpus must
contain evidence that is NOT in the current case's evidence_pool -- using
only the case's own pool would make retrieval trivially easy by
construction. See app.retrieval.corpus's module docstring.
"""

from __future__ import annotations

from pathlib import Path

from app.datasets.loader import load_dataset
from app.models.contracts import Evidence
from app.retrieval.corpus import EvidenceCorpus, build_full_version_corpus, compute_corpus_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATASET_DIR = REPO_ROOT / "datasets"


def _evidence(**overrides) -> Evidence:
    defaults = dict(
        evidence_id="EV-1", document_id="doc_1", title="A Title", period="2025-Q1",
        content="Some content.", structured_fields={"k": "v"},
    )
    defaults.update(overrides)
    return Evidence(**defaults)


class TestFullVersionCorpus:
    def test_corpus_is_much_larger_than_any_single_case_evidence_pool(self):
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus = build_full_version_corpus(dataset)
        assert len(corpus) == 72
        for case in dataset.cases:
            assert len(corpus) > len(case.evidence_pool)

    def test_corpus_contains_evidence_outside_a_specific_cases_pool(self):
        """The exact invariant the task calls out explicitly: pick a real
        case, and prove the corpus offered to a retriever contains
        evidence NOT in that case's own evidence_pool."""
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus = build_full_version_corpus(dataset)
        case = dataset.cases_by_id["AC-001"]
        corpus_ids = {e.evidence_id for e in corpus.evidence}
        pool_ids = set(case.evidence_pool)
        assert pool_ids <= corpus_ids  # the case's own pool IS present...
        assert corpus_ids - pool_ids  # ...but so is plenty more.

    def test_corpus_evidence_ids_are_unique(self):
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus = build_full_version_corpus(dataset)
        ids = [e.evidence_id for e in corpus.evidence]
        assert len(ids) == len(set(ids))

    def test_corpus_order_is_deterministic_sorted_by_evidence_id(self):
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus = build_full_version_corpus(dataset)
        ids = [e.evidence_id for e in corpus.evidence]
        assert ids == sorted(ids)

    def test_repeated_builds_are_identical(self):
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        first = build_full_version_corpus(dataset)
        second = build_full_version_corpus(dataset)
        assert [e.evidence_id for e in first.evidence] == [e.evidence_id for e in second.evidence]

    def test_v1_corpus_is_smaller_and_scoped_to_v1(self):
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v1")
        corpus = build_full_version_corpus(dataset)
        assert len(corpus) == 23
        assert corpus.version == "synthetic-v1"

    def test_v2_corpus_version_label(self):
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus = build_full_version_corpus(dataset)
        assert corpus.version == "synthetic-v2"

    def test_len_matches_evidence_list_length(self):
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus = build_full_version_corpus(dataset)
        assert len(corpus) == len(corpus.evidence)

    def test_dataset_evidence_json_files_not_modified_by_this_test_module(self):
        """Documents (does not itself enforce) that this test suite proves
        the corpus boundary using the REAL, unmodified dataset files --
        datasets/evidence.json was never touched to make these tests pass."""
        dataset = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        assert dataset.fingerprint == "35e54a143b90b9d8bf30db7e1cbdb27346eddba0e0c71dd8070e69f22fe63a72"


class TestCorpusFingerprint:
    """Phase 7B hardening: EvidenceCorpus.fingerprint is content identity
    for exactly what a retriever/CorpusEmbeddingIndex actually searched
    over -- see compute_corpus_fingerprint's own docstring."""

    def test_fingerprint_is_always_populated_even_for_a_hand_built_corpus(self):
        corpus = EvidenceCorpus(version="test-v", evidence=[_evidence()])
        assert corpus.fingerprint
        assert len(corpus.fingerprint) == 64  # SHA-256 hex digest

    def test_same_content_different_insertion_order_same_fingerprint(self):
        ev_a, ev_b = _evidence(evidence_id="EV-A"), _evidence(evidence_id="EV-B")
        corpus_1 = EvidenceCorpus(version="test-v", evidence=[ev_a, ev_b])
        corpus_2 = EvidenceCorpus(version="test-v", evidence=[ev_b, ev_a])
        assert corpus_1.fingerprint == corpus_2.fingerprint

    def test_changed_content_field_changes_fingerprint(self):
        corpus_1 = EvidenceCorpus(version="test-v", evidence=[_evidence(content="original")])
        corpus_2 = EvidenceCorpus(version="test-v", evidence=[_evidence(content="changed")])
        assert corpus_1.fingerprint != corpus_2.fingerprint

    def test_changed_title_changes_fingerprint(self):
        corpus_1 = EvidenceCorpus(version="test-v", evidence=[_evidence(title="Original Title")])
        corpus_2 = EvidenceCorpus(version="test-v", evidence=[_evidence(title="Changed Title")])
        assert corpus_1.fingerprint != corpus_2.fingerprint

    def test_changed_period_changes_fingerprint(self):
        corpus_1 = EvidenceCorpus(version="test-v", evidence=[_evidence(period="2025-Q1")])
        corpus_2 = EvidenceCorpus(version="test-v", evidence=[_evidence(period="2025-Q2")])
        assert corpus_1.fingerprint != corpus_2.fingerprint

    def test_changed_structured_field_changes_fingerprint(self):
        corpus_1 = EvidenceCorpus(version="test-v", evidence=[_evidence(structured_fields={"k": "v1"})])
        corpus_2 = EvidenceCorpus(version="test-v", evidence=[_evidence(structured_fields={"k": "v2"})])
        assert corpus_1.fingerprint != corpus_2.fingerprint

    def test_fingerprint_does_not_depend_on_version_label(self):
        """The fingerprint is content identity only -- changing just the
        version string with identical evidence must not move it (the
        version-string check in VectorRetriever is a SEPARATE, weaker
        check; see test_vector_retriever.py)."""
        ev = _evidence()
        corpus_1 = EvidenceCorpus(version="version-a", evidence=[ev])
        corpus_2 = EvidenceCorpus(version="version-b", evidence=[ev])
        assert corpus_1.fingerprint == corpus_2.fingerprint

    def test_compute_corpus_fingerprint_matches_evidence_corpus_fingerprint(self):
        records = [_evidence(evidence_id="EV-A"), _evidence(evidence_id="EV-B")]
        corpus = EvidenceCorpus(version="test-v", evidence=records)
        assert corpus.fingerprint == compute_corpus_fingerprint(records)

    def test_real_v2_corpus_fingerprint_is_deterministic_across_loads(self):
        dataset_a = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        dataset_b = load_dataset(REAL_DATASET_DIR, version="synthetic-v2")
        corpus_a = build_full_version_corpus(dataset_a)
        corpus_b = build_full_version_corpus(dataset_b)
        assert corpus_a.fingerprint == corpus_b.fingerprint

    def test_real_v1_and_v2_corpus_fingerprints_differ(self):
        v1_corpus = build_full_version_corpus(load_dataset(REAL_DATASET_DIR, version="synthetic-v1"))
        v2_corpus = build_full_version_corpus(load_dataset(REAL_DATASET_DIR, version="synthetic-v2"))
        assert v1_corpus.fingerprint != v2_corpus.fingerprint
