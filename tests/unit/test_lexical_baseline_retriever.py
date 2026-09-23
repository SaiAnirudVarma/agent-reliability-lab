"""Unit tests for app.retrieval.lexical_baseline.LexicalBaselineRetriever.

Uses small, synthetic, hand-built fixtures throughout -- NOT tuned against
AC-001..AC-030 (see the module's own docstring on why that would defeat its
purpose as a plumbing check, not an experimental result).
"""

from __future__ import annotations

import pytest

from app.models.contracts import Control, Evidence
from app.retrieval.contracts import RetrievalQuery
from app.retrieval.corpus import EvidenceCorpus
from app.retrieval.lexical_baseline import RETRIEVER_CONFIG_ID, LexicalBaselineRetriever


def _control(**overrides) -> Control:
    defaults = dict(
        control_id="CTRL-X", name="Firewall Recertification", description="Quarterly firewall rule review.",
        requirement_text="Recertify all inbound firewall rules every quarter.",
    )
    defaults.update(overrides)
    return Control(**defaults)


def _evidence(evidence_id, document_id, title, content) -> Evidence:
    return Evidence(
        evidence_id=evidence_id, document_id=document_id, title=title, period="2025-Q1",
        content=content, structured_fields={},
    )


def _corpus(*records) -> EvidenceCorpus:
    return EvidenceCorpus(version="test", evidence=list(records))


def _query(case_id="AC-999", control=None, scenario="Evaluate firewall recertification.") -> RetrievalQuery:
    return RetrievalQuery(
        case_id=case_id, control=control or _control(), scenario_description=scenario, period="2025-Q1",
    )


RELEVANT = _evidence(
    "EV-1", "doc_firewall", "Firewall Recertification Record",
    "Quarterly firewall rule review completed, all inbound rules recertified.",
)
DISTRACTOR = _evidence(
    "EV-2", "doc_unrelated", "Vendor Risk Assessment",
    "Third-party vendor risk assessment for cloud storage provider, unrelated to firewall rules.",
)
PARTIAL = _evidence(
    "EV-3", "doc_partial", "Change Management Log",
    "Quarterly change log; firewall rule change included among many unrelated changes.",
)


class TestDeterminism:
    def test_repeated_calls_produce_identical_results(self):
        retriever = LexicalBaselineRetriever()
        corpus = _corpus(RELEVANT, DISTRACTOR, PARTIAL)
        query = _query()
        first = retriever.retrieve(query, corpus, top_k=3)
        second = retriever.retrieve(query, corpus, top_k=3)
        assert [c.model_dump() for c in first.candidates] == [c.model_dump() for c in second.candidates]

    def test_relevant_evidence_outranks_unrelated_distractor(self):
        retriever = LexicalBaselineRetriever()
        corpus = _corpus(DISTRACTOR, RELEVANT, PARTIAL)  # deliberately scrambled input order
        result = retriever.retrieve(_query(), corpus, top_k=3)
        ranked_ids = [c.evidence_id for c in result.candidates]
        assert ranked_ids.index("EV-1") < ranked_ids.index("EV-2")

    def test_score_is_not_case_id_dependent(self):
        """Changing only case_id (all other query content identical) must
        not change scores or ranking at all -- proves no case-ID-specific
        retrieval logic exists."""
        retriever = LexicalBaselineRetriever()
        corpus = _corpus(RELEVANT, DISTRACTOR, PARTIAL)
        result_a = retriever.retrieve(_query(case_id="AC-001"), corpus, top_k=3)
        result_b = retriever.retrieve(_query(case_id="AC-999"), corpus, top_k=3)
        scores_a = [(c.evidence_id, c.score) for c in result_a.candidates]
        scores_b = [(c.evidence_id, c.score) for c in result_b.candidates]
        assert scores_a == scores_b


class TestTieBreaking:
    def test_equal_scores_break_ties_by_evidence_id_ascending(self):
        """Two records with IDENTICAL text (hence identical scores) must
        order by evidence_id, deterministically, regardless of corpus order."""
        twin_b = _evidence("EV-B", "doc_b", "Identical Title Text", "Identical content text here.")
        twin_a = _evidence("EV-A", "doc_a", "Identical Title Text", "Identical content text here.")
        retriever = LexicalBaselineRetriever()
        corpus = _corpus(twin_b, twin_a)  # EV-B listed first in the corpus
        result = retriever.retrieve(_query(), corpus, top_k=2)
        assert [c.evidence_id for c in result.candidates] == ["EV-A", "EV-B"]

    def test_tie_break_independent_of_corpus_insertion_order(self):
        twin_a = _evidence("EV-A", "doc_a", "Same Words Same Words", "Same words same words same words.")
        twin_b = _evidence("EV-B", "doc_b", "Same Words Same Words", "Same words same words same words.")
        retriever = LexicalBaselineRetriever()
        order1 = retriever.retrieve(_query(), _corpus(twin_a, twin_b), top_k=2)
        order2 = retriever.retrieve(_query(), _corpus(twin_b, twin_a), top_k=2)
        ids1 = [c.evidence_id for c in order1.candidates]
        ids2 = [c.evidence_id for c in order2.candidates]
        assert ids1 == ids2 == ["EV-A", "EV-B"]


class TestTopK:
    def test_top_k_limits_candidate_count(self):
        retriever = LexicalBaselineRetriever()
        corpus = _corpus(RELEVANT, DISTRACTOR, PARTIAL)
        result = retriever.retrieve(_query(), corpus, top_k=1)
        assert len(result.candidates) == 1

    def test_top_k_larger_than_corpus_returns_whole_corpus(self):
        retriever = LexicalBaselineRetriever()
        corpus = _corpus(RELEVANT, DISTRACTOR)
        result = retriever.retrieve(_query(), corpus, top_k=50)
        assert len(result.candidates) == 2
        assert result.top_k == 50

    def test_ranks_are_1_indexed_and_sequential(self):
        retriever = LexicalBaselineRetriever()
        corpus = _corpus(RELEVANT, DISTRACTOR, PARTIAL)
        result = retriever.retrieve(_query(), corpus, top_k=3)
        assert [c.rank for c in result.candidates] == [1, 2, 3]

    @pytest.mark.parametrize("bad_top_k", [0, -1, -100])
    def test_invalid_top_k_raises(self, bad_top_k):
        retriever = LexicalBaselineRetriever()
        corpus = _corpus(RELEVANT)
        with pytest.raises(ValueError, match="top_k"):
            retriever.retrieve(_query(), corpus, top_k=bad_top_k)


class TestEvidenceIdentityAndProvenance:
    def test_document_id_preserved_alongside_evidence_id(self):
        retriever = LexicalBaselineRetriever()
        corpus = _corpus(RELEVANT)
        result = retriever.retrieve(_query(), corpus, top_k=1)
        assert result.candidates[0].evidence_id == "EV-1"
        assert result.candidates[0].document_id == "doc_firewall"

    def test_retriever_config_id_is_the_documented_constant(self):
        retriever = LexicalBaselineRetriever()
        assert retriever.retriever_config_id == RETRIEVER_CONFIG_ID == "lexical-baseline-v1"

    def test_result_and_candidates_carry_matching_config_id(self):
        retriever = LexicalBaselineRetriever()
        corpus = _corpus(RELEVANT)
        result = retriever.retrieve(_query(), corpus, top_k=1)
        assert result.retriever_config_id == RETRIEVER_CONFIG_ID
        assert all(c.retrieval_method == RETRIEVER_CONFIG_ID for c in result.candidates)

    def test_query_is_embedded_in_the_result(self):
        retriever = LexicalBaselineRetriever()
        query = _query(case_id="AC-042")
        result = retriever.retrieve(query, _corpus(RELEVANT), top_k=1)
        assert result.query == query
        assert result.case_id == "AC-042"


class TestSatisfiesRetrieverProtocol:
    def test_isinstance_check(self):
        from app.retrieval.interface import Retriever

        assert isinstance(LexicalBaselineRetriever(), Retriever)
