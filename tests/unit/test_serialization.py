"""Unit tests for app.retrieval.serialization: the two deterministic
text-serialization functions defining what gets embedded.
"""

from __future__ import annotations

from app.models.contracts import Control, Evidence
from app.retrieval.contracts import RetrievalQuery
from app.retrieval.serialization import evidence_embedding_text, retrieval_query_embedding_text


def _evidence(**overrides) -> Evidence:
    defaults = dict(
        evidence_id="EV-1", document_id="doc_1", title="A Title", period="2025-Q1",
        content="Some content.", structured_fields={"b_key": "2", "a_key": "1"},
    )
    defaults.update(overrides)
    return Evidence(**defaults)


def _control(**overrides) -> Control:
    defaults = dict(control_id="CTRL-X", name="Ctrl Name", description="Ctrl description.", requirement_text="Ctrl requirement.")
    defaults.update(overrides)
    return Control(**defaults)


def _query(**overrides) -> RetrievalQuery:
    defaults = dict(case_id="AC-999", control=_control(), scenario_description="Scenario text.", period="2025-Q1")
    defaults.update(overrides)
    return RetrievalQuery(**defaults)


class TestEvidenceEmbeddingText:
    def test_deterministic_across_calls(self):
        evidence = _evidence()
        assert evidence_embedding_text(evidence) == evidence_embedding_text(evidence)

    def test_contains_title_period_content(self):
        evidence = _evidence()
        text = evidence_embedding_text(evidence)
        assert "A Title" in text
        assert "2025-Q1" in text
        assert "Some content." in text

    def test_structured_fields_rendered_sorted_by_key(self):
        evidence = _evidence(structured_fields={"z_key": "last", "a_key": "first"})
        text = evidence_embedding_text(evidence)
        assert text.index("a_key: first") < text.index("z_key: last")

    def test_exact_documented_format(self):
        evidence = _evidence(
            title="T", period="2025-Q1", content="C", structured_fields={"k1": "v1", "k2": "v2"},
        )
        assert evidence_embedding_text(evidence) == "T\n2025-Q1\nC\nk1: v1\nk2: v2"

    def test_empty_structured_fields_produces_no_extra_lines(self):
        evidence = _evidence(structured_fields={})
        assert evidence_embedding_text(evidence) == "A Title\n2025-Q1\nSome content."

    def test_never_includes_anything_from_expected_outcome(self):
        """Evidence has no such field structurally, but this documents the
        invariant explicitly: nothing benchmark-relevance-labeled can enter
        this text."""
        evidence = _evidence()
        text = evidence_embedding_text(evidence)
        for forbidden in ("required_evidence_ids", "expected_finding", "rationale", "failure_mode_tag"):
            assert forbidden not in text


class TestRetrievalQueryEmbeddingText:
    def test_deterministic_across_calls(self):
        query = _query()
        assert retrieval_query_embedding_text(query) == retrieval_query_embedding_text(query)

    def test_contains_control_and_scenario_and_period(self):
        text = retrieval_query_embedding_text(_query())
        assert "Ctrl Name" in text
        assert "Ctrl description." in text
        assert "Ctrl requirement." in text
        assert "Scenario text." in text
        assert "2025-Q1" in text

    def test_exact_documented_format(self):
        control = _control(name="N", description="D", requirement_text="R")
        query = _query(control=control, scenario_description="S", period="2025-Q1")
        assert retrieval_query_embedding_text(query) == "N\nD\nR\nS\n2025-Q1"

    def test_case_id_has_no_effect_on_embedding_text(self):
        """The exact test the review called for: changing ONLY case_id
        must leave the embedding text byte-for-byte identical."""
        control = _control()
        text_a = retrieval_query_embedding_text(_query(case_id="AC-001", control=control))
        text_b = retrieval_query_embedding_text(_query(case_id="AC-999", control=control))
        assert text_a == text_b

    def test_case_id_string_itself_never_appears_in_the_text(self):
        query = _query(case_id="AC-777-UNIQUE-MARKER")
        assert "AC-777-UNIQUE-MARKER" not in retrieval_query_embedding_text(query)
