"""The two deterministic text-serialization functions that define exactly
what gets embedded: one for an ``Evidence`` record, one for a
``RetrievalQuery``. Each is a single, auditable source of truth for its
representation -- never assembled ad hoc at a call site -- so "what did the
retriever actually search against" is always answerable from this module
alone.
"""

from __future__ import annotations

from app.models.contracts import Evidence
from app.retrieval.contracts import RetrievalQuery


def evidence_embedding_text(evidence: Evidence) -> str:
    """The text representation of one ``Evidence`` record that gets
    embedded.

    Includes only fields legitimately available at inference time --
    ``title``, ``period``, ``content``, and a canonical (key-sorted)
    rendering of ``structured_fields``. ``Evidence`` has no
    ``ExpectedOutcome``-shaped field to accidentally include in the first
    place, but the exact assembly is fixed here so the representation
    itself is deterministic and documented, not incidental to call-site
    string-formatting choices.

    Format (order is part of the contract):

        {title}
        {period}
        {content}
        {structured_fields sorted by key, one "key: value" line each}
    """

    lines = [evidence.title, evidence.period, evidence.content]
    for key in sorted(evidence.structured_fields):
        lines.append(f"{key}: {evidence.structured_fields[key]}")
    return "\n".join(lines)


def retrieval_query_embedding_text(query: RetrievalQuery) -> str:
    """The text representation of a ``RetrievalQuery`` that gets embedded.

    Uses only ``control.name``/``control.description``/
    ``control.requirement_text``, ``scenario_description``, and ``period``.
    Deliberately never ``query.case_id`` -- it remains opaque tracing
    metadata only (see ``RetrievalQuery``'s own docstring); two queries
    differing only in ``case_id`` must embed to the exact same text.

    Format:

        {control.name}
        {control.description}
        {control.requirement_text}
        {scenario_description}
        {period}
    """

    return "\n".join(
        [
            query.control.name,
            query.control.description,
            query.control.requirement_text,
            query.scenario_description,
            query.period,
        ]
    )
