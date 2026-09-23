"""The evidence corpus boundary: what a retriever is allowed to search over.

``EvaluationCase.evidence_pool`` is the small, hand-curated set of evidence
a benchmark case author attached to that ONE case. Using it as a
retriever's search corpus would make retrieval trivial by construction --
the retriever would only ever have to rank 1-4 pre-selected documents,
never actually distinguish relevant evidence from a genuinely large,
irrelevant candidate set. It is ground truth about intended difficulty, not
a search index, and must never be handed to a ``Retriever`` as one.

``EvidenceCorpus`` is deliberately a separate type from anything on
``Dataset`` (rather than just passing ``dict[str, Evidence]`` around) so
that "this is a retrieval corpus, not a case's evidence_pool" is visible at
every call site's type, not just by convention.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.datasets.loader import Dataset
from app.models.contracts import Evidence


@dataclass(frozen=True)
class EvidenceCorpus:
    """The pool of ``Evidence`` a ``Retriever`` may search over.

    ``evidence`` is a plain list (not a dict) in a fixed, deterministic
    order -- sorted by ``evidence_id`` -- so a retriever's tie-breaking
    behavior cannot accidentally depend on Python dict/set iteration order.
    """

    version: str
    evidence: list[Evidence]

    def __len__(self) -> int:
        return len(self.evidence)


def build_full_version_corpus(dataset: Dataset) -> EvidenceCorpus:
    """FULL VERSION CORPUS (Phase 7A): every ``Evidence`` record reachable
    by ``dataset``'s version -- i.e. the union across every case in that
    version, exactly ``dataset.evidence`` -- NOT any single case's
    ``evidence_pool``.

    For ``synthetic-v2`` this is 72 evidence records (vs. 1-4 per case),
    so a retriever exercised against this corpus must actually discriminate
    relevant from irrelevant evidence rather than trivially "finding" a
    pre-curated handful. Evidence attached only to a version other than
    ``dataset.version`` (there is none yet, since every version is a subset
    of the same shared files, but future versions could in principle carry
    version-exclusive evidence) is excluded, since ``dataset.evidence`` is
    already filtered to exactly what ``dataset.version``'s cases reference
    (see ``app.datasets.loader.load_dataset``).
    """

    return EvidenceCorpus(
        version=dataset.version,
        evidence=sorted(dataset.evidence.values(), key=lambda ev: ev.evidence_id),
    )
