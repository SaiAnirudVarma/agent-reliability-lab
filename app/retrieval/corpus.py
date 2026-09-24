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

import hashlib
import json
from dataclasses import dataclass, field

from app.datasets.loader import Dataset
from app.models.contracts import Evidence


def compute_corpus_fingerprint(evidence: list[Evidence]) -> str:
    """SHA-256 over a canonical JSON representation of ``evidence``, sorted
    by ``evidence_id`` -- NEVER insertion order, so two corpora holding the
    exact same records in a different order fingerprint identically.

    Mirrors ``app.datasets.loader``'s own fingerprint recipe exactly (each
    record serialized via its own ``model_dump(mode="json")``, the whole
    payload dumped with ``sort_keys=True`` and fixed separators) applied to
    a retrieval corpus instead of a dataset version's selected cases. This
    proves what a ``CorpusEmbeddingIndex`` actually embedded: it changes if
    ANY field of ANY record changes (title, period, content,
    structured_fields) and is stable under nothing else. ``Evidence`` has
    no ground-truth field to accidentally include in the first place.
    """

    sorted_evidence = sorted(evidence, key=lambda ev: ev.evidence_id)
    payload = [ev.model_dump(mode="json") for ev in sorted_evidence]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvidenceCorpus:
    """The pool of ``Evidence`` a ``Retriever`` may search over.

    ``evidence`` is a plain list (not a dict) in a fixed, deterministic
    order -- sorted by ``evidence_id`` -- so a retriever's tie-breaking
    behavior cannot accidentally depend on Python dict/set iteration order.

    ``fingerprint`` is ``init=False``: it is always DERIVED from
    ``evidence`` via ``compute_corpus_fingerprint``, never supplied by a
    caller, so it can never drift out of sync with the actual content --
    including for a hand-built test fixture constructed directly (not via
    ``build_full_version_corpus``).
    """

    version: str
    evidence: list[Evidence]
    fingerprint: str = field(init=False, default="")

    def __post_init__(self) -> None:
        object.__setattr__(self, "fingerprint", compute_corpus_fingerprint(self.evidence))

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
