"""Safe, read-only loader for a PRESERVED retrieval experiment artifact --
lets a future reranking experiment reuse frozen, already-paid-for
candidate lists (``RetrievalResult.candidates``) instead of making new
embedding calls.

Fails closed on any SHA-256 or provenance mismatch, exactly mirroring
``scripts/run_retrieval_eval.py --config``'s "verify before proceeding"
discipline, applied here to an artifact instead of a config file. Never
writes, rewrites, or normalizes the artifact -- this module has no write
path at all.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from app.retrieval.runner import RetrievalReport


class ArtifactVerificationError(Exception):
    """Raised when a preserved artifact's actual SHA-256 or provenance
    does not match the caller's expected values. The caller (a future
    reranking runner) must treat this as fatal and refuse to proceed --
    never fall back to using the artifact anyway."""


def load_preserved_retrieval_artifact(
    path: Path,
    *,
    expected_sha256: str,
    expected_git_commit_sha: Optional[str] = None,
    expected_dataset_version: Optional[str] = None,
    expected_dataset_fingerprint: Optional[str] = None,
    expected_corpus_fingerprint: Optional[str] = None,
    expected_retriever_config_id: Optional[str] = None,
) -> RetrievalReport:
    """Reads ``path`` (raw bytes, unmodified), verifies its SHA-256 against
    ``expected_sha256``, parses it as a ``RetrievalReport``, and verifies
    any other ``expected_*`` field the caller supplies against the
    report's own recorded value. Every ``expected_*`` beyond the hash is
    optional so a caller can check only what it cares about -- but the
    hash check is mandatory and always runs first, since a byte-level
    mismatch makes every other field's value untrustworthy anyway.

    Raises ``ArtifactVerificationError`` on any mismatch, or ``OSError``
    if ``path`` cannot be read. Never partially trusts a failed check.
    """

    raw = path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ArtifactVerificationError(
            f"{path}: actual SHA-256 {actual_sha256!r} does not match expected {expected_sha256!r}. "
            "Refusing to use this artifact -- it may have been modified or is not the file the "
            "caller believes it is."
        )

    report = RetrievalReport.model_validate_json(raw)

    checks = [
        ("git_commit_sha", expected_git_commit_sha, report.git_commit_sha),
        ("dataset_version", expected_dataset_version, report.dataset_version),
        ("dataset_fingerprint", expected_dataset_fingerprint, report.dataset_fingerprint),
        ("corpus_fingerprint", expected_corpus_fingerprint, report.corpus_fingerprint),
        ("retriever_config_id", expected_retriever_config_id, report.retriever_config_id),
    ]
    for field_name, expected_value, actual_value in checks:
        if expected_value is not None and actual_value != expected_value:
            raise ArtifactVerificationError(
                f"{path}: {field_name} {actual_value!r} does not match expected {expected_value!r}."
            )

    return report
