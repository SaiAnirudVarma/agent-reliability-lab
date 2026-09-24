"""Loads a preserved retrieval artifact as the candidate source for a
reranking experiment -- verifying every provenance fact a
``RerankerBaselineConfig`` requires BEFORE any reranker provider call is
made. No embeddings, no network, no re-retrieval: this reuses frozen,
already-paid-for candidate lists exactly as they were produced.
"""

from __future__ import annotations

from pathlib import Path

from app.reranking.artifact_loader import load_preserved_retrieval_artifact
from app.reranking.baseline_config import RerankerBaselineConfig
from app.retrieval.contracts import RetrievedEvidence
from app.retrieval.runner import RetrievalReport


class CandidateSourceValidationError(Exception):
    """Raised when the preserved retrieval artifact does not satisfy a
    ``RerankerBaselineConfig``'s requirements. Always fails closed -- a
    caller must never construct or invoke a reranker provider after this
    is raised."""


def load_and_validate_candidate_source(config: RerankerBaselineConfig, artifact_path: Path) -> RetrievalReport:
    """Loads ``artifact_path`` -- SHA-256 and provenance-checked against
    ``config``'s own recorded expectations (dataset/corpus fingerprint,
    retriever_config_id) via ``load_preserved_retrieval_artifact`` -- and
    additionally verifies:

    1. the artifact's ``run_id`` matches ``config.candidate_source_run_id``
    2. the artifact's own ``git_commit_sha`` is present (not ``None`` --
       an artifact with incomplete provenance is never used as a source)
    3. the artifact contains exactly 30 results
    4. every result has at least ``config.candidate_depth`` candidates

    Raises ``CandidateSourceValidationError`` (or
    ``app.reranking.artifact_loader.ArtifactVerificationError``, allowed to
    propagate unchanged) on any failure. Makes no provider call itself.
    """

    report = load_preserved_retrieval_artifact(
        artifact_path,
        expected_sha256=config.candidate_source_artifact_sha256,
        expected_dataset_version=config.dataset_version,
        expected_dataset_fingerprint=config.dataset_fingerprint,
        expected_corpus_fingerprint=config.corpus_fingerprint,
        expected_retriever_config_id=config.candidate_source_retriever_config_id,
    )

    if report.run_id != config.candidate_source_run_id:
        raise CandidateSourceValidationError(
            f"artifact run_id {report.run_id!r} does not match config's "
            f"candidate_source_run_id {config.candidate_source_run_id!r}"
        )

    if report.git_commit_sha is None:
        raise CandidateSourceValidationError(
            "preserved retrieval artifact has no recorded git_commit_sha -- refusing to use an "
            "artifact with incomplete provenance as a reranking candidate source."
        )

    if len(report.results) != 30:
        raise CandidateSourceValidationError(f"expected exactly 30 retrieval results, got {len(report.results)}")

    too_shallow = [
        result.case_id for result in report.results if len(result.candidates) < config.candidate_depth
    ]
    if too_shallow:
        raise CandidateSourceValidationError(
            f"case(s) {sorted(too_shallow)} have fewer preserved candidates than the configured "
            f"candidate_depth={config.candidate_depth}"
        )

    return report


def build_candidate_sets(report: RetrievalReport, candidate_depth: int) -> dict[str, list[RetrievedEvidence]]:
    """For every case in an already-validated ``report``, returns the
    first ``candidate_depth`` preserved candidates, sorted by rank --
    exactly what a reranker would receive. A pure truncation of already-
    frozen data; never re-derives, re-ranks, or re-scores anything."""

    return {
        result.case_id: sorted(result.candidates, key=lambda c: c.rank)[:candidate_depth]
        for result in report.results
    }
