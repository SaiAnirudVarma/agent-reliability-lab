"""Loads a preserved reranking artifact as the evidence source for a
reranked-LLM experiment -- verifying every provenance fact a
``RerankedLLMBaselineConfig`` requires BEFORE any LLM provider call is
made. No embeddings, no Cohere calls, no re-retrieval, no re-reranking:
this reuses a frozen, already-paid-for reranking result exactly as it was
produced (see ``app.reranking.runner.RerankingReport``).

Mirrors ``app.reranking.artifact_loader`` (SHA-256 + provenance
verification) and ``app.reranking.candidate_source`` (the additional
run_id/git/count/depth checks a config requires) exactly, one layer up the
pipeline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from app.integration.config import RerankedLLMBaselineConfig
from app.reranking.contracts import RerankedEvidence
from app.reranking.runner import RerankingReport


class ArtifactVerificationError(Exception):
    """Raised when a preserved reranking artifact's actual SHA-256 or
    provenance does not match the caller's expected values. The caller
    must treat this as fatal and refuse to proceed -- never fall back to
    using the artifact anyway."""


class RerankedEvidenceSourceValidationError(Exception):
    """Raised when the preserved reranking artifact does not satisfy a
    ``RerankedLLMBaselineConfig``'s requirements. Always fails closed -- a
    caller must never construct or invoke an LLM provider after this is
    raised."""


def load_preserved_reranking_artifact(
    path: Path,
    *,
    expected_sha256: str,
    expected_git_commit_sha: Optional[str] = None,
    expected_dataset_version: Optional[str] = None,
    expected_dataset_fingerprint: Optional[str] = None,
    expected_corpus_fingerprint: Optional[str] = None,
    expected_reranker_config_id: Optional[str] = None,
) -> RerankingReport:
    """Reads ``path`` (raw bytes, unmodified), verifies its SHA-256 against
    ``expected_sha256``, parses it as a ``RerankingReport``, and verifies
    any other ``expected_*`` field the caller supplies against the
    report's own recorded value.

    Raises ``ArtifactVerificationError`` on any mismatch, or ``OSError`` if
    ``path`` cannot be read. Never partially trusts a failed check. Has no
    write path at all -- this module never modifies, rewrites, or
    normalizes the artifact it loads.
    """

    raw = path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ArtifactVerificationError(
            f"{path}: actual SHA-256 {actual_sha256!r} does not match expected {expected_sha256!r}. "
            "Refusing to use this artifact -- it may have been modified or is not the file the "
            "caller believes it is."
        )

    report = RerankingReport.model_validate_json(raw)

    checks = [
        ("git_commit_sha", expected_git_commit_sha, report.git_commit_sha),
        ("dataset_version", expected_dataset_version, report.dataset_version),
        ("dataset_fingerprint", expected_dataset_fingerprint, report.dataset_fingerprint),
        ("corpus_fingerprint", expected_corpus_fingerprint, report.corpus_fingerprint),
        ("reranker_config_id", expected_reranker_config_id, report.reranker_config_id),
    ]
    for field_name, expected_value, actual_value in checks:
        if expected_value is not None and actual_value != expected_value:
            raise ArtifactVerificationError(
                f"{path}: {field_name} {actual_value!r} does not match expected {expected_value!r}."
            )

    return report


def load_and_validate_reranking_source(config: RerankedLLMBaselineConfig, artifact_path: Path) -> RerankingReport:
    """Loads ``artifact_path`` -- SHA-256 and provenance-checked against
    ``config``'s own recorded expectations -- and additionally verifies:

    1. the artifact's ``run_id`` matches ``config.source_reranking_run_id``
    2. the artifact's own ``git_commit_sha`` is present (not ``None`` -- an
       artifact with incomplete provenance is never used as a source)
    3. the artifact contains exactly 30 rerank results
    4. every result has at least ``config.evidence_top_k`` candidates
    5. ``config.evidence_top_k`` is one of the artifact's own
       ``evaluation_k_values`` -- so this experiment's before/after
       comparison capability (Recall@evidence_top_k, before vs. after
       reranking) can be read directly off the preserved artifact rather
       than recomputed

    Raises ``RerankedEvidenceSourceValidationError`` (or
    ``ArtifactVerificationError``, allowed to propagate unchanged) on any
    failure. Makes no provider call itself.
    """

    report = load_preserved_reranking_artifact(
        artifact_path,
        expected_sha256=config.source_reranking_artifact_sha256,
        expected_dataset_version=config.dataset_version,
        expected_dataset_fingerprint=config.dataset_fingerprint,
        expected_corpus_fingerprint=config.corpus_fingerprint,
        expected_reranker_config_id=config.source_reranking_config_id,
    )

    if report.run_id != config.source_reranking_run_id:
        raise RerankedEvidenceSourceValidationError(
            f"artifact run_id {report.run_id!r} does not match config's "
            f"source_reranking_run_id {config.source_reranking_run_id!r}"
        )

    if report.git_commit_sha is None:
        raise RerankedEvidenceSourceValidationError(
            "preserved reranking artifact has no recorded git_commit_sha -- refusing to use an "
            "artifact with incomplete provenance as a reranked-LLM evidence source."
        )

    if len(report.rerank_results) != 30:
        raise RerankedEvidenceSourceValidationError(
            f"expected exactly 30 rerank results, got {len(report.rerank_results)}"
        )

    too_shallow = [
        result.case_id for result in report.rerank_results if len(result.candidates) < config.evidence_top_k
    ]
    if too_shallow:
        raise RerankedEvidenceSourceValidationError(
            f"case(s) {sorted(too_shallow)} have fewer preserved reranked candidates than the "
            f"configured evidence_top_k={config.evidence_top_k}"
        )

    if config.evidence_top_k not in report.evaluation_k_values:
        raise RerankedEvidenceSourceValidationError(
            f"config's evidence_top_k={config.evidence_top_k} is not among the preserved artifact's "
            f"own evaluation_k_values={report.evaluation_k_values} -- before/after comparison at this "
            "K cannot be read from the artifact."
        )

    return report


def build_top_k_candidates_by_case(
    report: RerankingReport, evidence_top_k: int
) -> dict[str, list[RerankedEvidence]]:
    """For every case in an already-validated ``report``, returns the
    first ``evidence_top_k`` reranked candidates, sorted by
    ``reranked_rank`` -- exactly what the agent will receive. A pure
    truncation of already-frozen data; never re-derives, re-ranks, or
    re-scores anything."""

    return {
        result.case_id: sorted(result.candidates, key=lambda c: c.reranked_rank)[:evidence_top_k]
        for result in report.rerank_results
    }
