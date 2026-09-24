"""Unit tests for app.reranking.artifact_loader.load_preserved_retrieval_artifact.

Reads the REAL preserved retrieval-baseline-v1 artifact read-only (never
modifies it, never writes anything under results/); negative cases use a
tmp_path copy so no test can corrupt the real preserved file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.reranking.artifact_loader import ArtifactVerificationError, load_preserved_retrieval_artifact
from app.retrieval.runner import RetrievalReport

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESERVED_ARTIFACT_PATH = (
    REPO_ROOT / "results" / "retrieval_experiments"
    / "synthetic-v2__retrieval-baseline-v1__5b363b87-f452-4d1b-90e1-1f2cd1c2ef36.json"
)
EXPECTED_SHA256 = "e2e234a2cc8339e4a3d15410f04fcd63a74a15b1317e428defc32fce4a737b05"
EXPECTED_GIT_COMMIT_SHA = "e6c609548e9aa24f8f27862f5852cc55e7eafe68"
EXPECTED_DATASET_VERSION = "synthetic-v2"
EXPECTED_DATASET_FINGERPRINT = "35e54a143b90b9d8bf30db7e1cbdb27346eddba0e0c71dd8070e69f22fe63a72"
EXPECTED_CORPUS_FINGERPRINT = "8b9c0462d085addf206667e35aea7e7cc3cfb9473a4961ec902bc3e71e99a109"
EXPECTED_RETRIEVER_CONFIG_ID = "retrieval-baseline-v1"


class TestLoadPreservedArtifactSuccess:
    def test_loads_with_correct_hash_and_full_provenance(self):
        report = load_preserved_retrieval_artifact(
            PRESERVED_ARTIFACT_PATH,
            expected_sha256=EXPECTED_SHA256,
            expected_git_commit_sha=EXPECTED_GIT_COMMIT_SHA,
            expected_dataset_version=EXPECTED_DATASET_VERSION,
            expected_dataset_fingerprint=EXPECTED_DATASET_FINGERPRINT,
            expected_corpus_fingerprint=EXPECTED_CORPUS_FINGERPRINT,
            expected_retriever_config_id=EXPECTED_RETRIEVER_CONFIG_ID,
        )
        assert isinstance(report, RetrievalReport)
        assert report.run_id == "5b363b87-f452-4d1b-90e1-1f2cd1c2ef36"
        assert len(report.results) == 30
        assert len(report.case_metrics) == 30

    def test_loads_with_hash_only_no_other_checks_required(self):
        report = load_preserved_retrieval_artifact(PRESERVED_ARTIFACT_PATH, expected_sha256=EXPECTED_SHA256)
        assert report.dataset_version == "synthetic-v2"

    def test_frozen_candidate_lists_are_reusable(self):
        """The whole point of this loader: a future reranking experiment
        can pull RetrievalResult.candidates straight from here."""
        report = load_preserved_retrieval_artifact(PRESERVED_ARTIFACT_PATH, expected_sha256=EXPECTED_SHA256)
        ac001 = next(r for r in report.results if r.case_id == "AC-001")
        assert len(ac001.candidates) == 10  # top_k_values max was 10
        assert all(c.rank >= 1 for c in ac001.candidates)

    def test_reading_never_modifies_the_real_artifact(self):
        before = PRESERVED_ARTIFACT_PATH.read_bytes()
        load_preserved_retrieval_artifact(PRESERVED_ARTIFACT_PATH, expected_sha256=EXPECTED_SHA256)
        after = PRESERVED_ARTIFACT_PATH.read_bytes()
        assert before == after


class TestLoadPreservedArtifactFailsClosed:
    def test_wrong_expected_sha256_raises(self):
        with pytest.raises(ArtifactVerificationError, match="SHA-256"):
            load_preserved_retrieval_artifact(PRESERVED_ARTIFACT_PATH, expected_sha256="0" * 64)

    def test_tampered_copy_fails_hash_check(self, tmp_path):
        original = PRESERVED_ARTIFACT_PATH.read_text()
        tampered = original.replace('"dataset_version":"synthetic-v2"', '"dataset_version":"synthetic-v1"')
        if tampered == original:
            # Formatting differs from a naive replace target; fall back to
            # a guaranteed-to-differ tamper (append whitespace).
            tampered = original + " "
        tampered_path = tmp_path / "tampered.json"
        tampered_path.write_text(tampered)

        with pytest.raises(ArtifactVerificationError, match="SHA-256"):
            load_preserved_retrieval_artifact(tampered_path, expected_sha256=EXPECTED_SHA256)

    def test_wrong_expected_git_commit_sha_raises(self):
        with pytest.raises(ArtifactVerificationError, match="git_commit_sha"):
            load_preserved_retrieval_artifact(
                PRESERVED_ARTIFACT_PATH, expected_sha256=EXPECTED_SHA256,
                expected_git_commit_sha="0" * 40,
            )

    def test_wrong_expected_dataset_fingerprint_raises(self):
        with pytest.raises(ArtifactVerificationError, match="dataset_fingerprint"):
            load_preserved_retrieval_artifact(
                PRESERVED_ARTIFACT_PATH, expected_sha256=EXPECTED_SHA256,
                expected_dataset_fingerprint="0" * 64,
            )

    def test_wrong_expected_corpus_fingerprint_raises(self):
        with pytest.raises(ArtifactVerificationError, match="corpus_fingerprint"):
            load_preserved_retrieval_artifact(
                PRESERVED_ARTIFACT_PATH, expected_sha256=EXPECTED_SHA256,
                expected_corpus_fingerprint="0" * 64,
            )

    def test_wrong_expected_retriever_config_id_raises(self):
        with pytest.raises(ArtifactVerificationError, match="retriever_config_id"):
            load_preserved_retrieval_artifact(
                PRESERVED_ARTIFACT_PATH, expected_sha256=EXPECTED_SHA256,
                expected_retriever_config_id="not-the-real-config-id",
            )

    def test_missing_file_raises_oserror(self, tmp_path):
        with pytest.raises(OSError):
            load_preserved_retrieval_artifact(tmp_path / "does-not-exist.json", expected_sha256=EXPECTED_SHA256)

    def test_valid_hash_but_malformed_json_content_fails_to_parse(self, tmp_path):
        import hashlib

        bad_content = json.dumps({"not": "a valid RetrievalReport"})
        bad_hash = hashlib.sha256(bad_content.encode("utf-8")).hexdigest()
        bad_path = tmp_path / "bad.json"
        bad_path.write_text(bad_content)

        with pytest.raises(Exception):  # pydantic ValidationError -- hash matches, content doesn't parse
            load_preserved_retrieval_artifact(bad_path, expected_sha256=bad_hash)
