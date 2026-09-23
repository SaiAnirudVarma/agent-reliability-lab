"""Unit tests for app.observability.git_provenance.get_git_commit_sha.

Covers the ARL_GIT_COMMIT_SHA injection override (so callers -- including
scripts/run_eval.py's own tests -- never depend on this machine's actual
git state), the real `git rev-parse HEAD` path against this repository,
and the failure modes (not a git repo, git missing/erroring) that must
return None rather than raise.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.observability.git_provenance import get_git_commit_sha

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestOverride:
    def test_override_takes_precedence_and_needs_no_real_git_state(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "deterministic-fake-sha-for-tests")
        # A directory that is not a git repository at all -- if the
        # override were ignored, the real `git rev-parse HEAD` call below
        # would fail and return None instead.
        assert get_git_commit_sha(tmp_path) == "deterministic-fake-sha-for-tests"

    def test_empty_override_is_treated_as_unset(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ARL_GIT_COMMIT_SHA", "")
        assert get_git_commit_sha(tmp_path) is None  # falls through to a real (failing) git call

    def test_no_override_falls_through_to_real_git_call(self, monkeypatch):
        monkeypatch.delenv("ARL_GIT_COMMIT_SHA", raising=False)
        sha = get_git_commit_sha(REPO_ROOT)
        assert sha is not None
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)


class TestFailureModes:
    def test_not_a_git_repository_returns_none_not_raise(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ARL_GIT_COMMIT_SHA", raising=False)
        assert get_git_commit_sha(tmp_path) is None

    def test_git_binary_missing_returns_none(self, monkeypatch):
        monkeypatch.delenv("ARL_GIT_COMMIT_SHA", raising=False)

        def _raise_file_not_found(*args, **kwargs):
            raise FileNotFoundError("git: command not found")

        monkeypatch.setattr(subprocess, "run", _raise_file_not_found)
        assert get_git_commit_sha(REPO_ROOT) is None

    def test_git_timeout_returns_none(self, monkeypatch):
        monkeypatch.delenv("ARL_GIT_COMMIT_SHA", raising=False)

        def _raise_timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="git", timeout=10)

        monkeypatch.setattr(subprocess, "run", _raise_timeout)
        assert get_git_commit_sha(REPO_ROOT) is None

    def test_nonzero_exit_returns_none(self, monkeypatch):
        monkeypatch.delenv("ARL_GIT_COMMIT_SHA", raising=False)

        class _FakeResult:
            returncode = 128
            stdout = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeResult())
        assert get_git_commit_sha(REPO_ROOT) is None

    def test_blank_stdout_on_success_returns_none(self, monkeypatch):
        """Defensive: a returncode of 0 with empty output (should not
        happen in practice) must not be treated as a valid empty-string SHA."""
        monkeypatch.delenv("ARL_GIT_COMMIT_SHA", raising=False)

        class _FakeResult:
            returncode = 0
            stdout = "  \n"

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeResult())
        assert get_git_commit_sha(REPO_ROOT) is None
