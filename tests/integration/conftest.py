"""Regression guard: no integration test may alter the repository's real
results/ directory, ever -- not by writing, not by deleting.

This is the artifact-safety counterpart to tests/conftest.py's network
barrier: session-scoped and autouse, so it cannot be skipped by omission.
It does not prevent a bug (the CLI itself is what must never be pointed at
the real directory -- see ARL_RESULTS_DIR in scripts/run_eval.py and
docs/incidents/2026-09-23-llm-baseline-artifact-deletion.md for why that
matters), but it turns any regression into an immediate, loud, session-level
test failure rather than a silent loss discovered days later.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_RESULTS_DIR = REPO_ROOT / "results"


def _snapshot(directory: Path) -> dict[str, str]:
    if not directory.exists():
        return {}
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="session", autouse=True)
def _real_results_dir_untouched_by_integration_tests():
    before = _snapshot(REAL_RESULTS_DIR)
    yield
    after = _snapshot(REAL_RESULTS_DIR)
    assert after == before, (
        "A test altered the repository's real results/ directory. Every CLI "
        "subprocess launched from a test MUST set ARL_RESULTS_DIR to redirect "
        "its output into a pytest tmp_path -- never operate on the real "
        "results/ directory. See "
        "docs/incidents/2026-09-23-llm-baseline-artifact-deletion.md for why "
        "this guard exists."
    )
