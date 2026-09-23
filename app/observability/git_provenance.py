"""Git commit provenance capture for RunReport.

Captured exactly once per run -- never once per case/trace -- so a real LLM
experiment's result can always be traced back to the exact source code
(including the evaluator/metrics modules themselves; see
``RunReport.git_commit_sha``) that produced it.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


def get_git_commit_sha(repo_root: Path) -> Optional[str]:
    """The current HEAD commit SHA, or ``None`` if it cannot be determined.

    Checks the ``ARL_GIT_COMMIT_SHA`` environment variable first -- an
    explicit override so tests can inject a deterministic value without
    depending on this machine's actual git state or shelling out at all
    (mirrors ``ARL_RESULTS_DIR`` / ``ARL_DOTENV_PATH`` in
    ``scripts/run_eval.py``). Falls back to running ``git rev-parse HEAD``
    once against ``repo_root``.

    Returns ``None`` -- never raises -- on any failure: not a git
    repository, git not installed, or a non-zero exit. It is the caller's
    job to decide whether that failure is fatal (see ``scripts/run_eval.py``:
    fatal before a real LLM run spends any money; non-fatal, best-effort for
    a free, reproducible deterministic run).
    """

    override = os.environ.get("ARL_GIT_COMMIT_SHA")
    if override:
        return override

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    sha = result.stdout.strip()
    return sha or None
