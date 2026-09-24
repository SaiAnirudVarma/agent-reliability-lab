"""Immutable retrieval-experiment artifact naming and writing.

Mirrors ``scripts/run_eval.py``'s ``results/experiments/`` design
(``docs/ARTIFACT_POLICY.md``) for retrieval-only experiments, under a
separate, clearly distinguished path -- ``results/retrieval_experiments/``
-- so a real retrieval experiment can never collide with, or be confused
for, a real LLM experiment.
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from app.retrieval.runner import RetrievalReport


def retrieval_experiments_dir(results_dir: Path) -> Path:
    """``results_dir`` mirrors ``scripts/run_eval.py``'s ``RESULTS_DIR``
    convention -- callers pass the same, possibly ``ARL_RESULTS_DIR``
    -overridden, directory used for every other result path, so this
    naming stays test-isolated the same way."""

    return results_dir / "retrieval_experiments"


def retrieval_experiment_path(results_dir: Path, dataset_version: str, retriever_config_id: str, run_id: str) -> Path:
    """The immutable, collision-proof destination for one real retrieval
    run: ``retrieval_experiments/<dataset-version>__<retriever-config-id>__<run-id>.json``.

    ``run_id`` must be decided by the caller BEFORE this path is used to
    check for a collision -- see ``scripts/run_retrieval_eval.py`` -- so
    the exact destination is known, and can be checked, before any
    provider call.
    """

    return retrieval_experiments_dir(results_dir) / f"{dataset_version}__{retriever_config_id}__{run_id}.json"


def resolve_run_id() -> str:
    """A fresh UUID4, or the ``ARL_RUN_ID`` override -- the exact same
    test-injection mechanism ``scripts/run_eval.py`` uses for the LLM
    experiment path, reused here rather than a competing implementation."""

    return os.environ.get("ARL_RUN_ID") or str(uuid4())


def write_retrieval_report_atomically(path: Path, report: RetrievalReport) -> None:
    """Writes ``report`` to ``path`` atomically: the content is written to
    a sibling temp file first, then moved into place with ``os.replace``
    (an atomic rename on POSIX and Windows alike), so a crash or
    interruption mid-write can never leave a half-written file sitting at
    the real destination path -- there is either a complete file there, or
    none at all.

    Callers must check for collision (``path.exists()``) BEFORE calling
    this -- it does not check for you, and unconditionally overwrites
    ``path`` if it already exists (matching ``os.replace``'s own
    semantics). This function is the WRITE mechanism, not the
    immutability policy; the caller (``scripts/run_retrieval_eval.py``)
    enforces "never overwrite an existing experiment" before ever reaching
    this call.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{uuid4().hex}")
    try:
        tmp_path.write_text(report.model_dump_json(indent=2) + "\n")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
