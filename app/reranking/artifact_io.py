"""Immutable reranking-experiment artifact naming and writing.

Mirrors ``app.retrieval.artifact_io``'s design exactly, under its own
separate, clearly distinguished path -- ``results/reranking_experiments/``
-- so a real reranking experiment can never collide with, or be confused
for, a retrieval or LLM experiment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from app.reranking.runner import RerankingReport


def reranking_experiments_dir(results_dir: Path) -> Path:
    """``results_dir`` mirrors ``scripts/run_eval.py``'s ``RESULTS_DIR``
    convention -- callers pass the same, possibly ``ARL_RESULTS_DIR``
    -overridden, directory used for every other result path."""

    return results_dir / "reranking_experiments"


def reranking_experiment_path(results_dir: Path, dataset_version: str, reranker_config_id: str, run_id: str) -> Path:
    """The immutable, collision-proof destination for one real reranking
    run: ``reranking_experiments/<dataset-version>__<reranker-config-id>__<run-id>.json``.

    ``run_id`` must be decided by the caller BEFORE this path is used to
    check for a collision -- see ``scripts/run_reranking_eval.py`` -- so
    the exact destination is known, and can be checked, before any
    provider call.
    """

    return reranking_experiments_dir(results_dir) / f"{dataset_version}__{reranker_config_id}__{run_id}.json"


def resolve_run_id() -> str:
    """A fresh UUID4, or the ``ARL_RUN_ID`` override -- the same
    test-injection mechanism used throughout this project's experiment
    scripts, reused here rather than a competing implementation."""

    return os.environ.get("ARL_RUN_ID") or str(uuid4())


def write_reranking_report_atomically(path: Path, report: "RerankingReport") -> None:
    """Writes ``report`` to ``path`` atomically (temp file + ``os.replace``),
    exactly mirroring ``app.retrieval.artifact_io.write_retrieval_report_atomically``.
    Callers must check for collision BEFORE calling this -- it does not
    check for you, and unconditionally overwrites ``path`` if it already
    exists.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{uuid4().hex}")
    try:
        tmp_path.write_text(report.model_dump_json(indent=2) + "\n")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
