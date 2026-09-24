"""Immutable reranked-LLM-experiment artifact naming and writing.

Mirrors ``app.reranking.artifact_io``'s design exactly, under its own
separate, clearly distinguished path -- ``results/reranked_llm_experiments/``
-- so a real reranked-LLM experiment can never collide with, or be
confused for, a retrieval, reranking, or oracle-context LLM experiment.

``resolve_run_id`` is intentionally NOT redefined here -- it is imported
directly from ``app.reranking.artifact_io``, since that function is
already generic (a fresh UUID4, or the ``ARL_RUN_ID`` test-injection
override) and has nothing reranking-specific about it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from app.reranking.artifact_io import resolve_run_id

if TYPE_CHECKING:
    from app.integration.runner import RerankedLLMReport

__all__ = ["reranked_llm_experiments_dir", "reranked_llm_experiment_path", "resolve_run_id",
           "write_reranked_llm_report_atomically"]


def reranked_llm_experiments_dir(results_dir: Path) -> Path:
    """``results_dir`` mirrors ``scripts/run_eval.py``'s ``RESULTS_DIR``
    convention -- callers pass the same, possibly ``ARL_RESULTS_DIR``
    -overridden, directory used for every other result path."""

    return results_dir / "reranked_llm_experiments"


def reranked_llm_experiment_path(results_dir: Path, dataset_version: str, config_id: str, run_id: str) -> Path:
    """The immutable, collision-proof destination for one real
    reranked-LLM run:
    ``reranked_llm_experiments/<dataset-version>__<config-id>__<run-id>.json``.

    ``run_id`` must be decided by the caller BEFORE this path is used to
    check for a collision -- see ``scripts/run_reranked_llm_eval.py`` --
    so the exact destination is known, and can be checked, before any
    provider call.
    """

    return reranked_llm_experiments_dir(results_dir) / f"{dataset_version}__{config_id}__{run_id}.json"


def write_reranked_llm_report_atomically(path: Path, report: "RerankedLLMReport") -> None:
    """Writes ``report`` to ``path`` atomically (temp file + ``os.replace``),
    exactly mirroring ``app.reranking.artifact_io.write_reranking_report_atomically``.
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
