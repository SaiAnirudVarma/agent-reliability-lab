"""Loader/validator for a frozen retrieval-baseline configuration file
(e.g. ``configs/retrieval-baseline-v1.json``): the non-secret, tracked
record of exactly which reproducibility parameters an official retrieval
experiment run used.

Credentials/API keys are NEVER stored here -- they remain environment-only
(see ``scripts/run_retrieval_eval.py``). This file's job is to make an
official run's parameters explicit and versioned in git, not to configure
secrets.

``extra="forbid"`` means a config file cannot silently carry an
unrecognized field (e.g. a typo, or a field someone assumed existed) --
loading fails loudly instead.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class RetrievalBaselineConfig(BaseModel):
    """A frozen retrieval experiment configuration.

    This is deliberately data, not code: ``query_serialization`` and
    ``evidence_serialization`` are human-readable descriptions of which
    functions produced the frozen ``corpus_fingerprint`` below (see
    ``app.retrieval.serialization``), not something this loader executes
    or dispatches on. If those functions ever change, the fingerprint they
    produce will change too, and ``scripts/run_retrieval_eval.py``'s
    fingerprint verification (see its ``--config`` handling) will catch
    the drift before any provider call -- the description fields are for
    a human reader, the fingerprints are what's actually enforced.
    """

    model_config = ConfigDict(extra="forbid")

    config_id: str = Field(min_length=1)

    embedding_provider: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding_dimension: int = Field(gt=0)

    retrieval_method: str = Field(min_length=1)
    similarity: str = Field(min_length=1)
    tie_breaking: str = Field(min_length=1)

    query_serialization: str = Field(min_length=1)
    evidence_serialization: str = Field(min_length=1)

    corpus: str = Field(min_length=1)
    corpus_size: int = Field(gt=0)
    corpus_fingerprint: str = Field(min_length=1)

    dataset_version: str = Field(min_length=1)
    dataset_fingerprint: str = Field(min_length=1)

    k_values: list[int] = Field(min_length=1)
    mrr_convention: str = Field(min_length=1)
    zero_required_evidence_convention: str = Field(min_length=1)


def load_retrieval_baseline_config(path: Path) -> RetrievalBaselineConfig:
    """Loads and validates a frozen config file. Raises ``OSError`` if the
    file cannot be read, or ``pydantic.ValidationError`` if its content
    doesn't match the schema -- the caller (``scripts/run_retrieval_eval.py``)
    treats both as a configuration failure and refuses to proceed."""

    return RetrievalBaselineConfig.model_validate_json(path.read_text())
