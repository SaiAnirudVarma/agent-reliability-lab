"""Loader for the synthetic evaluation dataset.

Reads ``datasets/controls.json``, ``datasets/evidence.json`` and
``datasets/eval_cases.json``, validates every record through the Pydantic
contracts in :mod:`app.models.contracts`, checks cross-file referential
integrity (duplicate IDs, dangling control/evidence references), and returns
a single typed :class:`Dataset` container with O(1) lookup maps.

Nothing here is silently skipped: a malformed record, a duplicate ID, or a
dangling reference raises :class:`DatasetError` describing exactly what is
wrong and where.

Dataset versioning (Phase 6): ``datasets/versions.json`` maps a version name
(e.g. ``"synthetic-v1"``) to the ordered list of case IDs that belong to it.
A version is purely a named SUBSET of ``eval_cases.json``'s rows -- new
cases are appended to the shared files, never duplicated per version, and
``synthetic-v1``'s 10 rows are never touched by adding ``synthetic-v2``.
Referential integrity is always checked across the WHOLE file (every
control/evidence reference in every case, regardless of version), so a
broken v2-only case can never ship even when only v1 is requested.
``load_dataset`` defaults to ``version="synthetic-v1"`` specifically so that
every pre-Phase-6 call site (scripts, tests) keeps its exact original
behavior without modification, even after the shared files grow.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

from app.models.contracts import Control, Evidence, EvaluationCase

ModelT = TypeVar("ModelT", bound=BaseModel)


class DatasetError(Exception):
    """Raised when the synthetic dataset fails to load or validate.

    Every raise site in this module folds all problems it finds into a
    single, readable message rather than stopping at the first one — a
    dataset author fixing a bad file wants the whole list of issues, not one
    at a time.
    """


@dataclass(frozen=True)
class Dataset:
    """A fully validated, cross-referenced, version-filtered view of the
    synthetic dataset.

    ``controls``/``evidence``/``cases``/``cases_by_id`` are all filtered to
    exactly what ``version``'s selected cases reference -- this Dataset
    represents that version and nothing extra, even though the underlying
    JSON files hold every version's records together. ``cases`` preserves
    the version manifest's order.
    """

    version: str
    fingerprint: str
    controls: dict[str, Control]
    evidence: dict[str, Evidence]
    cases: list[EvaluationCase]
    cases_by_id: dict[str, EvaluationCase]

    def evidence_for_case(self, case: EvaluationCase) -> list[Evidence]:
        """Resolve a case's evidence_pool into the actual Evidence records, in pool order."""
        return [self.evidence[evidence_id] for evidence_id in case.evidence_pool]


def _read_json_array(path: Path) -> list[dict]:
    if not path.exists():
        raise DatasetError(f"Dataset file not found: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise DatasetError(f"{path} must contain a JSON array of records, got {type(raw).__name__}")
    return raw


def _validate_records(path: Path, model: type[ModelT], raw_records: list[dict]) -> list[ModelT]:
    validated: list[ModelT] = []
    errors: list[str] = []
    for index, record in enumerate(raw_records):
        try:
            validated.append(model.model_validate(record))
        except ValidationError as exc:
            record_id = (
                record.get("control_id") or record.get("evidence_id") or record.get("case_id")
                if isinstance(record, dict)
                else None
            )
            label = f"index {index}" + (f" ({record_id})" if record_id else "")
            errors.append(f"{path.name} record at {label}:\n{exc}")
    if errors:
        raise DatasetError(
            f"{len(errors)} record(s) in {path.name} failed validation:\n\n" + "\n\n".join(errors)
        )
    return validated


def _check_unique_ids(label: str, items: list[ModelT], id_getter: Callable[[ModelT], str]) -> None:
    seen: dict[str, int] = {}
    for item in items:
        item_id = id_getter(item)
        seen[item_id] = seen.get(item_id, 0) + 1
    duplicates = sorted(item_id for item_id, count in seen.items() if count > 1)
    if duplicates:
        raise DatasetError(f"Duplicate {label} id(s): {duplicates}")


def _load_version_manifest(dataset_dir: Path) -> dict:
    path = dataset_dir / "versions.json"
    if not path.exists():
        raise DatasetError(f"Dataset version manifest not found: {path}")
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise DatasetError(f"{path} must contain a JSON object mapping version name -> entry")
    return manifest


def _select_version_case_ids(dataset_dir: Path, version: str, all_case_ids: set[str]) -> list[str]:
    manifest = _load_version_manifest(dataset_dir)
    if version not in manifest:
        raise DatasetError(
            f"Unknown dataset version {version!r}; available: {sorted(manifest)}"
        )
    entry = manifest[version]
    case_ids = entry.get("case_ids") if isinstance(entry, dict) else None
    if not isinstance(case_ids, list) or not case_ids:
        raise DatasetError(f"versions.json entry for {version!r} must have a non-empty 'case_ids' list")
    missing = [cid for cid in case_ids if cid not in all_case_ids]
    if missing:
        raise DatasetError(
            f"Dataset version {version!r} references case ID(s) not present in eval_cases.json: {missing}"
        )
    return case_ids


def _compute_fingerprint(controls: list[Control], evidence: list[Evidence], cases: list[EvaluationCase]) -> str:
    """SHA-256 over a canonical JSON representation of exactly the data that
    defines one dataset version:

    - every Control referenced by ``cases`` (sorted by control_id)
    - every Evidence referenced by ``cases`` (sorted by evidence_id)
    - ``cases`` themselves, in the exact order they will be executed
      (the version manifest's order -- NOT re-sorted, since execution order
      is itself part of what "this dataset version" reproducibly means)

    Each object is serialized via its own ``model_dump(mode="json")``, and
    the whole payload is dumped with ``sort_keys=True`` and fixed
    separators, so neither field-declaration order in the source model nor
    key order in the source JSON file can change the fingerprint -- only
    the actual field VALUES can. Controls/evidence NOT referenced by
    ``cases`` (e.g. ones added later for a different version) never enter
    the hash, so a version's fingerprint is stable even as the shared files
    grow to support later versions.
    """

    used_control_ids = {case.control_id for case in cases}
    used_evidence_ids = {evidence_id for case in cases for evidence_id in case.evidence_pool}
    controls_by_id = {control.control_id: control for control in controls}
    evidence_by_id = {ev.evidence_id: ev for ev in evidence}

    payload = {
        "controls": [controls_by_id[cid].model_dump(mode="json") for cid in sorted(used_control_ids)],
        "evidence": [evidence_by_id[eid].model_dump(mode="json") for eid in sorted(used_evidence_ids)],
        "cases": [case.model_dump(mode="json") for case in cases],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_dataset(dataset_dir: Path | str, version: str = "synthetic-v1") -> Dataset:
    """Load, validate, cross-reference, and version-filter the synthetic dataset.

    Raises :class:`DatasetError` with a specific, actionable message on any
    problem: a missing file, malformed JSON, a record that fails Pydantic
    validation, a duplicate ID, a case that references a control or evidence
    ID that does not exist anywhere in the dataset, or an unknown/malformed
    dataset version.

    ``version`` defaults to ``"synthetic-v1"`` -- the original 10-case
    benchmark -- so every call site written before Phase 6 introduced
    ``synthetic-v2`` keeps its exact original behavior unless it explicitly
    opts into the larger set.
    """

    dataset_dir = Path(dataset_dir)

    controls_path = dataset_dir / "controls.json"
    evidence_path = dataset_dir / "evidence.json"
    cases_path = dataset_dir / "eval_cases.json"

    controls = _validate_records(controls_path, Control, _read_json_array(controls_path))
    evidence = _validate_records(evidence_path, Evidence, _read_json_array(evidence_path))
    all_cases = _validate_records(cases_path, EvaluationCase, _read_json_array(cases_path))

    _check_unique_ids("control", controls, lambda c: c.control_id)
    _check_unique_ids("evidence", evidence, lambda e: e.evidence_id)
    _check_unique_ids("case", all_cases, lambda c: c.case_id)

    control_ids = {c.control_id for c in controls}
    evidence_ids = {e.evidence_id for e in evidence}

    # Cross-reference integrity is checked over EVERY case in the file,
    # regardless of which version is being loaded -- a broken case must
    # never ship silently just because it happens to belong to a version
    # nobody requested this time.
    reference_errors: list[str] = []
    for case in all_cases:
        if case.control_id not in control_ids:
            reference_errors.append(
                f"{case.case_id} references unknown control_id '{case.control_id}'"
            )
        missing_evidence = [eid for eid in case.evidence_pool if eid not in evidence_ids]
        if missing_evidence:
            reference_errors.append(
                f"{case.case_id} references unknown evidence id(s): {missing_evidence}"
            )
    if reference_errors:
        raise DatasetError(
            f"{len(reference_errors)} referential integrity problem(s):\n"
            + "\n".join(reference_errors)
        )

    all_case_ids = {c.case_id for c in all_cases}
    selected_case_ids = _select_version_case_ids(dataset_dir, version, all_case_ids)

    cases_by_id_all = {c.case_id: c for c in all_cases}
    selected_cases = [cases_by_id_all[cid] for cid in selected_case_ids]

    fingerprint = _compute_fingerprint(controls, evidence, selected_cases)

    # Dataset.controls/.evidence are filtered to exactly what the selected
    # version's cases reference -- not the whole shared files -- so
    # `load_dataset(dir)` (version="synthetic-v1") returns a Dataset that IS
    # synthetic-v1, with nothing extra, matching its pre-Phase-6 behavior
    # even as the shared files grow to support later versions.
    used_control_ids = {case.control_id for case in selected_cases}
    used_evidence_ids = {eid for case in selected_cases for eid in case.evidence_pool}

    return Dataset(
        version=version,
        fingerprint=fingerprint,
        controls={c.control_id: c for c in controls if c.control_id in used_control_ids},
        evidence={e.evidence_id: e for e in evidence if e.evidence_id in used_evidence_ids},
        cases=selected_cases,
        cases_by_id={c.case_id: c for c in selected_cases},
    )
