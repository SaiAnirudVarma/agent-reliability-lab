#!/usr/bin/env python3
"""Check a candidate evaluation artifact against a regression manifest.

Usage:
    python scripts/run_regression_gate.py \\
        --manifest regression/regression_manifest_v1.json \\
        --candidate results/reranked_llm_experiments/synthetic-v2__reranked-llm-baseline-v1__12457851-fd75-4f45-8e58-79d95c3e13a1.json

Purely offline and deterministic: reads two JSON files, compares
already-computed EvaluationResult fields against already-frozen
manifest floors (app.regression.gate), and prints a concise
human-readable report. No provider call, no dataset load, no benchmark
or artifact mutation of any kind -- ``--candidate`` is read-only.

Exit codes:
    0  the gate passed (provenance OK, and every manifest case's floors held).
    1  the gate failed: a regression was detected, a manifest case is
       missing from the candidate, provenance did not match, or either
       input file could not be read/parsed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from app.regression.gate import CandidateLoadError, format_report, load_candidate_results, run_regression_gate
from app.regression.manifest import load_regression_manifest


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--manifest", type=Path, required=True,
        help="Path to a RegressionManifest JSON file (e.g. regression/regression_manifest_v1.json).",
    )
    parser.add_argument(
        "--candidate", type=Path, required=True,
        help="Path to a candidate evaluation artifact JSON file (any report with top-level "
        "'dataset_version' and 'results' fields, e.g. a RunReport or RerankedLLMReport). Read-only.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    try:
        manifest = load_regression_manifest(args.manifest)
    except OSError as exc:
        print(f"ERROR: could not read manifest file {args.manifest}: {exc}", file=sys.stderr)
        return 1
    except ValidationError as exc:
        print(f"ERROR: {args.manifest} is not a valid RegressionManifest: {exc}", file=sys.stderr)
        return 1

    try:
        dataset_version, results_by_case_id = load_candidate_results(args.candidate)
    except OSError as exc:
        print(f"ERROR: could not read candidate file {args.candidate}: {exc}", file=sys.stderr)
        return 1
    except CandidateLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    report = run_regression_gate(manifest, dataset_version, results_by_case_id)
    print(format_report(report))

    return 0 if report.gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
