"""Integration test for scripts/run_eval.py: invokes it as a real subprocess
against the actual shipped dataset and inspects both its console output and
the result file it produces.

HERMETIC CONFIGURATION, BY CONSTRUCTION
----------------------------------------
Every subprocess launched from this file uses `_clean_env()`, which:

1. Strips LLM-relevant variables from the inherited environment and points
   `ARL_DOTENV_PATH` at a path that is guaranteed not to exist, so this
   repository's real `.env` (and whatever real credentials it holds) can
   NEVER reach a subprocess started from this file -- see
   `scripts/run_eval.py`'s docstring for how that override works.
2. Points `ARL_RESULTS_DIR` at a pytest-managed temporary directory
   (never the repository's real `results/`), so NO test here can read,
   write, or delete anything under the real `results/` directory. This is
   the direct fix for
   `docs/incidents/2026-09-23-llm-baseline-artifact-deletion.md`, where a
   test that hardcoded the production results path deleted the one real
   LLM experiment artifact that existed. `tests/integration/conftest.py`
   additionally hashes the real `results/` directory before and after this
   entire test session and fails loudly if anything in it changed, as a
   second, independent guard against this class of regression.

A subprocess that reaches `_build_llm_agent()` with real-looking
credentials would still, if invoked, attempt a genuine network call --
`tests/conftest.py`'s socket barrier does NOT protect a child process (see
that module's docstring) -- so the only thing standing between "missing
config test" and "accidental live API call" is the hermetic-environment
guarantee in (1). No test in this file constructs real-looking OpenAI
credentials in a subprocess that also calls `main()` with `--agent llm`.

Any future genuinely-live API test must not live in this file, must not run
under plain `pytest`, and must require its own explicit opt-in — none
exists yet, and this file must never be extended to add one silently.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.models.contracts import RunReport

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_eval.py"

# Guaranteed not to exist: nested under a directory that also doesn't
# exist, so it can never accidentally collide with a real file anywhere.
NONEXISTENT_DOTENV_PATH = str(REPO_ROOT / ".pytest-isolation-nonexistent-dir" / ".env")


def _clean_env(results_dir: Path, **overrides: str) -> dict:
    """The base environment for every CLI subprocess launched from this
    file: the real shell environment with LLM-relevant vars stripped,
    ARL_DOTENV_PATH forced to a nonexistent path (so this repo's real .env
    can never be loaded), and ARL_RESULTS_DIR forced to the caller-supplied
    ``results_dir`` (so this repo's real results/ directory can never be
    read, written, or deleted). ``results_dir`` is REQUIRED, not defaulted,
    so every call site must consciously supply an isolated directory --
    normally a pytest ``tmp_path``.

    Pass e.g. ``MODEL_PROVIDER="openai"`` as a keyword to set exactly the
    variable(s) an individual test cares about. ``ARL_DOTENV_PATH`` and
    ``ARL_RESULTS_DIR`` themselves can also be overridden this way by tests
    that specifically test those mechanisms (see ``TestDotenvIsolation``).
    """

    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"AGENT_MODE", "MODEL_PROVIDER", "MODEL_NAME", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}
    }
    env["ARL_DOTENV_PATH"] = NONEXISTENT_DOTENV_PATH
    env["ARL_RESULTS_DIR"] = str(results_dir)
    env.update(overrides)
    return env


def _inspect_env_after_import(env: dict) -> dict:
    """Execute scripts/run_eval.py's MODULE-LEVEL code (which includes the
    dotenv-loading side effect) via runpy with run_name != "__main__", so
    the `if __name__ == "__main__": sys.exit(main())` guard at the bottom
    of that file never fires and main() is never called. This makes it
    IMPOSSIBLE for this helper to reach a network-call path OR touch any
    results directory, no matter what credentials end up loaded. Returns
    booleans (and our own fake MODEL_NAME value, when we planted one) only
    -- OPENAI_API_KEY's actual value is never read back or asserted on,
    only its presence.
    """

    code = (
        "import runpy, os, json\n"
        f"runpy.run_path({str(SCRIPT_PATH)!r}, run_name='not_main')\n"
        "print(json.dumps({\n"
        "    'MODEL_PROVIDER_present': 'MODEL_PROVIDER' in os.environ,\n"
        "    'MODEL_NAME_present': 'MODEL_NAME' in os.environ,\n"
        "    'MODEL_NAME_value': os.environ.get('MODEL_NAME'),\n"
        "    'OPENAI_API_KEY_present': 'OPENAI_API_KEY' in os.environ,\n"
        "}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=15, env=env,
    )
    assert result.returncode == 0, f"import-only invocation failed: {result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def results_dir(tmp_path_factory) -> Path:
    """One isolated, pytest-managed results directory shared by the
    module-scoped `run_cli` fixture below -- never the repository's real
    results/ directory."""

    return tmp_path_factory.mktemp("results")


@pytest.fixture(scope="module")
def run_cli(results_dir: Path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        env=_clean_env(results_dir),
    )
    return result


class TestCli:
    def test_successful_invocation_exits_zero(self, run_cli):
        assert run_cli.returncode == 0, run_cli.stderr

    def test_output_identifies_expected_vs_actual(self, run_cli):
        assert "expected=" in run_cli.stdout
        assert "actual=" in run_cli.stdout

    def test_output_reports_all_ten_cases(self, run_cli):
        for i in range(1, 11):
            assert f"AC-{i:03d}" in run_cli.stdout

    def test_output_includes_metrics_section(self, run_cli):
        assert "Finding accuracy" in run_cli.stdout
        assert "Macro F1" in run_cli.stdout
        assert "Abstention F1" in run_cli.stdout

    def test_output_reports_run_id_and_results_path(self, run_cli):
        assert "Run ID:" in run_cli.stdout
        assert "baseline.json" in run_cli.stdout

    def test_baseline_json_is_created(self, run_cli, results_dir):
        assert (results_dir / "baseline.json").exists()

    def test_baseline_json_validates_as_run_report(self, run_cli, results_dir):
        report = RunReport.model_validate_json((results_dir / "baseline.json").read_text())
        assert len(report.results) == 10
        assert len(report.traces) == 10

    def test_run_id_is_shared_across_results_and_traces_on_disk(self, run_cli, results_dir):
        report = RunReport.model_validate_json((results_dir / "baseline.json").read_text())
        run_ids = {report.run_id} | {r.run_id for r in report.results} | {t.run_id for t in report.traces}
        assert run_ids == {report.run_id}

    def test_baseline_json_contains_no_env_or_secret_looking_keys(self, run_cli, results_dir):
        raw = (results_dir / "baseline.json").read_text()
        for forbidden in ("API_KEY", "SECRET", "PASSWORD", "TOKEN"):
            assert forbidden not in raw.upper() or forbidden == "TOKEN"
        # "TOKEN" legitimately appears as input_tokens/output_tokens field
        # names; explicitly check no actual secret-shaped value is present.
        assert "sk-" not in raw


class TestDotenvIsolation:
    """Proves the .env-loading mechanism itself, with zero risk of reaching
    a network-call path or a results directory (see
    _inspect_env_after_import's docstring: main() is never invoked by
    either test here).
    """

    def test_real_env_on_disk_cannot_leak_into_isolated_subprocess(self, tmp_path):
        """The core safety guarantee: even though this repository has a
        real .env with real credentials sitting on disk, a subprocess whose
        ARL_DOTENV_PATH points at a nonexistent path sees NONE of it. This
        is proven without ever reading the real .env's contents -- the
        proof is purely behavioral: nothing shows up that wasn't explicitly
        placed in this subprocess's environment.
        """
        observed = _inspect_env_after_import(_clean_env(tmp_path))
        assert observed["MODEL_PROVIDER_present"] is False
        assert observed["MODEL_NAME_present"] is False
        assert observed["OPENAI_API_KEY_present"] is False

    def test_dotenv_loads_values_from_an_explicitly_pointed_at_file(self, tmp_path):
        """The mechanism DOES work when deliberately pointed at a
        controlled file -- proves this isn't just "dotenv loading is
        broken/disabled," it's specifically isolated from the real .env."""
        fake_dotenv = tmp_path / "fake.env"
        fake_dotenv.write_text(
            "MODEL_PROVIDER=openai\n"
            "MODEL_NAME=fake-model-for-dotenv-mechanism-test\n"
            "OPENAI_API_KEY=sk-fake-dotenv-mechanism-test-only-never-real\n"
        )
        observed = _inspect_env_after_import(_clean_env(tmp_path, ARL_DOTENV_PATH=str(fake_dotenv)))
        assert observed["MODEL_PROVIDER_present"] is True
        assert observed["MODEL_NAME_value"] == "fake-model-for-dotenv-mechanism-test"
        assert observed["OPENAI_API_KEY_present"] is True  # presence only -- value never read back

    def test_shell_exported_value_takes_precedence_over_dotenv_value(self, tmp_path):
        """Requirement: existing real shell environment variables must take
        precedence over .env values."""
        fake_dotenv = tmp_path / "fake.env"
        fake_dotenv.write_text("MODEL_NAME=from-dotenv-should-be-overridden\n")
        env = _clean_env(tmp_path, ARL_DOTENV_PATH=str(fake_dotenv))
        env["MODEL_NAME"] = "from-shell-should-win"  # simulates an already-exported shell var
        observed = _inspect_env_after_import(env)
        assert observed["MODEL_NAME_value"] == "from-shell-should-win"


class TestCliLlmModeConfiguration:
    """These tests exercise the CLI's --agent llm wiring and config
    validation ONLY -- they never construct a real provider or spend API
    money. LLMAgent's actual success/failure behavior against a provider is
    covered at the unit level (tests/unit/test_llm_agent.py,
    tests/unit/test_openai_provider.py) via FakeLLMProvider / mocks. Here we
    only prove: the flag exists, deterministic mode is unaffected by its
    presence, and missing configuration fails fast and clearly rather than
    silently falling back to deterministic mode or crashing with a
    traceback -- and, critically, that this holds regardless of the real
    .env on disk (see _clean_env / NONEXISTENT_DOTENV_PATH above).

    Every test method here receives pytest's function-scoped `tmp_path`
    fixture, giving each one (and therefore each subprocess it launches)
    its own private results directory.
    """

    def _run(self, tmp_path: Path, args: list[str], env_overrides: dict) -> subprocess.CompletedProcess:
        env = _clean_env(tmp_path, **env_overrides)
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), *args],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, env=env,
        )

    def test_llm_mode_without_any_config_fails_clearly_not_silently(self, tmp_path):
        result = self._run(tmp_path, ["--agent", "llm"], {})
        assert result.returncode == 1
        assert "MODEL_PROVIDER" in result.stderr
        assert "MODEL_NAME" in result.stderr

    def test_llm_mode_with_provider_but_no_model_name_fails_clearly(self, tmp_path):
        result = self._run(tmp_path, ["--agent", "llm"], {"MODEL_PROVIDER": "openai"})
        assert result.returncode == 1
        assert "MODEL_NAME" in result.stderr

    def test_llm_mode_with_unknown_provider_fails_clearly(self, tmp_path):
        result = self._run(
            tmp_path, ["--agent", "llm"], {"MODEL_PROVIDER": "not-a-real-provider", "MODEL_NAME": "x"}
        )
        assert result.returncode == 1
        assert "not-a-real-provider" in result.stderr

    def test_llm_mode_with_full_config_but_no_api_key_fails_clearly(self, tmp_path):
        result = self._run(
            tmp_path, ["--agent", "llm"],
            {"MODEL_PROVIDER": "openai", "MODEL_NAME": "gpt-5.4-mini-2026-03-17"},
        )
        assert result.returncode == 1
        assert "OPENAI_API_KEY" in result.stderr

    def test_llm_mode_never_writes_to_baseline_json(self, tmp_path):
        """A failed/attempted LLM run must never touch the deterministic
        baseline's result file, in its own isolated results directory."""
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text('{"sentinel": true}')  # pre-existing, must survive untouched
        before = baseline_path.read_text()
        self._run(tmp_path, ["--agent", "llm"], {})
        after = baseline_path.read_text()
        assert before == after

    def test_llm_mode_does_not_silently_run_as_deterministic(self, tmp_path):
        """Presence of the --agent llm flag with no config must fail, not
        quietly fall back to the deterministic agent."""
        result = self._run(tmp_path, ["--agent", "llm"], {})
        assert "deterministic-baseline-v1" not in result.stdout

    def test_deterministic_mode_unaffected_by_llm_env_vars_being_set(self, tmp_path):
        """Setting MODEL_PROVIDER/MODEL_NAME/OPENAI_API_KEY must not cause
        the DEFAULT (no --agent flag) invocation to switch to LLM mode."""
        result = self._run(
            tmp_path, [],
            {
                "MODEL_PROVIDER": "openai",
                "MODEL_NAME": "gpt-5.4-mini-2026-03-17",
                "OPENAI_API_KEY": "sk-fake-for-this-test",
            },
        )
        assert result.returncode == 0
        assert "deterministic-baseline-v1" in result.stdout
        assert "baseline.json" in result.stdout


class TestLlmOverwriteProtection:
    """Section 4 (Phase 6 correction): a real LLM result is now written to a
    unique, run-ID-qualified path under results/experiments/ (see
    docs/ARTIFACT_POLICY.md), computed and checked for a collision BEFORE
    run_evaluation() is ever called -- so this class proves that check fires
    with zero network calls made, using ARL_RUN_ID (a test-only override,
    mirroring ARL_RESULTS_DIR/ARL_DOTENV_PATH) to make the otherwise-random
    UUID4 run ID -- and therefore the exact destination path -- predictable
    enough for a black-box subprocess test to pre-create a collision.
    """

    AGENT_CONFIG_ID = "llm-baseline-v1"  # LLMAgent.agent_config_id / PROMPT_VERSION

    def _experiment_path(self, tmp_path: Path, run_id: str, dataset_version: str = "synthetic-v1") -> Path:
        return tmp_path / "experiments" / f"{dataset_version}__{self.AGENT_CONFIG_ID}__{run_id}.json"

    def test_refuses_to_overwrite_existing_experiment_artifact_without_any_network_call(self, tmp_path):
        run_id = "fixed-test-run-id-for-collision-test"
        existing = self._experiment_path(tmp_path, run_id)
        existing.parent.mkdir(parents=True)
        existing.write_text('{"sentinel": "pre-existing real experiment result, must survive"}')
        env = _clean_env(
            tmp_path, MODEL_PROVIDER="openai", MODEL_NAME="gpt-5.4-mini-2026-03-17",
            OPENAI_API_KEY="sk-fake-for-this-test", ARL_RUN_ID=run_id, ARL_GIT_COMMIT_SHA="fake-sha-for-this-test",
        )
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--agent", "llm"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, env=env,
        )
        assert result.returncode == 1
        assert "already exists" in result.stderr
        # Untouched -- the CLI must refuse before doing anything else,
        # including before any real network call to a provider.
        assert existing.read_text() == '{"sentinel": "pre-existing real experiment result, must survive"}'

    def test_force_overwrite_does_not_bypass_experiment_immutability(self, tmp_path):
        """--force-overwrite no longer has any effect on a real LLM
        artifact -- see docs/ARTIFACT_POLICY.md Category A. This is the one
        behavior change from the flag's old semantics, verified directly."""
        run_id = "fixed-test-run-id-for-force-overwrite-test"
        existing = self._experiment_path(tmp_path, run_id)
        existing.parent.mkdir(parents=True)
        existing.write_text('{"sentinel": "must survive even with --force-overwrite"}')
        env = _clean_env(
            tmp_path, MODEL_PROVIDER="openai", MODEL_NAME="gpt-5.4-mini-2026-03-17",
            OPENAI_API_KEY="sk-fake-for-this-test", ARL_RUN_ID=run_id, ARL_GIT_COMMIT_SHA="fake-sha-for-this-test",
        )
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--agent", "llm", "--force-overwrite"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, env=env,
        )
        assert result.returncode == 1
        assert "already exists" in result.stderr
        assert existing.read_text() == '{"sentinel": "must survive even with --force-overwrite"}'

    def test_overwrite_check_and_config_validation_both_run_before_any_api_call(self, tmp_path):
        """Config validation (missing OPENAI_API_KEY) is reached first and
        reports its own specific error -- it happens before the new
        experiment-collision check even runs, since that check needs a
        constructed agent's agent_config_id. Both still run entirely before
        run_evaluation(), so neither ordering wastes a real API call."""
        existing = tmp_path / "llm-baseline.json"  # legacy path; no longer written to at all
        existing.write_text("{}")
        env = _clean_env(tmp_path, MODEL_PROVIDER="openai", MODEL_NAME="gpt-5.4-mini-2026-03-17")
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--agent", "llm"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, env=env,
        )
        assert result.returncode == 1
        assert "OPENAI_API_KEY" in result.stderr
        assert existing.read_text() == "{}"

    def test_deterministic_mode_always_overwrites_freely(self, tmp_path):
        """Deterministic output is reproducible on demand -- no protection,
        no --force-overwrite needed, matches docs/ARTIFACT_POLICY.md
        Category B."""
        existing = tmp_path / "baseline.json"
        existing.write_text('{"sentinel": "old deterministic run"}')
        env = _clean_env(tmp_path)
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, env=env,
        )
        assert result.returncode == 0
        assert "sentinel" not in existing.read_text()  # really overwritten with a real report


class TestCliCompare:
    def test_compare_without_llm_results_fails_clearly(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--compare"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, env=_clean_env(tmp_path),
        )
        assert result.returncode == 1
        assert "llm-baseline.json" in result.stderr

    def test_compare_with_both_files_present_prints_a_table(self, run_cli, results_dir, tmp_path):
        # run_cli (module-scoped) has already produced baseline.json in
        # results_dir. Fabricate a minimal, real (non-fake-money)
        # llm-baseline.json by copying the baseline report's shape into a
        # SEPARATE, per-test tmp_path (never results_dir, never the real
        # repo directory) -- this only tests the comparison table's
        # rendering, not LLMAgent itself.
        baseline_report = RunReport.model_validate_json((results_dir / "baseline.json").read_text())
        llm_like = baseline_report.model_copy(update={"run_id": "fake-llm-run-for-compare-test"})
        (tmp_path / "baseline.json").write_text(baseline_report.model_dump_json(indent=2))
        (tmp_path / "llm-baseline.json").write_text(llm_like.model_dump_json(indent=2))

        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--compare"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, env=_clean_env(tmp_path),
        )
        assert result.returncode == 0
        assert "Deterministic" in result.stdout
        assert "LLM Baseline" in result.stdout
        assert "Finding accuracy" in result.stdout


class TestRealResultsDirectoryNeverTouched:
    """Explicit, direct regression test (in addition to the session-wide
    guard in tests/integration/conftest.py): every CLI invocation in THIS
    test file must leave the repository's real results/ directory
    completely unchanged."""

    def test_real_results_dir_unchanged_after_a_full_cli_invocation(self, tmp_path):
        real_results_dir = REPO_ROOT / "results"
        before = {
            p.name: p.stat().st_mtime for p in real_results_dir.iterdir()
        } if real_results_dir.exists() else {}

        subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, env=_clean_env(tmp_path),
        )
        subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--agent", "llm"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, env=_clean_env(tmp_path),
        )

        after = {
            p.name: p.stat().st_mtime for p in real_results_dir.iterdir()
        } if real_results_dir.exists() else {}
        assert after == before
