"""Smoke test for run.py, the single config-driven entrypoint (see
config/runs/*.yaml and run.py's own docstring). Run as a real subprocess,
matching how it's actually invoked (CLI, not imported) - the interesting
behavior here is argument parsing/--override wiring, not anything worth
importing run.py's internals for.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args, workdir):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "run.py"), *args, "--workdir", str(workdir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_run_py_completes_with_overrides(tmp_path):
    result = run_cli(
        "--config", "config/runs/scaling_sweep_1d.yaml",
        "--override", "population.n_super_individuals=15",
        "--override", "time.duration_years=0.02",
        workdir=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert "final_population_size=" in result.stdout
    # Output lands under workdir/<run.name>/ - run.name is scaling_sweep_1d
    # yaml's own "scaling_sweep" (see PascalSimulation.outputfolder).
    output_dir = tmp_path / "scaling_sweep"
    assert (output_dir / "output_ps.nc").exists()
    assert (output_dir / "lifestats.csv").exists()


def test_run_py_rejects_incomplete_advection_cmems_file_config(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("run:\n  scenario: advection_cmems_file\n")

    result = run_cli("--config", str(config_path), workdir=tmp_path)

    assert result.returncode != 0
    assert "cmems_file" in result.stderr
