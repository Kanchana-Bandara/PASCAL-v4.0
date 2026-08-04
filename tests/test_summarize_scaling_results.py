"""Tests for benchmarks/summarize_scaling_results.py's parsing logic.

No Slurm/HPC access needed - just checks the result-file parser against
synthetic files matching container/hpc_scaling_test.sbatch's naming and
content convention.
"""

from summarize_scaling_results import parse_result_file


def write_result(tmp_path, cpus, mode, job_id, wall_time_s):
    path = tmp_path / f"scaling_cpus{cpus}_{mode}_job{job_id}.txt"
    path.write_text(
        f"job={job_id} cpus={cpus} mode={mode} scenario=1d n_super=2000 "
        f"duration=0.3 seed=0\n"
        f"scenario=1d mode={mode} n_super=2000 n_virtual=10000 "
        f"duration=0.3y seed=0 n_workers=None\n"
        f"wall_time_s={wall_time_s}\n"
        f"n_timesteps=219\n"
    )
    return path


def test_parse_result_file_extracts_cpus_mode_and_wall_time(tmp_path):
    path = write_result(tmp_path, cpus=4, mode="parallel", job_id=101, wall_time_s=40.5)
    result = parse_result_file(path)
    assert result == {
        "cpus": 4,
        "mode": "parallel",
        "job_id": "101",
        "wall_time_s": 40.5,
        "file": path.name,
    }


def test_parse_result_file_returns_none_if_wall_time_missing(tmp_path):
    path = tmp_path / "scaling_cpus4_parallel_job101.txt"
    path.write_text("job=101 cpus=4 mode=parallel\n(job crashed before finishing)\n")
    assert parse_result_file(path) is None
