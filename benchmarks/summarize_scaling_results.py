#!/usr/bin/env python3
"""Summarize an HPC scaling sweep's results into a speedup/efficiency table.

Reads hpc/submit.sbatch's output files (one per job,
named scaling_cpus<N>_<mode>_job<id>.txt, containing run_local_benchmark.py's
stdout including a `wall_time_s=...` line) and prints a table comparing
each parallel job's wall time against the single sequential baseline.

Usage:
    python benchmarks/summarize_scaling_results.py /path/to/results/
"""

import argparse
import re
import sys
from pathlib import Path

WALL_TIME_RE = re.compile(r"^wall_time_s=([\d.]+)")
FILENAME_RE = re.compile(r"scaling_cpus(\d+)_(sequential|parallel)_job(\d+)\.txt$")


def parse_result_file(path):
    cpus_from_name, mode_from_name, job_id = FILENAME_RE.match(path.name).groups()
    wall_time_s = None
    for line in path.read_text().splitlines():
        m = WALL_TIME_RE.match(line.strip())
        if m:
            wall_time_s = float(m.group(1))
            break
    if wall_time_s is None:
        return None
    return {
        "cpus": int(cpus_from_name),
        "mode": mode_from_name,
        "job_id": job_id,
        "wall_time_s": wall_time_s,
        "file": path.name,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()

    files = sorted(args.results_dir.glob("scaling_cpus*_*_job*.txt"))
    if not files:
        print(f"No result files matching scaling_cpus*_*_job*.txt found in "
              f"{args.results_dir}", file=sys.stderr)
        sys.exit(1)

    results = [r for r in (parse_result_file(f) for f in files) if r is not None]
    skipped = len(files) - len(results)
    if skipped:
        print(f"warning: {skipped} file(s) had no wall_time_s line (job may have "
              f"failed) - skipped", file=sys.stderr)

    sequential = [r for r in results if r["mode"] == "sequential"]
    if not sequential:
        print("No completed sequential baseline job found - cannot compute "
              "speedup/efficiency, printing raw wall times only.\n", file=sys.stderr)
        baseline = None
    else:
        if len(sequential) > 1:
            print(f"warning: {len(sequential)} sequential result files found, "
                  f"using the fastest as baseline", file=sys.stderr)
        baseline = min(r["wall_time_s"] for r in sequential)

    parallel = sorted(
        (r for r in results if r["mode"] == "parallel"), key=lambda r: r["cpus"]
    )

    header = f"{'cpus':>6} {'mode':>10} {'wall_time_s':>12}"
    if baseline is not None:
        header += f" {'speedup':>9} {'efficiency':>11}"
    print(header)
    print("-" * len(header))

    if baseline is not None:
        print(f"{1:>6} {'sequential':>10} {baseline:>12.2f} {1.0:>9.2f} {100.0:>10.1f}%")

    for r in parallel:
        row = f"{r['cpus']:>6} {'parallel':>10} {r['wall_time_s']:>12.2f}"
        if baseline is not None:
            speedup = baseline / r["wall_time_s"]
            efficiency = 100.0 * speedup / r["cpus"]
            row += f" {speedup:>9.2f} {efficiency:>10.1f}%"
        print(row)

    if baseline is not None and parallel:
        best = max(parallel, key=lambda r: baseline / r["wall_time_s"])
        best_speedup = baseline / best["wall_time_s"]
        print()
        if best_speedup > 1.0:
            print(f"Best: {best['cpus']} cpus, {best_speedup:.2f}x speedup over "
                  f"sequential.")
        else:
            print(f"No configuration tested beat the sequential baseline "
                  f"(best was {best['cpus']} cpus at {best_speedup:.2f}x) - "
                  f"consistent with the laptop-scale finding in "
                  f"BENCHMARKING.md's Phase 2, if so.")


if __name__ == "__main__":
    main()
