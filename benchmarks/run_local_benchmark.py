#!/usr/bin/env python3
"""Local benchmark/profiling harness for the PASCAL simulation.

Runs a synthetic scenario (scenarios/builders.py) at a configurable size,
sequential or parallel, optionally under cProfile.

Usage:
    # Phase 1 baseline: sequential, single water column, cheap to run
    python benchmarks/run_local_benchmark.py --scenario 1d --n-super 200 \\
        --duration 0.5 --profile --profile-out profile.stats
    python benchmarks/view_profile.py profile.stats

    # Phase 2: advection scenario (one tracker element per super-individual,
    # so environment_profiles arrays actually have shape
    # (n_depth, n_super) - this is what coupler_parallel.py's per-individual
    # slicing exists for) sequential vs. parallel
    python benchmarks/run_local_benchmark.py --scenario advection \\
        --n-super 200 --duration 0.5 --mode sequential
    python benchmarks/run_local_benchmark.py --scenario advection \\
        --n-super 200 --duration 0.5 --mode parallel --n-workers 4

    # Phase 8: real 3D run against a local CMEMS file (see
    # hpc/download_cmems_data.py to produce one first)
    python benchmarks/run_local_benchmark.py --scenario advection_cmems_file \\
        --cmems-file /path/to/barents_2022_2024.nc --n-super 10000 \\
        --duration 2 --start-date 2022-01-01 --mode sequential
"""

import argparse
import cProfile
import datetime as dt
import os
import pstats
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# See tests/conftest.py for why: the repo tracks an uninitialized
# `opendrift` git submodule (empty dir) at its root, which can shadow the
# real installed opendrift package as a broken namespace package if the
# repo root ends up on sys.path.
for _bad in ("", str(REPO_ROOT)):
    while _bad in sys.path:
        sys.path.remove(_bad)

SCENARIOS_DIR = REPO_ROOT / "scenarios"
if str(SCENARIOS_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIOS_DIR))

from builders import (  # noqa: E402
    build_1d_scenario,
    build_advection_scenario,
    build_cmems_advection_scenario_from_file,
)


def build_sim(scenario_name, mode, n_super, n_virtual, duration, seed, n_workers,
              rng_seed, cmems_file=None, start_lon=None, start_lat=None,
              start_date=None):
    if scenario_name == "1d":
        kwargs = build_1d_scenario(
            n_super_individuals=n_super,
            n_virtual_per_super=n_virtual,
            duration_years=duration,
            seed=seed,
            headless="bench_run",
        )
        from pascal.coupler import Pascal1D
        from pascal.coupler_parallel import Pascal1DParallel
        sequential_cls, parallel_cls = Pascal1D, Pascal1DParallel
    elif scenario_name == "advection":
        kwargs = build_advection_scenario(
            n_super_individuals=n_super,
            n_virtual_per_super=n_virtual,
            duration_years=duration,
            seed=seed,
            headless="bench_run",
        )
        from pascal.coupler import PascalAdvection
        from pascal.coupler_parallel import PascalAdvectionParallel
        sequential_cls, parallel_cls = PascalAdvection, PascalAdvectionParallel
    elif scenario_name == "advection_cmems_file":
        if cmems_file is None:
            raise ValueError(
                "--cmems-file is required for --scenario advection_cmems_file"
                " (see hpc/download_cmems_data.py)"
            )
        scenario_kwargs = {}
        if start_lon is not None and start_lat is not None:
            scenario_kwargs["start_location"] = (start_lon, start_lat)
        if start_date is not None:
            scenario_kwargs["start_date"] = start_date
        kwargs = build_cmems_advection_scenario_from_file(
            cmems_file,
            n_super_individuals=n_super,
            n_virtual_per_super=n_virtual,
            duration_years=duration,
            seed=seed,
            headless="bench_run",
            **scenario_kwargs,
        )
        from pascal.coupler import PascalAdvection
        from pascal.coupler_parallel import PascalAdvectionParallel
        sequential_cls, parallel_cls = PascalAdvection, PascalAdvectionParallel
    else:
        raise ValueError(scenario_name)

    if mode == "sequential":
        return sequential_cls(**kwargs)
    elif mode == "parallel":
        return parallel_cls(
            **kwargs, use_parallel=True, n_workers=n_workers, rng_seed=rng_seed
        )
    else:
        raise ValueError(mode)


def run_once(scenario_name, mode, n_super, n_virtual, duration, seed, n_workers,
             rng_seed, workdir, profile_out=None, cmems_file=None, start_lon=None,
             start_lat=None, start_date=None):
    os.chdir(workdir)
    sim = build_sim(
        scenario_name, mode, n_super, n_virtual, duration, seed, n_workers, rng_seed,
        cmems_file=cmems_file, start_lon=start_lon, start_lat=start_lat,
        start_date=start_date,
    )

    if profile_out:
        profiler = cProfile.Profile()
        start = time.perf_counter()
        profiler.enable()
        sim.run()
        profiler.disable()
        elapsed = time.perf_counter() - start
        profiler.dump_stats(profile_out)
    else:
        start = time.perf_counter()
        sim.run()
        elapsed = time.perf_counter() - start

    return elapsed, sim


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scenario",
        choices=["1d", "advection", "advection_cmems_file"],
        default="1d",
    )
    parser.add_argument("--mode", choices=["sequential", "parallel"],
                         default="sequential")
    parser.add_argument("--n-super", type=int, default=200)
    parser.add_argument("--n-virtual", type=int, default=10000)
    parser.add_argument("--duration", type=float, default=0.5, help="years")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--rng-seed", type=int, default=None,
                         help="Reproducible per-worker RNG seed (parallel mode only)")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-out", default="profile.stats")
    parser.add_argument(
        "--cmems-file",
        default=None,
        help="Path to a local netCDF produced by "
             "hpc/download_cmems_data.py - required for "
             "--scenario advection_cmems_file",
    )
    parser.add_argument("--start-lon", type=float, default=None)
    parser.add_argument("--start-lat", type=float, default=None)
    parser.add_argument(
        "--start-date",
        default=None,
        help="ISO date, e.g. 2022-01-01 (advection_cmems_file only; must fall"
             " within the downloaded file's time range)",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Directory to run in (default: a fresh temp dir)",
    )
    args = parser.parse_args()

    workdir = args.workdir or tempfile.mkdtemp(prefix="pascal_bench_")
    profile_out = os.path.abspath(args.profile_out) if args.profile else None
    start_date = (
        dt.datetime.fromisoformat(args.start_date) if args.start_date else None
    )

    print(f"scenario={args.scenario} mode={args.mode} n_super={args.n_super} "
          f"n_virtual={args.n_virtual} duration={args.duration}y seed={args.seed} "
          f"n_workers={args.n_workers}")
    print(f"workdir={workdir}")

    elapsed, sim = run_once(
        args.scenario, args.mode, args.n_super, args.n_virtual, args.duration,
        args.seed, args.n_workers, args.rng_seed, workdir, profile_out=profile_out,
        cmems_file=args.cmems_file, start_lon=args.start_lon,
        start_lat=args.start_lat, start_date=start_date,
    )

    n_steps = len(sim.all_steps)
    print(f"\nwall_time_s={elapsed:.3f}")
    print(f"n_timesteps={n_steps}")
    print(f"ms_per_timestep={1000 * elapsed / n_steps:.3f}")
    print(f"final_population_size={sim.population_size()}")

    if profile_out:
        print(f"\nprofile written to {profile_out}")
        stats = pstats.Stats(profile_out)
        stats.strip_dirs().sort_stats("cumulative")
        print("\nTop 20 by cumulative time:")
        stats.print_stats(20)


if __name__ == "__main__":
    main()
