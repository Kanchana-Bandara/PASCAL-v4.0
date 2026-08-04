#!/usr/bin/env python3
"""Local benchmark/profiling harness for the sequential PASCAL simulation.

This runs the synthetic Pascal1D scenario (benchmarks/scenario.py) at a
configurable size and reports wall-clock time, optionally under cProfile.
It is deliberately sequential-only for now: Phase 1 is about establishing
where time actually goes before Phase 2 changes anything about how the
super-individual update is parallelized.

Usage:
    python benchmarks/run_local_benchmark.py --n-super 200 --duration 0.5
    python benchmarks/run_local_benchmark.py --n-super 200 --duration 0.5 \\
        --profile --profile-out profile.stats
    python view_profile.py profile.stats
"""

import argparse
import cProfile
import os
import pstats
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenario import build_1d_scenario  # noqa: E402
from coupler import Pascal1D  # noqa: E402


def run_once(n_super, n_virtual, duration, seed, workdir, profile_out=None):
    os.chdir(workdir)
    kwargs = build_1d_scenario(
        n_super_individuals=n_super,
        n_virtual_per_super=n_virtual,
        duration_years=duration,
        seed=seed,
        headless="bench_run",
    )
    sim = Pascal1D(**kwargs)

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-super", type=int, default=200)
    parser.add_argument("--n-virtual", type=int, default=10000)
    parser.add_argument("--duration", type=float, default=0.5, help="years")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-out", default="profile.stats")
    parser.add_argument(
        "--workdir",
        default=None,
        help="Directory to run in (default: a fresh temp dir, deleted-on-reboot)",
    )
    args = parser.parse_args()

    workdir = args.workdir or tempfile.mkdtemp(prefix="pascal_bench_")
    profile_out = None
    if args.profile:
        profile_out = os.path.abspath(args.profile_out)

    print(f"n_super={args.n_super} n_virtual={args.n_virtual} "
          f"duration={args.duration}y seed={args.seed}")
    print(f"workdir={workdir}")

    elapsed, sim = run_once(
        args.n_super, args.n_virtual, args.duration, args.seed, workdir,
        profile_out=profile_out,
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
