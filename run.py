#!/usr/bin/env python3
"""Single entrypoint for running PASCAL from a YAML config (see config/runs/).

Replaces controlling a run via scattered CLI flags/env vars/hardcoded
scenario defaults - see config/schema.py for every field a config can set,
and config/runs/*.yaml for real examples.

Usage:
    python run.py --config config/runs/advection_quick_check.yaml

    # Override individual fields without editing the file - e.g. a quick
    # cheap sanity check of the real HPC config before submitting it:
    python run.py --config config/runs/advection_hpc_barents.yaml \\
        --override population.n_super_individuals=20 \\
        --override time.duration_years=0.02

    # Profile a run:
    python run.py --config config/runs/scaling_sweep_1d.yaml \\
        --profile --profile-out profile.stats
    python benchmarks/view_profile.py profile.stats
"""

import argparse
import cProfile
import os
import pstats
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# See tests/conftest.py for why: the repo tracks an uninitialized
# `opendrift` git submodule (empty dir) at its root, which can shadow the
# real installed opendrift package as a broken namespace package if the
# repo root ends up on sys.path.
for _bad in ("", str(REPO_ROOT)):
    while _bad in sys.path:
        sys.path.remove(_bad)

for _dir in (REPO_ROOT / "config", REPO_ROOT / "scenarios"):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

from loader import load_config  # noqa: E402
from builders import build_scenario_from_config  # noqa: E402


def build_sim(config):
    kwargs = build_scenario_from_config(config)
    scenario = config["run"]["scenario"]
    mode = config["run"]["mode"]

    if mode == "sequential":
        if scenario == "1d":
            from pascal.coupler import Pascal1D as sim_cls
        else:
            from pascal.coupler import PascalAdvection as sim_cls
        return sim_cls(**kwargs)
    elif mode == "parallel":
        if scenario == "1d":
            from pascal.coupler_parallel import Pascal1DParallel as sim_cls
        else:
            from pascal.coupler_parallel import PascalAdvectionParallel as sim_cls
        return sim_cls(
            **kwargs, use_parallel=True, n_workers=config["run"]["n_workers"]
        )
    else:
        raise ValueError(f"Unknown run.mode: {mode!r}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", required=True, help="Path to a YAML run config (see config/runs/)"
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="key.path=value",
        help="Override one config field, e.g. "
             "--override population.n_super_individuals=20. Repeatable.",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Directory to run in, and to write output_ps.nc/lifestats.csv "
             "under (default: a fresh temp dir)",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-out", default="profile.stats")
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.override)
    run = config["run"]

    workdir = args.workdir or tempfile.mkdtemp(prefix="pascal_run_")
    profile_out = os.path.abspath(args.profile_out) if args.profile else None

    print(
        f"run={run['name']} scenario={run['scenario']} mode={run['mode']} "
        f"n_super={config['population']['n_super_individuals']} "
        f"duration={config['time']['duration_years']}y seed={run['seed']}"
    )
    print(f"workdir={workdir}")

    os.chdir(workdir)
    sim = build_sim(config)

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
