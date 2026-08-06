#!/usr/bin/env python3
"""Splits PascalSimulation.run()'s per-timestep wall time into two buckets:

  tracker_time - update_environment() alone (opendrift's tracker.run_1step())
  ibm_time     - everything else in the loop (sync_environment_references,
                 update_lifestage, log_spatial, gene_hunt, clean_dead, respawn)

Answers the question raised in prompt_debug.txt: is opendrift's own
single-process stepping actually the bottleneck at HPC-scale
super-individual counts, in a way that would justify parallelising the
particle tracker itself? See BENCHMARKING.md's "Benchmarking OpenDrift
stepping vs. IBM stepping" section for the measured answer (no, at every
scale tested) and why.

Usage:
    python benchmarks/tracker_vs_ibm_benchmark.py --n-super 200 1000 5000 10000
    python benchmarks/tracker_vs_ibm_benchmark.py --reader cmems_file \\
        --cmems-file /path/to/barents_2022_2024.nc --n-super 200 1000 5000 10000
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# See tests/conftest.py for why: the repo tracks an uninitialized `opendrift`
# git submodule (empty dir) at its root, which can shadow the real installed
# opendrift package as a broken namespace package if the repo root ends up
# on sys.path.
for _bad in ("", str(REPO_ROOT)):
    while _bad in sys.path:
        sys.path.remove(_bad)

SCENARIOS_DIR = REPO_ROOT / "scenarios"
if str(SCENARIOS_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIOS_DIR))

import numpy as np  # noqa: E402

from pascal.coupler import PascalAdvection  # noqa: E402
from builders import (  # noqa: E402
    build_advection_scenario,
    build_cmems_advection_scenario_from_file,
)


def run_split(n_super, duration_years, reader, cmems_file, seed=0, seeding_rate=None,
              headless_dir=None):
    seeding_rate = seeding_rate if seeding_rate is not None else max(1, n_super // 40)
    if reader == "constant":
        kwargs = build_advection_scenario(
            n_super_individuals=n_super, n_virtual_per_super=10000,
            duration_years=duration_years, seed=seed, seeding_rate=seeding_rate,
            headless=f"tracker_vs_ibm_{n_super}",
        )
    else:
        kwargs = build_cmems_advection_scenario_from_file(
            cmems_file,
            n_super_individuals=n_super, n_virtual_per_super=10000,
            duration_years=duration_years, seed=seed, seeding_rate=seeding_rate,
            headless=f"tracker_vs_ibm_{n_super}",
        )

    if headless_dir is not None:
        os.chdir(headless_dir)

    sim = PascalAdvection(**kwargs)

    tracker_time = 0.0
    ibm_time = 0.0

    seed_locations_ind = np.random.choice(
        len(sim.start_locations), size=sim.seeding_rate, replace=True
    )
    seed_locations = [sim.start_locations[i] for i in seed_locations_ind]
    sim.seed(sim.seeding_rate, seed_locations, genome=None)

    for _ in sim.all_steps:
        t0 = time.perf_counter()
        sim.update_environment()
        t1 = time.perf_counter()
        tracker_time += t1 - t0

        sim.sync_environment_references()
        for _isplit in np.arange(0, sim.isplit):
            sim.update_lifestage()
        sim.log_spatial()
        sim.gene_hunt()
        sim.clean_dead()
        sim.respawn()
        sim.current_time += sim.timestep
        t2 = time.perf_counter()
        ibm_time += t2 - t1

    sim.finish_run()
    return tracker_time, ibm_time, sim.population_size(), len(sim.active_supindividuals())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-super", type=int, nargs="+", default=[200, 1000, 5000, 10000])
    p.add_argument("--duration", type=float, default=0.02)
    p.add_argument("--reader", choices=["constant", "cmems_file"], default="constant")
    p.add_argument("--cmems-file", type=str, default=None,
                    help="required if --reader cmems_file")
    p.add_argument("--workdir", type=str, default=None,
                    help="directory to write run output under (default: cwd)")
    args = p.parse_args()

    if args.reader == "cmems_file" and not args.cmems_file:
        p.error("--reader cmems_file requires --cmems-file")

    print(f"{'n_super':>8} {'tracker_s':>10} {'ibm_s':>10} {'tracker_%':>10} "
          f"{'pop':>14} {'n_active':>9}")
    for n in args.n_super:
        tt, it, pop, nact = run_split(
            n, args.duration, args.reader, args.cmems_file,
            headless_dir=args.workdir,
        )
        pct = 100 * tt / (tt + it)
        print(f"{n:>8} {tt:>10.3f} {it:>10.3f} {pct:>9.1f}% {pop:>14.1f} {nact:>9}")


if __name__ == "__main__":
    main()
