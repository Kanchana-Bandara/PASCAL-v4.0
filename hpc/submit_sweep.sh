#!/bin/bash
# Submit the HPC scaling sweep: one Slurm job per (core count, mode)
# combination, via submit.sh (which just fills in env vars and `sbatch`s
# submit.sbatch). Sequential mode is core-count-independent so it's only
# submitted once (as the cores=1 baseline every parallel job gets
# compared against).
#
# Usage (run on the HPC login/submit node, from anywhere):
#   REPO_ROOT=/path/to/pascal_classparallel_branch \
#   OPENDRIFT_ROOT=/path/to/opendrift_pascal \
#   ./submit_sweep.sh
#
# Optional overrides: CONFIG (default: the cheap synthetic 1D scenario -
# see config/runs/scaling_sweep_1d.yaml), CORE_COUNTS (space-separated),
# OVERRIDES (extra --override tokens forwarded to every job, e.g. to
# change population.n_super_individuals/time.duration_years).
#
# Requires hpc/pascal.sif already built - see container.def's %help for
# the build command.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:?set REPO_ROOT to the pascal_classparallel_branch checkout path}"
OPENDRIFT_ROOT="${OPENDRIFT_ROOT:?set OPENDRIFT_ROOT to the opendrift_pascal checkout path}"
CONFIG="${CONFIG:-config/runs/scaling_sweep_1d.yaml}"
CORE_COUNTS="${CORE_COUNTS:-1 2 4 8 16 32 64}"
OVERRIDES="${OVERRIDES:-}"

SUBMIT="$(dirname "$0")/submit.sh"

# Sequential baseline (cores=1 point on the scaling curve) - submitted once.
REPO_ROOT="$REPO_ROOT" OPENDRIFT_ROOT="$OPENDRIFT_ROOT" CONFIG="$CONFIG" \
    MODE=sequential CPUS=1 JOB_NAME=pascal_scaling_sequential OVERRIDES="$OVERRIDES" \
    "$SUBMIT"

for cores in $CORE_COUNTS; do
    REPO_ROOT="$REPO_ROOT" OPENDRIFT_ROOT="$OPENDRIFT_ROOT" CONFIG="$CONFIG" \
        MODE=parallel CPUS="$cores" JOB_NAME="pascal_scaling_parallel_${cores}" OVERRIDES="$OVERRIDES" \
        "$SUBMIT"
done

echo "Submitted. Once all jobs finish, summarize with:"
echo "  python $REPO_ROOT/benchmarks/summarize_scaling_results.py $REPO_ROOT/results"
