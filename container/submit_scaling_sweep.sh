#!/bin/bash
# Submit the HPC scaling sweep: one Slurm job per (core count, mode)
# combination, via hpc_scaling_test.sbatch. Sequential mode is
# core-count-independent so it's only submitted once (as the cores=1
# baseline every parallel job gets compared against).
#
# Usage (run on the HPC login/submit node, from anywhere):
#   REPO_ROOT=/path/to/pascal_classparallel_branch \
#   OPENDRIFT_ROOT=/path/to/opendrift_pascal \
#   ./submit_scaling_sweep.sh
#
# Optional overrides: N_SUPER, DURATION, SCENARIO (1d|advection),
# CORE_COUNTS (space-separated), SEED.
#
# Requires container/pascal.sif already built - see container.def's
# %help for the build command.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:?set REPO_ROOT to the pascal_classparallel_branch checkout path}"
OPENDRIFT_ROOT="${OPENDRIFT_ROOT:?set OPENDRIFT_ROOT to the opendrift_pascal checkout path}"
N_SUPER="${N_SUPER:-2000}"
DURATION="${DURATION:-0.3}"
SCENARIO="${SCENARIO:-1d}"
SEED="${SEED:-0}"
CORE_COUNTS="${CORE_COUNTS:-1 2 4 8 16 32 64}"

if [ ! -f "$REPO_ROOT/container/pascal.sif" ]; then
    echo "submit_scaling_sweep.sh: $REPO_ROOT/container/pascal.sif not found." >&2
    echo "Build it first: apptainer build container/pascal.sif container/container.def" >&2
    exit 1
fi

mkdir -p "$REPO_ROOT/results"

# Sequential baseline (cores=1 point on the scaling curve) - submitted once.
sbatch \
    --chdir="$REPO_ROOT" \
    --cpus-per-task=1 \
    --job-name="pascal_scaling_sequential" \
    --export=ALL,MODE=sequential,N_SUPER="$N_SUPER",DURATION="$DURATION",SCENARIO="$SCENARIO",SEED="$SEED",REPO_ROOT="$REPO_ROOT",OPENDRIFT_ROOT="$OPENDRIFT_ROOT" \
    "$REPO_ROOT/container/hpc_scaling_test.sbatch"

for cores in $CORE_COUNTS; do
    sbatch \
        --chdir="$REPO_ROOT" \
        --cpus-per-task="$cores" \
        --job-name="pascal_scaling_parallel_${cores}" \
        --export=ALL,MODE=parallel,N_SUPER="$N_SUPER",DURATION="$DURATION",SCENARIO="$SCENARIO",SEED="$SEED",REPO_ROOT="$REPO_ROOT",OPENDRIFT_ROOT="$OPENDRIFT_ROOT" \
        "$REPO_ROOT/container/hpc_scaling_test.sbatch"
done

echo "Submitted. Once all jobs finish, summarize with:"
echo "  python $REPO_ROOT/benchmarks/summarize_scaling_results.py $REPO_ROOT/results"
