#!/bin/bash
#SBATCH --job-name=pascal_scaling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=results/pascal_scaling_%j.log

# Single job in the HPC scaling sweep - normally submitted via
# submit_scaling_sweep.sh, which sets --cpus-per-task and --export per
# (core count, mode) combination. Can also be submitted directly with
# `sbatch --cpus-per-task=N --export=MODE=...,...` for a one-off run.
#
# Single-node only: this tests whether multiprocessing.Pool's economics
# (found to be a net LOSS - 22-27% slower - up to 1500 super-individuals
# on an 8-core laptop; see BENCHMARKING.md's Phase 2) change at real HPC
# node core counts (a node with 32-128 cores gives workers much more to
# do per dispatch, which is the dimension Phase 2 couldn't test). This is
# NOT a multi-node test - coupler_parallel.py has no MPI/multi-node
# implementation; see BENCHMARKING.md for why that would be a separate,
# larger effort if this sweep shows single-node HPC economics still don't
# favor multiprocessing.

set -euo pipefail

MODE="${MODE:-sequential}"
N_SUPER="${N_SUPER:-2000}"
DURATION="${DURATION:-0.3}"
SCENARIO="${SCENARIO:-1d}"
SEED="${SEED:-0}"
REPO_ROOT="${REPO_ROOT:?set REPO_ROOT to the pascal_classparallel_branch checkout path}"
OPENDRIFT_ROOT="${OPENDRIFT_ROOT:?set OPENDRIFT_ROOT to the opendrift_pascal checkout path}"
SIF="${SIF:-$REPO_ROOT/container/pascal.sif}"

mkdir -p "$REPO_ROOT/results"
RESULT_FILE="$REPO_ROOT/results/scaling_cpus${SLURM_CPUS_PER_TASK}_${MODE}_job${SLURM_JOB_ID}.txt"

echo "job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK mode=$MODE " \
     "scenario=$SCENARIO n_super=$N_SUPER duration=$DURATION seed=$SEED" \
     | tee "$RESULT_FILE"

apptainer exec --writable-tmpfs \
    --bind "$REPO_ROOT":/pascal \
    --bind "$OPENDRIFT_ROOT":/opendrift \
    "$SIF" \
    /pascal/container/run_in_container.sh \
    python /pascal/benchmarks/run_local_benchmark.py \
        --scenario "$SCENARIO" \
        --mode "$MODE" \
        --n-super "$N_SUPER" \
        --duration "$DURATION" \
        --n-workers "$SLURM_CPUS_PER_TASK" \
        --seed "$SEED" \
    | tee -a "$RESULT_FILE"

echo "---" >> "$RESULT_FILE"
echo "sacct memory (once accounting has flushed - may be empty here):" >> "$RESULT_FILE"
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,NCPUS >> "$RESULT_FILE" 2>&1 || true
