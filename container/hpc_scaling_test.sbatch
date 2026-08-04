#!/bin/bash
#SBATCH --job-name=pascal_scaling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=results/pascal_scaling_%j.log

# Single job - normally submitted via submit_scaling_sweep.sh (a sweep of
# many jobs across core counts, using the small synthetic 1D scenario by
# default) or submit_advection_run.sh (a single real-scale run against a
# local CMEMS file - see that script and
# container/download_cmems_data.py). Can also be submitted directly with
# `sbatch --cpus-per-task=N --export=MODE=...,...` for a one-off run.
#
# Single-node only: coupler_parallel.py has no MPI/multi-node
# implementation. The scaling sweep's original purpose was checking
# whether multiprocessing.Pool's economics (a net LOSS - 22-27% slower -
# up to 1500 super-individuals on an 8-core laptop; see BENCHMARKING.md's
# Phase 2) change at real HPC node core counts; see BENCHMARKING.md for
# why a multi-node test would be separate, larger work if that turns out
# to matter.

set -euo pipefail

MODE="${MODE:-sequential}"
N_SUPER="${N_SUPER:-2000}"
DURATION="${DURATION:-0.3}"
SCENARIO="${SCENARIO:-1d}"
SEED="${SEED:-0}"
REPO_ROOT="${REPO_ROOT:?set REPO_ROOT to the pascal_classparallel_branch checkout path}"
OPENDRIFT_ROOT="${OPENDRIFT_ROOT:?set OPENDRIFT_ROOT to the opendrift_pascal checkout path}"
SIF="${SIF:-$REPO_ROOT/container/pascal.sif}"
# Only used by SCENARIO=advection_cmems_file - see
# container/download_cmems_data.py (must be downloaded ahead of time, on
# a node with internet access, before this job runs).
CMEMS_FILE="${CMEMS_FILE:-}"
START_LON="${START_LON:-}"
START_LAT="${START_LAT:-}"
START_DATE="${START_DATE:-}"
# If set, the model's actual output (output_ps.nc, lifestats.csv - not
# just this job's wall_time_s summary line) is kept under
# results/$RUN_OUTPUT_SUBDIR/job<id>/ instead of an ephemeral /tmp dir
# inside the container that disappears (or may live on node-local scratch
# never seen again) once the job ends. The scaling sweep leaves this
# unset deliberately - it only cares about the timing line, and 60-odd
# sweep jobs' worth of full model output would just be clutter.
RUN_OUTPUT_SUBDIR="${RUN_OUTPUT_SUBDIR:-}"

mkdir -p "$REPO_ROOT/results"
RESULT_FILE="$REPO_ROOT/results/scaling_cpus${SLURM_CPUS_PER_TASK}_${MODE}_job${SLURM_JOB_ID}.txt"

echo "job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK mode=$MODE " \
     "scenario=$SCENARIO n_super=$N_SUPER duration=$DURATION seed=$SEED" \
     | tee "$RESULT_FILE"

extra_args=()
if [ -n "$CMEMS_FILE" ]; then
    extra_args+=(--cmems-file "$CMEMS_FILE")
fi
if [ -n "$START_LON" ]; then
    extra_args+=(--start-lon "$START_LON")
fi
if [ -n "$START_LAT" ]; then
    extra_args+=(--start-lat "$START_LAT")
fi
if [ -n "$START_DATE" ]; then
    extra_args+=(--start-date "$START_DATE")
fi
if [ -n "$RUN_OUTPUT_SUBDIR" ]; then
    # Host-side path (for mkdir, run outside the container) and
    # container-side path (for --workdir, seen by the process running
    # inside the container) both point at the same bind-mounted
    # location - REPO_ROOT is bind-mounted to /pascal below.
    HOST_OUTPUT_DIR="$REPO_ROOT/results/$RUN_OUTPUT_SUBDIR/job${SLURM_JOB_ID}"
    mkdir -p "$HOST_OUTPUT_DIR"
    extra_args+=(--workdir "/pascal/results/$RUN_OUTPUT_SUBDIR/job${SLURM_JOB_ID}")
    echo "model output will be written under $HOST_OUTPUT_DIR" | tee -a "$RESULT_FILE"
fi

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
        "${extra_args[@]}" \
    | tee -a "$RESULT_FILE"

echo "---" >> "$RESULT_FILE"
echo "sacct memory (once accounting has flushed - may be empty here):" >> "$RESULT_FILE"
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,NCPUS >> "$RESULT_FILE" 2>&1 || true
