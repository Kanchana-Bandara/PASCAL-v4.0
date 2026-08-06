#!/bin/bash
#SBATCH --job-name=pascal_run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=results/pascal_run_%j.log

# Generic PASCAL job: runs run.py --config $CONFIG inside the container.
# Normally submitted via submit.sh (a single run) or submit_sweep.sh (many
# jobs across core counts) - both just fill in the env vars below and
# `sbatch` this file; see either script's header, or `sbatch
# --cpus-per-task=N --export=...` directly for a one-off run.
#
# Single-node only: coupler_parallel.py has no MPI/multi-node
# implementation. See BENCHMARKING.md's Phase 2 for why parallel mode's
# payoff (or lack of it) needs checking per-cluster before relying on it.

set -euo pipefail

CONFIG="${CONFIG:?set CONFIG to a YAML run config path relative to REPO_ROOT, e.g. config/runs/advection_hpc_barents.yaml}"
MODE="${MODE:-sequential}"
REPO_ROOT="${REPO_ROOT:?set REPO_ROOT to the pascal_classparallel_branch checkout path}"
OPENDRIFT_ROOT="${OPENDRIFT_ROOT:?set OPENDRIFT_ROOT to the opendrift_pascal checkout path}"
SIF="${SIF:-$REPO_ROOT/hpc/pascal.sif}"
# Extra --override key.path=value tokens (space-separated - see
# config/loader.py's apply_overrides()), for anything a submission needs
# to change beyond what CONFIG's file already says.
OVERRIDES="${OVERRIDES:-}"
# If set, the model's actual output (output_ps.nc, lifestats.csv - not
# just this job's wall_time_s summary line) is kept under
# results/$RUN_OUTPUT_SUBDIR/job<id>/ instead of an ephemeral /tmp dir
# inside the container that disappears (or may live on node-local scratch
# never seen again) once the job ends. submit_sweep.sh leaves this unset
# deliberately - it only cares about the timing line, and many sweep
# jobs' worth of full model output would just be clutter.
RUN_OUTPUT_SUBDIR="${RUN_OUTPUT_SUBDIR:-}"

mkdir -p "$REPO_ROOT/results"
RESULT_FILE="$REPO_ROOT/results/scaling_cpus${SLURM_CPUS_PER_TASK}_${MODE}_job${SLURM_JOB_ID}.txt"

echo "job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK mode=$MODE config=$CONFIG" \
     | tee "$RESULT_FILE"

# run.mode/run.n_workers always passed (matching this job's actual
# resource allocation) - config/schema.py's run.mode default is
# "sequential" and run.n_workers is only read in parallel mode, so this
# is a no-op unless MODE=parallel is actually requested.
override_args=(--override "run.mode=$MODE" --override "run.n_workers=$SLURM_CPUS_PER_TASK")
for tok in $OVERRIDES; do
    override_args+=(--override "$tok")
done

workdir_args=()
if [ -n "$RUN_OUTPUT_SUBDIR" ]; then
    # Host-side path (for mkdir, run outside the container) and
    # container-side path (for --workdir, seen by the process running
    # inside the container) both point at the same bind-mounted
    # location - REPO_ROOT is bind-mounted to /pascal below.
    HOST_OUTPUT_DIR="$REPO_ROOT/results/$RUN_OUTPUT_SUBDIR/job${SLURM_JOB_ID}"
    mkdir -p "$HOST_OUTPUT_DIR"
    workdir_args+=(--workdir "/pascal/results/$RUN_OUTPUT_SUBDIR/job${SLURM_JOB_ID}")
    echo "model output will be written under $HOST_OUTPUT_DIR" | tee -a "$RESULT_FILE"
fi

apptainer exec --writable-tmpfs \
    --bind "$REPO_ROOT":/pascal \
    --bind "$OPENDRIFT_ROOT":/opendrift \
    "$SIF" \
    /pascal/hpc/run_in_container.sh \
    python /pascal/run.py --config "/pascal/$CONFIG" \
        "${override_args[@]}" \
        "${workdir_args[@]}" \
    | tee -a "$RESULT_FILE"

echo "---" >> "$RESULT_FILE"
echo "sacct memory (once accounting has flushed - may be empty here):" >> "$RESULT_FILE"
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,NCPUS >> "$RESULT_FILE" 2>&1 || true
