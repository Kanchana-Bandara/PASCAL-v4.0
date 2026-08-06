#!/bin/bash
# Submit a single PASCAL Slurm job, driven by a YAML run config (see
# config/runs/*.yaml and config/schema.py for every field a config can
# set). Replaces the old submit_advection_run.sh (this script covers the
# same real Phase 8 HPC test - see config/runs/advection_hpc_barents.yaml
# - just via CONFIG instead of N_SUPER/DURATION/... flags); for a sweep
# across core counts, see submit_sweep.sh, which just calls this
# repeatedly.
#
# Two-step workflow for the CMEMS-backed scenario, because compute nodes
# typically have no internet access:
#   1. On the login node (has internet + your CMEMS credentials):
#        python hpc/download_cmems_data.py \
#            --min-lon ... --max-lon ... --min-lat ... --max-lat ... \
#            --start-date ... --end-date ... \
#            --output-directory inputdata/cmems --output-filename barents_run.nc
#   2. Then, still on the login node, submit this job:
#        REPO_ROOT=/path/to/pascal_classparallel_branch \
#        OPENDRIFT_ROOT=/path/to/opendrift_pascal \
#        CONFIG=config/runs/advection_hpc_barents.yaml \
#        ./hpc/submit.sh
#
# See usermanual.md's "Running the model in parallel using the container"
# section for the full walkthrough, and container.def's %help for the
# apptainer build/run commands.
#
# Requires hpc/pascal.sif already built - see container.def's %help.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:?set REPO_ROOT to the pascal_classparallel_branch checkout path}"
OPENDRIFT_ROOT="${OPENDRIFT_ROOT:?set OPENDRIFT_ROOT to the opendrift_pascal checkout path}"
CONFIG="${CONFIG:?set CONFIG to a YAML run config path relative to REPO_ROOT, e.g. config/runs/advection_hpc_barents.yaml}"

MODE="${MODE:-sequential}"          # sequential|parallel - see BENCHMARKING.md
                                     # Phase 2 before defaulting to parallel
CPUS="${CPUS:-1}"
# Extra --override key.path=value tokens (space-separated), for anything
# a submission needs to change beyond what CONFIG's file already says -
# e.g. OVERRIDES="reader.cmems_file=inputdata/cmems/barents_run.nc".
OVERRIDES="${OVERRIDES:-}"
JOB_NAME="${JOB_NAME:-pascal_$(basename "$CONFIG" .yaml)}"
RUN_OUTPUT_SUBDIR="${RUN_OUTPUT_SUBDIR:-$(basename "$CONFIG" .yaml)}"
# Rough, unmeasured-at-scale estimates - see BENCHMARKING.md's advection-
# scenario timing sections. Check the job's actual wall time/memory (this
# script's underlying job writes a `sacct` line to its result file) and
# tighten on the next submission rather than trusting these numbers.
TIME="${TIME:-24:00:00}"
MEM="${MEM:-64G}"

if [ ! -f "$REPO_ROOT/hpc/pascal.sif" ]; then
    echo "submit.sh: $REPO_ROOT/hpc/pascal.sif not found." >&2
    echo "Build it first: apptainer build --fakeroot hpc/pascal.sif hpc/container.def" >&2
    exit 1
fi
if [ ! -f "$REPO_ROOT/$CONFIG" ]; then
    echo "submit.sh: config $REPO_ROOT/$CONFIG not found." >&2
    exit 1
fi

mkdir -p "$REPO_ROOT/results"

sbatch \
    --chdir="$REPO_ROOT" \
    --job-name="$JOB_NAME" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --output="$REPO_ROOT/results/${JOB_NAME}_%j.log" \
    --export=ALL,CONFIG="$CONFIG",MODE="$MODE",OVERRIDES="$OVERRIDES",REPO_ROOT="$REPO_ROOT",OPENDRIFT_ROOT="$OPENDRIFT_ROOT",RUN_OUTPUT_SUBDIR="$RUN_OUTPUT_SUBDIR" \
    "$REPO_ROOT/hpc/submit.sbatch"

echo "Submitted: config=$CONFIG mode=$MODE cpus=$CPUS"
echo "Timing/summary line: $REPO_ROOT/results/scaling_cpus${CPUS}_${MODE}_job<id>.txt"
echo "Full model output (output_ps.nc, lifestats.csv):"
echo "  $REPO_ROOT/results/$RUN_OUTPUT_SUBDIR/job<id>/<run.name from $CONFIG>/"
