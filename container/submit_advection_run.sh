#!/bin/bash
# Submit a single, real-scale PASCAL validation run: a multi-year
# advective (3D) simulation at 10,000 super-individuals, driven by real
# CMEMS forcing read from a local file (not the live/streaming reader -
# see container/download_cmems_data.py for why). This is the actual
# Phase 8 HPC test, as distinct from submit_scaling_sweep.sh (which
# sweeps many small/cheap jobs across core counts using the synthetic 1D
# scenario, to answer a different question: whether multiprocessing pays
# off at real core counts).
#
# Two-step workflow, because compute nodes typically have no internet
# access:
#   1. On the login node (has internet + your CMEMS credentials):
#        python container/download_cmems_data.py \
#            --min-lon ... --max-lon ... --min-lat ... --max-lat ... \
#            --start-date ... --end-date ... \
#            --output-directory ... --output-filename barents_run.nc
#   2. Then, still on the login node, submit this job:
#        REPO_ROOT=/path/to/pascal_classparallel_branch \
#        OPENDRIFT_ROOT=/path/to/opendrift_pascal \
#        CMEMS_FILE=/path/to/barents_run.nc \
#        ./container/submit_advection_run.sh
#
# See usermanual.md's "Running the model in parallel using the container"
# section for the full walkthrough, and container.def's %help for the
# apptainer build/run commands.
#
# Requires container/pascal.sif already built - see container.def's %help.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:?set REPO_ROOT to the pascal_classparallel_branch checkout path}"
OPENDRIFT_ROOT="${OPENDRIFT_ROOT:?set OPENDRIFT_ROOT to the opendrift_pascal checkout path}"
CMEMS_FILE="${CMEMS_FILE:?set CMEMS_FILE to a local netCDF from container/download_cmems_data.py}"

N_SUPER="${N_SUPER:-10000}"
DURATION="${DURATION:-2}"          # years
MODE="${MODE:-sequential}"         # sequential|parallel - see BENCHMARKING.md
                                    # Phase 2 before defaulting to parallel
CPUS="${CPUS:-1}"
SEED="${SEED:-0}"
START_LON="${START_LON:-14.25}"
START_LAT="${START_LAT:-69.8}"     # default: same Barents Sea point as
                                    # testcase_nongit/advection_test_barents.py
START_DATE="${START_DATE:-2022-01-01}"  # must fall inside CMEMS_FILE's time range
# Rough, unmeasured-at-this-scale estimates (see BENCHMARKING.md's
# advection-scenario timing: ~13.5s for 200 super-individuals/0.5y with a
# cheap synthetic ConstantReader). Real CMEMS interpolation per timestep
# is unmeasured at 10,000 super-individuals/multi-year, and could be
# notably slower - these are deliberately generous starting points, not
# a validated estimate. Check the job's actual wall time/memory (this
# script's underlying job writes a `sacct` line to its result file) and
# tighten on the next submission rather than trusting these numbers.
TIME="${TIME:-24:00:00}"
MEM="${MEM:-64G}"

if [ ! -f "$REPO_ROOT/container/pascal.sif" ]; then
    echo "submit_advection_run.sh: $REPO_ROOT/container/pascal.sif not found." >&2
    echo "Build it first: apptainer build --fakeroot container/pascal.sif container/container.def" >&2
    exit 1
fi
if [ ! -f "$CMEMS_FILE" ]; then
    echo "submit_advection_run.sh: CMEMS_FILE=$CMEMS_FILE not found." >&2
    echo "Download it first with container/download_cmems_data.py (see this" \
         "script's header comment for the two-step workflow)." >&2
    exit 1
fi

mkdir -p "$REPO_ROOT/results"
RUN_OUTPUT_SUBDIR="advection_${N_SUPER}si_${DURATION}y"

sbatch \
    --chdir="$REPO_ROOT" \
    --job-name="pascal_advection_${N_SUPER}si_${DURATION}y" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --output="$REPO_ROOT/results/pascal_advection_%j.log" \
    --export=ALL,MODE="$MODE",N_SUPER="$N_SUPER",DURATION="$DURATION",SCENARIO=advection_cmems_file,SEED="$SEED",REPO_ROOT="$REPO_ROOT",OPENDRIFT_ROOT="$OPENDRIFT_ROOT",CMEMS_FILE="$CMEMS_FILE",START_LON="$START_LON",START_LAT="$START_LAT",START_DATE="$START_DATE",RUN_OUTPUT_SUBDIR="$RUN_OUTPUT_SUBDIR" \
    "$REPO_ROOT/container/hpc_scaling_test.sbatch"

echo "Submitted: $N_SUPER super-individuals, ${DURATION}y, mode=$MODE, $CPUS cpu(s)."
echo "Timing/summary line: $REPO_ROOT/results/scaling_cpus${CPUS}_${MODE}_job<id>.txt"
echo "Full model output (output_ps.nc, lifestats.csv):"
echo "  $REPO_ROOT/results/$RUN_OUTPUT_SUBDIR/job<id>/bench_run/"
