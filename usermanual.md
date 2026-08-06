# Pan-Arctic Behavioral and Life-history Simulator for Calanus (PASCAL v4.0)
## Overview
PASCAL is the 4th iteration of a very-high biological resolution behavioral and life-history simulation model designed for the copepods of genus _Calanus_ inhabiting the North Atlantic and the Arctic. It was initiated in 2016 under the VISTA PhD project of Kanchana Bandara (Akvaplan niva AS, Norway), titled "High-Resolution Modelling of Diel and Seasonal Vertical Migration of High-Latitude Zooplankton" supervised by Ketil Eiane (Nord University, Norway), Øystein Varpe (University of Bergen) and Rubao Ji (Woods Hole Oceanographic Institution, USA). Since its inception, PASCAL recieved 2 subsequent upgrades; PASCAL v2.0 in 2018 under the same VISTA project and and PASCAL v3.1 in 2021 under the NFR GLIDER Phase - I project with the oversight of Vigdis Tverberg (Nord University, Norway). PASCAL v4.0 is the latest iteration that sees a total overhaul of the core model architecture and brings improved performance. A basic comparison of the four versions of PASCAL is listed below:

#### Table 1: A basic comparison of four versions of PASCAL
|Attribute|PASCAL v1.0|PASCAL v2.0|PASCAL v3.1|Pascal v4.0|
|-------|-------|-------|------|------|
|Model Architecture|strategy-oriented|strategy-oriented|individual-based|super-individual-based
|Model Dimensions|2D|2D|2D|4D|
|Temporal Resolution|1 h|1 h|6 h|6 h
|Spatial Resolution (x, y)|-|-|-|> 9 km
|Spatial Resolution (z)|1 m |1 m|1 m|> 1 m (37 levels)
|Simulated Population Size|1 x 10<sup>6</sup>|2.5 x 10<sup>6</sup>|1 x 10<sup>6</sup>|resource/risk-dependent ceiling
|Simulated Taxa|Generalized copepod|_Calanus_ spp. |_C.finmarchicus_|_C.finmarchicus_
|Programming Language|R|R|FORTRAN(95)|Python(3.x)
|Project Attribution|VISTA 6165|VISTA 6165|GLIDER Phase-I|Migratory Crossroads
|Funding|VISTA, StatOil|VISTA, Statoil|Norwegian Research Council|Norwegian Research Council
|

## Running the model in parallel using the container

This section covers the `class_parallel` branch's HPC deployment path:
running PASCAL inside an Apptainer/Singularity container, either as a
quick multiprocessing scaling check or as a real multi-year, 3D
(advective) run against real ocean forcing at production scale (10,000
super-individuals). All of the infrastructure referenced below lives in
`hpc/`, `config/` and `scenarios/`; the full research log behind the design
decisions (what was tried, measured, and why) is in `BENCHMARKING.md` -
this section is the "how do I actually run it" summary, not a repeat of
that log.

A real run is driven entirely by one YAML config (see `config/runs/*.yaml`
and `config/schema.py` for every field a config can set) instead of a long
list of CLI flags/env vars - see step 4 below.

### Why a container at all

`coupler_parallel.py` parallelizes `update_lifestage()` across CPU cores
with `multiprocessing.Pool` - useful mainly on a shared HPC cluster with
many cores per node, where a plain conda environment is awkward to
reproduce identically across nodes/users. The container bakes in the
`pascal_modular` conda environment (see `hpc/environment.yml`) only
- **not** this repository or `opendrift_pascal`, both of which are
actively-developed local checkouts that get bind-mounted in at run time
(see `hpc/container.def`'s `%post` comments for why baking them in
would be wrong). This means rebuilding the image is only needed when
`environment.yml` changes, not every time the model code changes.

**Always pass `--writable-tmpfs`** (as every example below does) when
running either app. Both apps editable-install the bind-mounted
`opendrift_pascal`/this repo into the conda env on every invocation, which
needs a writable container filesystem. Without the flag, this doesn't
just fail cleanly - `pip` silently retries as a `--user` install, and
because Apptainer bind-mounts your real `$HOME` by default, that writes
broken editable-install metadata straight into your **host's** real
Python environment, breaking `import opendrift`/`import coupler` outside
the container too. `run_in_container.sh` now checks for this and refuses
to continue rather than let it happen silently (found the hard way
2026-08-04 - see `BENCHMARKING.md`), but the flag is still required for
either app to actually work.

### 1. Build the image

From this repo's root, on a machine where you have `--fakeroot` (or root)
- typically your own workstation, not the HPC login node, which usually
doesn't grant ordinary users fakeroot:

```bash
apptainer build --fakeroot hpc/pascal.sif hpc/container.def
```

Takes a few minutes (mostly `mamba env create`). Copy the resulting
`hpc/pascal.sif` (a single ~1.2GB file, gitignored) to the cluster
rather than trying to build there.

### 2. Download CMEMS forcing data (login node, needs internet)

Real Copernicus Marine (CMEMS) ocean data is what actually drives the
advective model at HPC scale - not the synthetic scenarios used for
routine benchmarking. Compute nodes on most clusters have no internet
access, so this is a separate, one-time step run on the login node (or
anywhere with internet + credentials), **before** submitting the actual
Slurm job - not something the compute job itself does. Credentials:
either set `COPERNICUSMARINE_SERVICE_USERNAME`/`COPERNICUSMARINE_SERVICE_PASSWORD`,
or add a `machine copernicusmarine` entry to `~/.netrc`.

```bash
apptainer run --app download-cmems --writable-tmpfs \
    --bind "$(pwd)":/pascal \
    --bind /path/to/opendrift_pascal:/opendrift \
    hpc/pascal.sif \
    --min-lon 9 --max-lon 20 --min-lat 67 --max-lat 73 \
    --start-date 2022-01-01 --end-date 2024-01-01 \
    --output-directory inputdata/cmems --output-filename barents_2022_2024.nc
```

Use `--output-directory inputdata/cmems` (i.e. somewhere under this repo,
which is already bind-mounted) rather than an arbitrary path - that way
the file is guaranteed visible inside the container at run time via the
same `--bind`, without depending on Apptainer's default `/tmp`/`$HOME`
auto-mounts, which some HPC sites disable for security. `*.nc` is already
gitignored, so this won't accidentally get committed. Pick a bounding box
generous enough for however far your super-individuals might actually
drift over the run's duration - a multi-year run in a region with real
currents can travel further than the ~1-2 degree margins used for short
verification runs. See `hpc/download_cmems_data.py --help` for all
options (variables, depth range, dataset ID). This is also the path a
run config's `reader.cmems_file` field (step 4) needs to point at.

Re-running the same command later reuses the existing file by default
(`--no-skip-existing` to force a fresh download).

### 3. Write (or reuse) a run config, and try it small first

A real run is defined entirely by one YAML file - see
`config/runs/advection_hpc_barents.yaml` for the production-scale example
(10,000 super-individuals, 2 years) and `config/schema.py` for every field
a config can set (population size, timing, location, reader/tracker
settings, biology parameters, output options). Copy/edit one of
`config/runs/*.yaml` for your own run, or just override individual fields
at submission time with `--override key.path=value` (repeatable) - no
need to write a new file for a one-off change.

Before submitting anything to Slurm, sanity-check the config (and the
downloaded file/aliasing/start-location) with a tiny/cheap run of the same
scenario - `config/runs/advection_quick_check.yaml` is exactly this,
already sized down:

```bash
apptainer run --app run --writable-tmpfs \
    --bind "$(pwd)":/pascal \
    --bind /path/to/opendrift_pascal:/opendrift \
    hpc/pascal.sif \
    --config config/runs/advection_quick_check.yaml
```

If this completes and prints a `final_population_size` greater than 0,
it's safe to submit the full-scale job.

### 4. Submit to Slurm

Two different scripts, for two different questions:

- **`hpc/submit.sh`** - the real test: one job, a multi-year advective run
  at 10,000 super-individuals against the file from step 2. This is what
  you want for an actual scientific-scale HPC run:

  ```bash
  REPO_ROOT=/path/to/pascal_classparallel_branch \
  OPENDRIFT_ROOT=/path/to/opendrift_pascal \
  CONFIG=config/runs/advection_hpc_barents.yaml \
  ./hpc/submit.sh
  ```

  Defaults to `MODE=sequential` and 1 cpu - see the caveat below before
  changing `MODE=parallel`. `TIME`/`MEM`/`CPUS` are all overridable (see
  the script's header comment); the defaults are deliberately generous,
  *unvalidated* starting points, not a measured requirement - check the
  actual wall time/memory of the first run (written into the job's result
  file via `sacct`) and tighten from there. Anything the config file
  itself doesn't already cover for this specific submission (e.g. a
  different `reader.cmems_file` path on this cluster) can be added via
  `OVERRIDES="reader.cmems_file=... time.start_date=..."` (space-separated
  `key.path=value` tokens).

- **`hpc/submit_sweep.sh`** - a different question: whether
  `multiprocessing.Pool` (`MODE=parallel`) actually pays off at real HPC
  core counts. On an 8-core laptop it was measured as a **net loss**,
  22-27% slower than sequential at every size tested up to 1500
  super-individuals (`BENCHMARKING.md`'s Phase 2) - this sweep exists to
  check whether that changes with more cores per worker-dispatch, by
  running the cheap synthetic 1D scenario (`config/runs/scaling_sweep_1d.yaml`)
  across a range of core counts. Not the scenario you want for a real
  scientific run; see `submit_sweep.sh`'s own header for usage. Summarize
  its results with `python benchmarks/summarize_scaling_results.py results/`.

**Caveat on `MODE=parallel`:** given the Phase 2 finding above, there is
no evidence yet that parallel mode is faster for this model on any
hardware tested so far - it's `sequential` by default in `submit.sh` for
that reason. Only switch to `parallel` for the real run after
`submit_sweep.sh`'s results on your actual cluster hardware show it's
worthwhile at the core count you plan to use.

### Where output lands

`submit.sh` writes the model's real output (`output_ps.nc`,
`lifestats.csv`, etc.) to
`results/<config basename>/job<slurm-job-id>/<run.name>/` under
`REPO_ROOT` (e.g. `results/advection_hpc_barents/job12345/advection_10000si_2y/`
for the example config above - `run.name` is the config's own field, see
`config/schema.py`) - not to an ephemeral `/tmp` directory inside the
container, which would otherwise disappear (or live on node-local scratch
never seen again) once the job ends. `submit_sweep.sh`'s jobs don't do
this (only their timing summary line is kept) - that sweep is about
wall-clock scaling, not the scientific output itself.

### Known limitations to be aware of before a real run

- `coupler_parallel.py` is single-node only (no MPI) - parallelism is
  limited to one node's core count.
- `food1concentration`, `irradiance`, `pred1dens` and `pred1lightdep` are
  constants in the CMEMS-backed scenario, not real forcing - there's no
  working CMEMS source for them yet (an unresolved OpenDrift reader
  interaction for the chlorophyll route, and no product at all for the
  others). See `BENCHMARKING.md`'s "Found, NOT fixed" section for the
  full investigation. Population dynamics driven by these variables
  should be interpreted with that in mind.


