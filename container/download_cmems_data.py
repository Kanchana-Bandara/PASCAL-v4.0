#!/usr/bin/env python3
"""Download a local CMEMS (Copernicus Marine) netCDF subset for an offline
PASCAL advection run.

Why this exists: benchmarks/scenario.py::build_cmems_advection_scenario()
uses opendrift's reader_copernicusmarine.Reader, which streams data live
over the network on every timestep query. That's fine for short
verification runs from a machine with internet access, but wrong for an
actual HPC job - compute nodes on most clusters have no internet egress,
and a multi-year run would otherwise make thousands of blocking network
calls against a rate-limited external API. This script does the network
part once, up front (intended to run on the HPC login node, which
typically does have internet access), producing a local file that
benchmarks/scenario.py::build_cmems_advection_scenario_from_file() then
reads entirely offline.

Default variables/dataset match the mapping confirmed working in
BENCHMARKING.md ("The standard_name_mapping fix") for the physical Arctic
product:
    thetao (sea_water_potential_temperature) -> PASCAL 'temperature'
    mlotst (ocean_mixed_layer_thickness_defined_by_sigma_theta) -> PASCAL 'mld'
    vxo/vyo already carry standard_name eastward_/northward_sea_water_velocity
    (confirmed directly against the dataset 2026-08-04) - OpenDrift matches
    these automatically, no override needed.
food1concentration/irradiance/pred1dens/pred1lightdep are NOT downloaded
here - per the same section of BENCHMARKING.md, there's no working CMEMS
source for them yet (either an unresolved reader bug for the BGC/chl
route, or no product at all), so build_cmems_advection_scenario_from_file()
supplies those as constants, same as the live-reader version does.

Credential resolution mirrors opendrift's reader_copernicusmarine.Reader:
environment variables first, then a .netrc entry for machine
"copernicusmarine". Deliberately does NOT fall through to
copernicusmarine's own interactive-prompt fallback - an HPC login-node
script hanging on stdin waiting for a password is worse than failing
fast with a clear message.

Usage:
    python download_cmems_data.py \\
        --min-lon 9 --max-lon 20 --min-lat 67 --max-lat 73 \\
        --start-date 2022-01-01 --end-date 2024-01-01 \\
        --output-directory /path/to/inputdata/cmems \\
        --output-filename barents_2022_2024.nc

Run on the login node (or anywhere with internet + credentials) BEFORE
submitting the actual Slurm job - see container/hpc_advection_run.sbatch
and usermanual.md's "Running the model in parallel using the container"
section.
"""

import argparse
import os
import sys
from netrc import netrc
from pathlib import Path

DEFAULT_DATASET_ID = "cmems_mod_arc_phy_anfc_6km_detided_P1D-m"
# thetao/mlotst need the explicit rename below; vxo/vyo already carry
# CF standard_names OpenDrift recognizes without help (confirmed 2026-08-04).
DEFAULT_VARIABLES = ["thetao", "mlotst", "vxo", "vyo"]
# Matches coupler.DEFAULT_DEPTHRANGE's max (1246m) - the deepest level
# PASCAL's global_settings['depthrange'] ever asks for by default.
DEFAULT_MAX_DEPTH = 1246.0


def resolve_credentials():
    """Set COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD from the environment
    or ~/.netrc (machine "copernicusmarine"), matching
    opendrift.readers.reader_copernicusmarine.Reader's own resolution
    order. Raises with a clear message rather than letting
    copernicusmarine fall through to an interactive prompt, which would
    hang non-interactive HPC jobs indefinitely."""
    if (
        os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
        and os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")
    ):
        return

    try:
        n = netrc()
        username, _, password = n.authenticators("copernicusmarine")
    except Exception as e:
        raise RuntimeError(
            "No CMEMS credentials found. Set COPERNICUSMARINE_SERVICE_USERNAME"
            " and COPERNICUSMARINE_SERVICE_PASSWORD, or add a 'machine"
            " copernicusmarine' entry to ~/.netrc. (This script deliberately"
            " does not fall back to an interactive prompt.)"
        ) from e

    if username is None or password is None:
        raise RuntimeError(
            "Found a ~/.netrc entry for 'copernicusmarine' but it's missing a"
            " login or password."
        )

    os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = username
    os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = password


def download(
    min_lon, max_lon, min_lat, max_lat, start_date, end_date,
    output_directory, output_filename, dataset_id=DEFAULT_DATASET_ID,
    variables=None, min_depth=0.0, max_depth=DEFAULT_MAX_DEPTH,
    skip_existing=True, dry_run=False,
):
    import copernicusmarine

    variables = list(variables) if variables else list(DEFAULT_VARIABLES)
    resolve_credentials()

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    target = output_directory / output_filename
    if skip_existing and target.exists() and not dry_run:
        print(f"download_cmems_data: {target} already exists, skipping "
              f"(pass --no-skip-existing to force).")
        return target

    print(f"download_cmems_data: requesting dataset={dataset_id} "
          f"variables={variables} lon=[{min_lon},{max_lon}] "
          f"lat=[{min_lat},{max_lat}] depth=[{min_depth},{max_depth}] "
          f"time=[{start_date},{end_date}] -> {target}")

    response = copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        minimum_depth=min_depth,
        maximum_depth=max_depth,
        start_datetime=start_date,
        end_datetime=end_date,
        output_directory=str(output_directory),
        output_filename=output_filename,
        overwrite=not skip_existing,
        dry_run=dry_run,
    )
    print(f"download_cmems_data: done ({response})")
    return target


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--variables", nargs="+", default=None,
                         help=f"default: {DEFAULT_VARIABLES}")
    parser.add_argument("--min-lon", type=float, required=True)
    parser.add_argument("--max-lon", type=float, required=True)
    parser.add_argument("--min-lat", type=float, required=True)
    parser.add_argument("--max-lat", type=float, required=True)
    parser.add_argument("--min-depth", type=float, default=0.0)
    parser.add_argument("--max-depth", type=float, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--start-date", required=True,
                         help="e.g. 2022-01-01")
    parser.add_argument("--end-date", required=True, help="e.g. 2024-01-01")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--output-filename", required=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing",
                         action="store_false",
                         help="Overwrite the target file if it already exists"
                              " (default: skip download and reuse it).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Validate the request against the CMEMS service"
                              " without downloading data.")
    args = parser.parse_args()

    try:
        download(
            args.min_lon, args.max_lon, args.min_lat, args.max_lat,
            args.start_date, args.end_date, args.output_directory,
            args.output_filename, dataset_id=args.dataset_id,
            variables=args.variables, min_depth=args.min_depth,
            max_depth=args.max_depth, skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )
    except RuntimeError as e:
        print(f"download_cmems_data: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
