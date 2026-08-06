"""Self-contained synthetic scenario for benchmarking/profiling PASCAL.

Builds a Pascal1D scenario entirely from synthetic in-memory arrays, with no
dependency on real netCDF input data or the (currently missing) aux_funcs.py
helper that the old testcase/ scripts relied on. This makes benchmark runs
reproducible on any machine with the conda environment installed.

Values for genuinely calibrated constants (critical molting mass thresholds,
developmental coefficients) are illustrative, not validated against field
data -- fine for timing/scaling/profiling work, not for scientific runs.
"""

import datetime as dt
import os

import numpy as np

from pascal.coupler import DEFAULT_DEPTHRANGE

PROFILE_VARS = ["temperature", "food1concentration", "irradiance", "pred1dens"]


def compute_total_tsteps(start_date, duration_years, timestep, isplit=1):
    """Mirror PascalSimulation's timestep-count logic so synthetic arrays
    are sized exactly as long as the run will need them to be."""
    end_time = start_date + dt.timedelta(days=365 * duration_years)
    total_tsteps = int(np.ceil((end_time - start_date) / timestep))
    rem = total_tsteps % isplit
    if rem != 0:
        total_tsteps -= (isplit - rem)
    return total_tsteps


def build_global_settings(stochastic=True, rng=None):
    """Synthetic but structurally-plausible global settings.

    cmm_lower/cmm_upper are monotonically increasing per developmental stage
    (0=egg .. 12=adult), matching the shape/ordering the real model expects.
    """
    cmm_mid = np.array(
        [0.3, 0.5, 0.8, 1.5, 3, 6, 12, 25, 45, 70, 110, 160, 230]
    )
    return {
        "developmentalcoefficient": np.array([595.00, 388.00, 581.00]),
        "maxirradiance": 0.3,
        "minirradiance": 0.001,
        "cmm_lower": cmm_mid * 0.9,
        "cmm_upper": cmm_mid * 1.1,
        "ageceiling": 2180,
        "fecundityceiling": 10,
        "nonvisualpredatorreldensity": 0.00075,
        "backgroundmortalityrisk": 0.00075,
        "virtualindividualthrehold": 100,
        "diapausemetabolicrateadj0": 0.2500,
        "diapausemetabolicrateadj1": 0.5000,
        "energyallocthreshold1": 38.00,
        "energyallocthreshold2": 159.00,
        "stochastic": stochastic,
        "diapausedepththreshold0": 50,
        "diapausedepththreshold1": 500,
        "diapausedepththreshold2": 1200,
        "maxmatingdistance": 1000,
    }


def build_1d_reader(time_len, depthrange=DEFAULT_DEPTHRANGE, rng=None):
    """Synthetic single-water-column environment for Pascal1D.

    Shapes follow Pascal1D.update_environment's expectations:
      - profile variables: [time, depth, n_index]
      - 'mld' (not depth-resolved): [time, n_index]
      - 'z': depth levels, static (no leading time dim)
    n_index is 1 because Pascal1D pins every super-individual to the same
    (only) spatial column.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n_depth = len(depthrange)
    n_index = 1

    surface_irradiance = 0.25
    attenuation = 0.05  # per depth-index, loose stand-in for a real k_d profile
    irradiance_profile = surface_irradiance * np.exp(
        -attenuation * np.arange(n_depth)
    )

    data = {
        "z": -np.asarray(depthrange, dtype=float),
        "temperature": rng.uniform(2, 10, size=(time_len, n_depth, n_index)),
        "food1concentration": rng.uniform(
            0.01, 0.4, size=(time_len, n_depth, n_index)
        ),
        "irradiance": np.tile(
            irradiance_profile[None, :, None], (time_len, 1, n_index)
        ),
        "pred1dens": rng.uniform(0, 0.001, size=(time_len, n_depth, n_index)),
        "mld": np.full((time_len, n_index), 50.0),
    }
    return data


def build_1d_scenario(
    n_super_individuals=50,
    n_virtual_per_super=10000,
    duration_years=0.05,
    timestep_seconds=21600,
    seeding_rate=10,
    stochastic=True,
    seed=0,
    headless="bench_run",
):
    """Return kwargs ready to pass to coupler.Pascal1D(**kwargs)."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # the model itself uses the legacy global RNG

    timestep = dt.timedelta(seconds=timestep_seconds)
    start_date = dt.datetime(2010, 1, 1)

    total_tsteps = compute_total_tsteps(start_date, duration_years, timestep)
    # +5 timesteps of headroom: update_environment() is called once more
    # than the outer step count during prep_environment().
    reader = build_1d_reader(total_tsteps + 5, rng=rng)

    return {
        "nsupindividuals": n_super_individuals,
        "nvindividualspersupindividual": n_virtual_per_super,
        "global_settings": build_global_settings(stochastic=stochastic, rng=rng),
        "reader": reader,
        "timestep": timestep,
        "start_date": start_date,
        "duration": duration_years,
        "seeding_rate": seeding_rate,
        "headless": headless,
    }


def build_advection_scenario(
    n_super_individuals=50,
    n_virtual_per_super=10000,
    duration_years=0.05,
    timestep_seconds=21600,
    seeding_rate=10,
    stochastic=True,
    seed=0,
    headless="bench_run",
):
    """Return kwargs ready to pass to coupler.PascalAdvection(**kwargs)
    (or coupler_parallel.PascalAdvectionParallel(**kwargs)).

    Unlike Pascal1D (where every super-individual shares a single
    environment_index=0, so environment_profiles arrays only ever have one
    column), this gives each super-individual its own OpenDrift tracker
    element/column via a spatially-uniform ConstantReader. That's what
    actually exercises the thing coupler_parallel.py's per-individual
    slicing exists for: in this scenario environment_profiles arrays have
    shape (n_depth, n_super_individuals), and every SuperIndividual holds a
    reference to the *whole* array, not just its own column.
    """
    from opendrift.readers.reader_constant import Reader as ConstantReader

    np.random.seed(seed)

    timestep = dt.timedelta(seconds=timestep_seconds)
    start_date = dt.datetime(2010, 1, 1)

    reader = ConstantReader({
        "x_sea_water_velocity": 0.1,
        "y_sea_water_velocity": 0.05,
        "x_wind": 0,
        "y_wind": 0,
        "temperature": 8,
        "sea_water_salinity": 35,
        "land_binary_mask": 0,
        "ocean_vertical_diffusivity": 0.02,
        "food1concentration": 0.05,
        "irradiance": 0.1,
        "pred1dens": 0.00001,
        "pred1lightdep": 0.1,
        "mld": 100,
    })

    outputgrid = {"lon": [-1, 0, 1], "lat": [69, 70, 71]}

    return {
        "nsupindividuals": n_super_individuals,
        "nvindividualspersupindividual": n_virtual_per_super,
        "global_settings": build_global_settings(stochastic=stochastic),
        "reader": reader,
        "timestep": timestep,
        "start_date": start_date,
        "duration": duration_years,
        "seeding_rate": seeding_rate,
        "start_locations": [[0.0, 70.0]],
        "outputgrid": outputgrid,
        "headless": headless,
    }


def _alias_reader_variable(reader, new_name, existing_name):
    """Register `new_name` as an alias for `existing_name` on an
    already-constructed CF-based reader (e.g. reader_copernicusmarine.Reader).

    reader.variable_mapping (built once in __init__) maps standard_name ->
    raw file variable name; reader.variables is a *snapshot* list taken
    from variable_mapping.keys() at construction time, not a live view of
    it - so both need updating for the reader to actually offer the alias
    to OpenDrift's environment system.
    """
    reader.variable_mapping[new_name] = reader.variable_mapping[existing_name]
    if new_name not in reader.variables:
        reader.variables.append(new_name)


def build_cmems_advection_scenario(
    n_super_individuals=50,
    n_virtual_per_super=10000,
    duration_years=0.05,
    timestep_seconds=21600,
    seeding_rate=10,
    stochastic=True,
    seed=0,
    start_date=None,
    start_location=(14.25, 69.8),
    food1concentration_constant=0.05,
    pred1dens_constant=0.00001,
    pred1lightdep_constant=0.1,
    irradiance_constant=0.1,
    headless="bench_run",
):
    """Return kwargs ready to pass to coupler.PascalAdvection(**kwargs),
    backed by live Copernicus Marine (CMEMS) physical data instead of a
    synthetic reader - for checking the realistic 3D deployment path
    specifically, not for routine/repeatable benchmarking (this hits a
    real, rate-limited external API and needs network + credentials - see
    BENCHMARKING.md's "live-CMEMS reader configuration feeds PASCAL zero
    real data" section for why this function exists at all).

    PASCAL's required_variables (pascal_drift.py) use short internal names
    that don't match CMEMS's real CF standard names, so nothing gets
    matched and everything silently falls back to constant defaults unless
    aliased. Confirmed mapping (2026-08-04, reviewed):
        temperature -> sea_water_temperature       (physical reader)
        mld         -> ocean_mixed_layer_thickness (physical reader)
    x/y_sea_water_velocity need no alias - OpenDrift's reader machinery
    already rotates eastward/northward -> x/y internally.
    ocean_vertical_diffusivity/land_binary_mask need no alias either -
    both handled via tracker_config below (a parameterized diffusivity
    model and OpenDrift's auto-landmask), not reader lookup.

    Only the physical CMEMS product is fetched here - food1concentration
    is given as a constant (not sourced from the biogeochemistry reader),
    per 2026-08-05 decision: the food1concentration alias to
    mass_concentration_of_chlorophyll_a_in_sea_water was confirmed
    correctly *registered*, but returns 0.0 in practice via OpenDrift's
    interpolated fetch despite the raw underlying variable having real
    data nearby - an unresolved OpenDrift/reader interpolation issue (see
    BENCHMARKING.md for the full writeup), to be investigated separately.
    Using a constant here means only one live CMEMS product needs
    retrieving for benchmarking.

    IMPORTANT, found while wiring up the constant: this isn't only a
    chlorophyll/BGC-reader problem. Even a plain ConstantReader-supplied
    food1concentration reads back as 0.0 here when the live `physical`
    CMEMS reader is *also* in the reader list - identical constant, same
    settings, works correctly (reads back as 0.05) when ConstantReader is
    the *only* reader. So the underlying bug is a broader interaction
    between a live CMEMS reader and any other reader providing profile
    variables, not something specific to the chl mapping - worth knowing
    for the separate investigation. Population still collapses in this
    scenario as a result; not something this constant swap fixes.

    irradiance/pred1dens/pred1lightdep have no CMEMS equivalent at all:
    pred1dens/pred1lightdep are meant to come from a separate, not-yet-
    available precomputed netCDF (see input_pascal_cmems1/); irradiance
    needs a real derivation from a surface radiation product (e.g. ERA5 via
    the Copernicus Climate Data Store) not yet integrated. All three (plus
    food1concentration) are given as constants here as an explicit
    stand-in, using the same values already proven not to cause the
    population collapse a food1concentration=0 fallback does (see
    build_advection_scenario()).
    """
    from netrc import netrc

    from opendrift.readers.reader_constant import Reader as ConstantReader
    from opendrift.readers.reader_copernicusmarine import Reader as CMEMSReader

    np.random.seed(seed)

    if "COPERNICUSMARINE_SERVICE_USERNAME" not in os.environ:
        credentials = netrc()
        os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = credentials.hosts["copernicusmarine"][0]
        os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = credentials.hosts["copernicusmarine"][1]

    physical = CMEMSReader("cmems_mod_arc_phy_anfc_6km_detided_P1D-m")
    _alias_reader_variable(physical, "temperature", "sea_water_temperature")
    _alias_reader_variable(physical, "mld", "ocean_mixed_layer_thickness")

    constants = ConstantReader({
        "food1concentration": food1concentration_constant,
        "pred1dens": pred1dens_constant,
        "pred1lightdep": pred1lightdep_constant,
        "irradiance": irradiance_constant,
    })

    timestep = dt.timedelta(seconds=timestep_seconds)
    if start_date is None:
        start_date = dt.datetime(2024, 6, 1)

    lon, lat = start_location
    outputgrid = {
        "lon": [lon - 1, lon, lon + 1, lon + 2],
        "lat": [lat - 1, lat, lat + 1],
    }

    return {
        "nsupindividuals": n_super_individuals,
        "nvindividualspersupindividual": n_virtual_per_super,
        "global_settings": build_global_settings(stochastic=stochastic),
        "reader": [physical, constants],
        "timestep": timestep,
        "start_date": start_date,
        "duration": duration_years,
        "seeding_rate": seeding_rate,
        "start_locations": [[lon, lat]],
        "outputgrid": outputgrid,
        "tracker_config": {
            "general:use_auto_landmask": True,
            "vertical_mixing:diffusivitymodel": "windspeed_Sundby1983",
        },
        "headless": headless,
    }


def build_cmems_advection_scenario_from_file(
    cmems_file,
    n_super_individuals=50,
    n_virtual_per_super=10000,
    duration_years=0.05,
    timestep_seconds=21600,
    seeding_rate=10,
    stochastic=True,
    seed=0,
    start_date=None,
    start_location=(14.25, 69.8),
    food1concentration_constant=0.05,
    pred1dens_constant=0.00001,
    pred1lightdep_constant=0.1,
    irradiance_constant=0.1,
    headless="bench_run",
):
    """Same scientific setup as build_cmems_advection_scenario(), but reads
    a local netCDF file (as produced by hpc/download_cmems_data.py)
    via reader_netCDF_CF_generic instead of streaming live data via
    reader_copernicusmarine. Intended for actual HPC runs, where compute
    nodes typically have no internet access and a multi-year run can't
    afford per-timestep calls to a rate-limited external API - see
    hpc/download_cmems_data.py's module docstring and
    usermanual.md's "Running the model in parallel using the container"
    section.

    Unlike the live reader (reader_copernicusmarine.Reader), which doesn't
    forward a standard_name_mapping kwarg to its parent class - hence
    _alias_reader_variable()'s post-construction monkeypatch above -
    reader_netCDF_CF_generic.Reader takes standard_name_mapping directly,
    so the thetao/mlotst -> temperature/mld rename happens cleanly at
    construction time. vxo/vyo need no mapping: confirmed 2026-08-04 they
    already carry CF standard_names eastward_/northward_sea_water_velocity,
    which OpenDrift matches to x_/y_sea_water_velocity automatically.

    food1concentration/irradiance/pred1dens/pred1lightdep are constants
    here for the same reason as build_cmems_advection_scenario(): no
    working CMEMS source for them yet (see BENCHMARKING.md).
    """
    from opendrift.readers.reader_constant import Reader as ConstantReader
    from opendrift.readers.reader_netCDF_CF_generic import Reader as CFReader

    np.random.seed(seed)

    physical = CFReader(
        str(cmems_file),
        standard_name_mapping={"thetao": "temperature", "mlotst": "mld"},
    )

    constants = ConstantReader({
        "food1concentration": food1concentration_constant,
        "pred1dens": pred1dens_constant,
        "pred1lightdep": pred1lightdep_constant,
        "irradiance": irradiance_constant,
    })

    timestep = dt.timedelta(seconds=timestep_seconds)
    if start_date is None:
        start_date = dt.datetime(2024, 6, 1)

    lon, lat = start_location
    outputgrid = {
        "lon": [lon - 1, lon, lon + 1, lon + 2],
        "lat": [lat - 1, lat, lat + 1],
    }

    return {
        "nsupindividuals": n_super_individuals,
        "nvindividualspersupindividual": n_virtual_per_super,
        "global_settings": build_global_settings(stochastic=stochastic),
        "reader": [physical, constants],
        "timestep": timestep,
        "start_date": start_date,
        "duration": duration_years,
        "seeding_rate": seeding_rate,
        "start_locations": [[lon, lat]],
        "outputgrid": outputgrid,
        "tracker_config": {
            "general:use_auto_landmask": True,
            "vertical_mixing:diffusivitymodel": "windspeed_Sundby1983",
        },
        "headless": headless,
    }


def build_global_settings_from_config(biology):
    """Build the global_settings dict PascalSimulation expects directly
    from a run config's biology section (see config/schema.py's DEFAULTS
    for the values this reproduces when nothing is overridden), instead
    of build_global_settings()'s fixed constants above - this is what
    actually makes biology parameters configurable per run rather than
    only the stochastic flag build_global_settings() ever exposed.
    """
    cmm_mid = np.array(biology["cmm_mid"])
    settings = {
        "developmentalcoefficient": np.array(biology["developmentalcoefficient"]),
        "maxirradiance": biology["maxirradiance"],
        "minirradiance": biology["minirradiance"],
        "cmm_lower": cmm_mid * 0.9,
        "cmm_upper": cmm_mid * 1.1,
        "ageceiling": biology["ageceiling"],
        "fecundityceiling": biology["fecundityceiling"],
        "nonvisualpredatorreldensity": biology["nonvisualpredatorreldensity"],
        "backgroundmortalityrisk": biology["backgroundmortalityrisk"],
        "virtualindividualthrehold": biology["virtualindividualthrehold"],
        "diapausemetabolicrateadj0": biology["diapausemetabolicrateadj0"],
        "diapausemetabolicrateadj1": biology["diapausemetabolicrateadj1"],
        "energyallocthreshold1": biology["energyallocthreshold1"],
        "energyallocthreshold2": biology["energyallocthreshold2"],
        "stochastic": biology["stochastic"],
        "diapausedepththreshold0": biology["diapausedepththreshold0"],
        "diapausedepththreshold1": biology["diapausedepththreshold1"],
        "diapausedepththreshold2": biology["diapausedepththreshold2"],
        "maxmatingdistance": biology["maxmatingdistance"],
    }
    if biology.get("depthrange") is not None:
        settings["depthrange"] = np.array(biology["depthrange"])
    return settings


def build_scenario_from_config(config):
    """Build PascalSimulation kwargs from a validated run config (see
    config/loader.py::load_config()). Dispatches to the build_*_scenario()
    functions above for reader/tracker wiring (unchanged, so behavior for
    a default/empty config matches what those functions always did), then
    overlays the config's population/time/biology/tracker/output sections
    so a single YAML file controls everything a run needs - not just the
    handful of fields the old CLI/env-var surface exposed.
    """
    run = config["run"]
    population = config["population"]
    time_cfg = config["time"]
    location = config["location"]
    reader = config["reader"]

    common = dict(
        n_super_individuals=population["n_super_individuals"],
        n_virtual_per_super=population["n_virtual_per_super"],
        duration_years=time_cfg["duration_years"],
        timestep_seconds=time_cfg["timestep_seconds"],
        seeding_rate=population["seeding_rate"],
        stochastic=config["biology"]["stochastic"],
        seed=run["seed"],
        headless=run["name"],
    )

    scenario = run["scenario"]
    if scenario == "1d":
        kwargs = build_1d_scenario(**common)
    elif scenario == "advection":
        kwargs = build_advection_scenario(**common)
        if location.get("start_lon") is not None and location.get("start_lat") is not None:
            kwargs["start_locations"] = [[location["start_lon"], location["start_lat"]]]
    elif scenario == "advection_cmems_file":
        start_date = (
            dt.datetime.fromisoformat(time_cfg["start_date"])
            if time_cfg.get("start_date") else None
        )
        kwargs = build_cmems_advection_scenario_from_file(
            reader["cmems_file"],
            start_date=start_date,
            start_location=(location["start_lon"], location["start_lat"]),
            food1concentration_constant=reader["food1concentration"],
            pred1dens_constant=reader["pred1dens"],
            pred1lightdep_constant=reader["pred1lightdep"],
            irradiance_constant=reader["irradiance"],
            **common,
        )
    else:
        raise ValueError(f"Unknown run.scenario: {scenario!r}")

    kwargs["global_settings"] = build_global_settings_from_config(config["biology"])

    tracker = config.get("tracker") or {}
    kwargs.setdefault("tracker_config", {})
    if tracker.get("use_auto_landmask") is not None:
        kwargs["tracker_config"]["general:use_auto_landmask"] = tracker["use_auto_landmask"]
    if tracker.get("diffusivitymodel") is not None:
        kwargs["tracker_config"]["vertical_mixing:diffusivitymodel"] = tracker["diffusivitymodel"]
    if tracker.get("diapause_depth") is not None:
        kwargs["diapause_depth"] = tracker["diapause_depth"]

    output = config.get("output") or {}
    if output.get("outputgrid") is not None:
        kwargs["outputgrid"] = output["outputgrid"]
    if output.get("debug_variables"):
        kwargs["debug"] = output["debug_variables"]
    if output.get("verbose") is not None:
        kwargs["verbose"] = output["verbose"]

    return kwargs
