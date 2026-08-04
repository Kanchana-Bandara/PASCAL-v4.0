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

import numpy as np

from coupler import DEFAULT_DEPTHRANGE

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
