"""Default values and validation for PASCAL run configs (see loader.py).

DEFAULTS mirrors the values that used to be hardcoded across
benchmarks/scenario.py's build_*_scenario() functions and
build_global_settings() specifically - loading an empty/minimal YAML
reproduces the same run those functions produced before, just now
overridable per-field from one file instead of requiring a code change.
"""

SCENARIOS = ("1d", "advection", "advection_cmems_file")

DEFAULTS = {
    "run": {
        "name": "bench_run",
        "scenario": "1d",
        "mode": "sequential",
        "seed": 0,
        "n_workers": None,
    },
    "population": {
        "n_super_individuals": 50,
        "n_virtual_per_super": 10000,
        "seeding_rate": 10,
    },
    "time": {
        "start_date": None,
        "duration_years": 0.05,
        "timestep_seconds": 21600,
    },
    "location": {
        # Matches build_cmems_advection_scenario[_from_file]()'s former
        # hardcoded default (a Barents Sea point) - the "advection"
        # synthetic-ConstantReader scenario keeps its own separate
        # hardcoded default ([0.0, 70.0]) unless start_lon/start_lat are
        # explicitly set here.
        "start_lon": 14.25,
        "start_lat": 69.8,
    },
    "reader": {
        "cmems_file": None,  # required when run.scenario == advection_cmems_file
        "food1concentration": 0.05,
        "pred1dens": 0.00001,
        "pred1lightdep": 0.1,
        "irradiance": 0.1,
    },
    "tracker": {
        "use_auto_landmask": None,  # None = leave the scenario builder's own default
        "diffusivitymodel": None,
        "diapause_depth": None,
    },
    "biology": {
        "stochastic": True,
        "developmentalcoefficient": [595.00, 388.00, 581.00],
        "maxirradiance": 0.3,
        "minirradiance": 0.001,
        # Critical molting mass midpoints per developmental stage
        # (0=egg .. 12=adult) - cmm_lower/cmm_upper are derived as
        # +/-10% of this, matching build_global_settings()'s prior
        # hardcoded derivation exactly.
        "cmm_mid": [0.3, 0.5, 0.8, 1.5, 3, 6, 12, 25, 45, 70, 110, 160, 230],
        "ageceiling": 2180,
        "fecundityceiling": 10,
        "nonvisualpredatorreldensity": 0.00075,
        "backgroundmortalityrisk": 0.00075,
        "virtualindividualthrehold": 100,
        "diapausemetabolicrateadj0": 0.2500,
        "diapausemetabolicrateadj1": 0.5000,
        "energyallocthreshold1": 38.00,
        "energyallocthreshold2": 159.00,
        "diapausedepththreshold0": 50,
        "diapausedepththreshold1": 500,
        "diapausedepththreshold2": 1200,
        "maxmatingdistance": 1000,
        "depthrange": None,  # None = PascalSimulation's own DEFAULT_DEPTHRANGE
    },
    "output": {
        "outputgrid": None,  # None = the scenario builder's own default
        "debug_variables": [],
        "verbose": False,
    },
}


class ConfigError(ValueError):
    """Raised for an invalid or incomplete run config."""


def validate(config):
    """Raise ConfigError if config is missing anything its run.scenario
    needs. Does not check that referenced files exist on disk - that's
    the reader's job at run time, not the config loader's."""
    scenario = config["run"]["scenario"]
    if scenario not in SCENARIOS:
        raise ConfigError(
            f"run.scenario must be one of {SCENARIOS}, got {scenario!r}"
        )
    if scenario == "advection_cmems_file" and not config["reader"]["cmems_file"]:
        raise ConfigError(
            "reader.cmems_file is required when run.scenario is "
            "'advection_cmems_file' (see hpc/download_cmems_data.py)"
        )
