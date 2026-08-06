"""Tests for the YAML run-config system (config/schema.py, config/loader.py)
and its use in building a scenario (scenarios/builders.py).
"""

import numpy as np
import pytest
import yaml

from loader import ConfigError, load_config
from builders import build_global_settings_from_config, build_scenario_from_config
from schema import DEFAULTS


def write_yaml(tmp_path, data, name="run.yaml"):
    path = tmp_path / name
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return path


def test_minimal_config_fills_in_defaults(tmp_path):
    path = write_yaml(tmp_path, {"run": {"scenario": "1d"}})
    config = load_config(path)

    assert config["run"]["scenario"] == "1d"
    assert config["population"]["n_super_individuals"] == (
        DEFAULTS["population"]["n_super_individuals"]
    )
    assert config["biology"]["ageceiling"] == DEFAULTS["biology"]["ageceiling"]


def test_partial_override_leaves_rest_of_section_at_default(tmp_path):
    path = write_yaml(tmp_path, {
        "run": {"scenario": "1d"},
        "population": {"n_super_individuals": 5},
    })
    config = load_config(path)

    assert config["population"]["n_super_individuals"] == 5
    # untouched fields in the same section keep their default
    assert config["population"]["n_virtual_per_super"] == (
        DEFAULTS["population"]["n_virtual_per_super"]
    )


def test_cli_overrides_apply_on_top_of_the_file(tmp_path):
    path = write_yaml(tmp_path, {"run": {"scenario": "1d"}})
    config = load_config(path, overrides=[
        "population.n_super_individuals=7",
        "biology.stochastic=false",
    ])

    assert config["population"]["n_super_individuals"] == 7
    assert config["biology"]["stochastic"] is False


def test_advection_cmems_file_without_cmems_file_is_rejected(tmp_path):
    path = write_yaml(tmp_path, {"run": {"scenario": "advection_cmems_file"}})
    with pytest.raises(ConfigError):
        load_config(path)


def test_unknown_scenario_is_rejected(tmp_path):
    path = write_yaml(tmp_path, {"run": {"scenario": "not_a_real_scenario"}})
    with pytest.raises(ConfigError):
        load_config(path)


def test_build_global_settings_from_config_matches_hardcoded_defaults(tmp_path):
    """The whole point of build_global_settings_from_config() is that an
    unmodified config reproduces build_global_settings()'s previously
    hardcoded values exactly - only now every field is overridable."""
    from builders import build_global_settings

    path = write_yaml(tmp_path, {"run": {"scenario": "1d"}})
    config = load_config(path)

    from_config = build_global_settings_from_config(config["biology"])
    hardcoded = build_global_settings(stochastic=True)

    assert from_config.keys() == hardcoded.keys()
    for key in hardcoded:
        np.testing.assert_array_equal(from_config[key], hardcoded[key])


def test_build_scenario_from_config_produces_a_working_1d_run(tmp_path, monkeypatch):
    from pascal.coupler import Pascal1D

    monkeypatch.chdir(tmp_path)
    path = write_yaml(tmp_path, {
        "run": {"scenario": "1d", "name": "config_smoke"},
        "population": {"n_super_individuals": 10, "seeding_rate": 5},
        "time": {"duration_years": 0.02},
    })
    config = load_config(path)
    kwargs = build_scenario_from_config(config)

    sim = Pascal1D(**kwargs)
    sim.run()

    assert sim.population_size() > 0


def test_build_scenario_from_config_overrides_biology(tmp_path, monkeypatch):
    """A biology override (ageceiling) actually reaches global_settings,
    not just build_global_settings()'s old hardcoded 2180."""
    monkeypatch.chdir(tmp_path)
    path = write_yaml(tmp_path, {
        "run": {"scenario": "1d"},
        "biology": {"ageceiling": 42},
    })
    config = load_config(path)
    kwargs = build_scenario_from_config(config)

    assert kwargs["global_settings"]["ageceiling"] == 42
