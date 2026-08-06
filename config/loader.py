"""Load a PASCAL run config: YAML file -> defaults-filled, validated dict.

Usage:
    from loader import load_config
    config = load_config("config/runs/advection_hpc_barents.yaml",
                          overrides=["population.n_super_individuals=20"])
"""

import copy

import yaml

from schema import DEFAULTS, ConfigError, validate  # noqa: F401 (ConfigError re-exported)


def _deep_merge(base, override):
    """Merge override onto a deep copy of base, recursing into nested
    dicts but replacing (not merging) lists/scalars - so a config only
    needs to state the fields it actually changes."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_dotted(config, dotted_key, value):
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def apply_overrides(config, overrides):
    """Apply a list of "a.b.c=value" strings on top of config, in place.
    value is parsed with yaml.safe_load so plain scalars/lists/null come
    through as the right Python type (e.g. "20" -> int, "null" -> None,
    "[1,2]" -> list) without a bespoke parser - same rules as the YAML
    file itself.
    """
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(
                f"--override expects key.path=value, got {item!r}"
            )
        dotted_key, raw_value = item.split("=", 1)
        _set_dotted(config, dotted_key.strip(), yaml.safe_load(raw_value))
    return config


def load_config(path, overrides=None):
    """Read a YAML run config, fill in unset fields from DEFAULTS, apply
    any --override entries, validate, and return the final nested dict."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    config = _deep_merge(DEFAULTS, raw)
    apply_overrides(config, overrides)
    validate(config)
    return config
