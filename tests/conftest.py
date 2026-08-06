import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The repo tracks an uninitialized `opendrift` git submodule at its root
# (empty directory, no __init__.py, pointing at upstream OpenDrift/opendrift
# - not the PASCAL-specific opendrift_pascal fork this project actually
# uses). If the repo root ever ends up on sys.path - which `python -m`,
# `python -c`/`python -`, and some IDE "run file" configurations all do in
# different ways, there's no single invocation style to defend against -
# Python's import system can resolve `import opendrift` to that empty
# directory as a broken namespace package instead of the real,
# pip-installed opendrift_pascal package. The failure is silent until
# something actually instantiates a PascalDrift/OceanDrift, which then
# dies deep inside OpenDrift's BaseModel.__init__ with
# "module 'opendrift' has no attribute '__version__'". Purge the repo root
# (and '') from sys.path defensively rather than relying on invocation
# order.
for _bad in ("", str(REPO_ROOT)):
    while _bad in sys.path:
        sys.path.remove(_bad)

# `pip install -e .` (setup.py already declares this package) makes the
# `pascal` package (pascal.coupler/pascal.individual/etc.) importable
# without needing the repo root on sys.path at all; only benchmarks/ needs
# adding here since it isn't part of that install.
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

SCENARIOS_DIR = REPO_ROOT / "scenarios"
if str(SCENARIOS_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIOS_DIR))

CONFIG_DIR = REPO_ROOT / "config"
if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))
