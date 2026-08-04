#!/bin/bash
# Run inside the container (via `apptainer exec --writable-tmpfs ...`,
# see container.def's %help / BENCHMARKING.md) after bind-mounting this
# repo to /pascal and opendrift_pascal to /opendrift.
#
# Editable-installs both actively-developed local packages - not baked
# into the image itself (see container.def) - then execs the given
# command. Needs --writable-tmpfs (or a sandbox/writable container):
# a plain .sif image's filesystem is read-only, and pip install needs to
# write into the conda env's site-packages.
set -euo pipefail

if [ ! -d /opendrift ]; then
    echo "run_in_container.sh: /opendrift is not mounted - bind-mount" \
         "opendrift_pascal there (see container.def's %help)" >&2
    exit 1
fi
if [ ! -d /pascal ]; then
    echo "run_in_container.sh: /pascal is not mounted - bind-mount this" \
         "repo there (see container.def's %help)" >&2
    exit 1
fi

pip install --no-deps -q -e /opendrift
pip install --no-deps -q -e /pascal

exec "$@"
