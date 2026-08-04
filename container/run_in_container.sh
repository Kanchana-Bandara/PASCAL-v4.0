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

# Fail loudly here rather than let `pip install` silently degrade: without
# --writable-tmpfs (or --writable) at `apptainer run/exec` time, the conda
# env below is read-only, and pip's default behavior on a permission
# failure is to *silently* retry as a --user install instead of erroring.
# Found the hard way 2026-08-04: because Apptainer bind-mounts $HOME from
# the host by default, that silent fallback writes editable-install
# metadata straight into the *host's* real ~/.local/lib/pythonX.Y/
# site-packages, hardcoded with container-internal paths - which breaks
# `import opendrift`/`import coupler` on the host outside the container
# entirely, for anyone who forgets the flag. See BENCHMARKING.md.
site_packages="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [ ! -w "$site_packages" ]; then
    echo "run_in_container.sh: $site_packages is not writable - re-run with" \
         "--writable-tmpfs (see container.def's %help/%apphelp). Refusing to" \
         "continue: pip would otherwise silently fall back to a --user" \
         "install, which - because \$HOME is bind-mounted from the host by" \
         "default - corrupts the host's real Python environment instead of" \
         "failing visibly." >&2
    exit 1
fi

pip install --no-deps -q -e /opendrift
pip install --no-deps -q -e /pascal

exec "$@"
