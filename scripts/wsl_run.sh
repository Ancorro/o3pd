#!/usr/bin/env bash
# Run a python module from the repo in the WSL venv, detached, logging to a file.
# usage: bash scripts/wsl_run.sh LOGFILE -m o3pd.evaluate ...
cd "$(dirname "$0")/.."
log=$1; shift
PY=${O3PD_PY:-$HOME/venvs/o3pd/bin/python}
setsid nohup env PYTHONUNBUFFERED=1 "$PY" "$@" > "$log" 2>&1 < /dev/null &
echo "pid $!"
