#!/usr/bin/env bash
# Run FB22 model grid first, then CP25 surface brightness grid.
# Expectation: you've already activated the desired conda/venv in this shell.
# Usage:
#   ./run.sh [--logdir logs] [--dry-run] [--skip-fb22] [--skip-sb]

set -Eeuo pipefail

LOGDIR="logs"
DRYRUN=0
SKIP_FB22=0
SKIP_CP25=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --logdir)     LOGDIR="$2"; shift 2;;
    --dry-run|--dryrun) DRYRUN=1; shift;;
    --skip-fb22)  SKIP_FB22=1; shift;;
    --skip-sb)  SKIP_SB=1; shift;;
    -h|--help)    sed -n '1,25p' "$0"; exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

# Resolve script directory so relative paths (../*) work.
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Confirm python availability from current environment
if ! command -v python >/dev/null 2>&1; then
  echo "Error: 'python' not found on PATH. Activate your conda/venv first." >&2
  exit 1
fi
echo ">>> Using python: $(command -v python)"
echo ">>> Python version: $(python -V 2>&1)"

mkdir -p "$LOGDIR"
ts() { date +"%Y-%m-%d_%H-%M-%S"; }

run_step() {
  local name="$1"; shift
  local cmd=( "$@" )
  local logfile="${LOGDIR}/${name}_$(ts).log"
  echo ">>> [$name] Starting: ${cmd[*]}"
  echo ">>> Logging to: $logfile"
  if [[ "$DRYRUN" -eq 1 ]]; then
    echo ">>> [dry-run] Skipping execution."
    return 0
  fi
  if "${cmd[@]}" 2>&1 | tee "$logfile"; then
    echo ">>> [$name] Completed OK."
  else
    local ec=$?
    echo "!!! [$name] FAILED with exit code $ec. See $logfile" >&2
    return "$ec"
  fi
}

# Sanity checks: scripts exist
[[ -f fb22_model_grid.py ]] || { echo "Missing fb22_model_grid.py" >&2; exit 1; }
[[ -f sb_model_grid.py ]] || { echo "Missing sb_model_grid.py" >&2; exit 1; }

START_ALL=$(ts)

if [[ "$SKIP_FB22" -eq 0 ]]; then
  run_step "fb22_model_grid" python fb22_model_grid.py
fi

if [[ "$SKIP_CP25" -eq 0 ]]; then
  run_step "sb_model_grid" python sb_model_grid.py
fi

echo ">>> All steps finished. Started at: $START_ALL ; Ended at: $(ts)"
