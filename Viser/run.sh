#!/usr/bin/env bash
set -euo pipefail
VISER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${VISER_PYTHON:-}" ]]; then
  if [[ -n "${HYBRID_DEPLOY_PYTHON:-}" ]]; then
    VISER_PYTHON="$HYBRID_DEPLOY_PYTHON"
  elif [[ -x "${CONDA_PREFIX:-/nonexistent}/bin/python" ]] && "$CONDA_PREFIX/bin/python" -c 'import onnx, onnxruntime, viser, yourdfpy' >/dev/null 2>&1; then
    VISER_PYTHON="$CONDA_PREFIX/bin/python"
  elif [[ -x "$HOME/miniconda3/envs/CWC/bin/python" ]]; then
    VISER_PYTHON="$HOME/miniconda3/envs/CWC/bin/python"
  elif [[ -x "$VISER_DIR/.venv/bin/python" ]]; then
    VISER_PYTHON="$VISER_DIR/.venv/bin/python"
  else
    echo "Run bash $VISER_DIR/setup.sh or set VISER_PYTHON." >&2
    exit 1
  fi
fi
export PYTHONDONTWRITEBYTECODE=1
unset CYCLONEDDS_URI RMW_IMPLEMENTATION PYTHONPATH AMENT_PREFIX_PATH LD_LIBRARY_PATH
# A bare `./Viser/run.sh` is the real-robot web controller. Explicit flags
# remain overridable for offline and read-only modes.
args=("$@")
has_mode=0
has_interface=0
has_port=0
for arg in "${args[@]}"; do
  case "$arg" in
    --mode|--mode=*) has_mode=1 ;;
    --interface|--interface=*) has_interface=1 ;;
    --port|--port=*) has_port=1 ;;
  esac
done
if (( ! has_mode )); then args+=(--mode control); fi
if (( ! has_interface )); then args+=(--interface eth0); fi
if (( ! has_port )); then args+=(--port 8086); fi
exec "$VISER_PYTHON" -B "$VISER_DIR/Vender/app.py" "${args[@]}"
