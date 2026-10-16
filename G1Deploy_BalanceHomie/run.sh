#!/usr/bin/env bash
set -euo pipefail
deploy_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$deploy_root"
# All browser entry points use the sibling Viser UI and its deferred acquire flow.
web_requested=false
for arg in "$@"; do
  if [[ "$arg" == --viser || "$arg" == simulate-web ]]; then web_requested=true; fi
done
if [[ "$web_requested" == true ]]; then
  web_mode="${1:-shadow}"
  if (( $# )); then shift; fi
  web_args=()
  case "$web_mode" in
    control|shadow)
      if (( $# == 0 )); then echo "Robot interface is required" >&2; exit 1; fi
      web_args+=(--interface "$1"); shift
      if [[ "$web_mode" == shadow ]]; then web_mode=observe; fi
      ;;
    simulate|simulate-web) web_mode=sim ;;
    *) echo "Unknown web mode: $web_mode" >&2; exit 1 ;;
  esac
  while (( $# )); do
    case "$1" in
      --viser) shift ;;
      --viser-host) web_args+=(--host "${2:?missing host}"); shift 2 ;;
      --viser-port) web_args+=(--port "${2:?missing port}"); shift 2 ;;
      *) web_args+=("$1"); shift ;;
    esac
  done
  exec bash "$deploy_root/../Viser/run.sh" --mode "$web_mode" "${web_args[@]}"
fi
deploy_python="${HYBRID_DEPLOY_PYTHON:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
if [[ -z "$deploy_python" || ! -x "$deploy_python" ]]; then
  echo "Activate the CWC Conda environment (conda activate CWC), or set HYBRID_DEPLOY_PYTHON to an existing environment's Python." >&2
  exit 1
fi
mode="${1:-shadow}"
if (( $# )); then shift; fi
case "$mode" in
  shadow|control)
    if (( $# == 0 )); then echo "Usage: bash run.sh $mode <robot-interface> [--duration N]" >&2; exit 1; fi
    bridge_args=(--interface "$1"); shift
    if [[ "$mode" == control ]]; then bridge_args+=(--control --release-mode); fi
    python_mode="$mode"
    ;;
  simulate)
    bridge_args=(--simulate --control --scripted --duration 9)
    python_mode=control
    ;;
  *) echo "Modes: shadow, control, simulate, simulate-web" >&2; exit 1 ;;
esac
export PYTHONPATH="$deploy_root/python${PYTHONPATH:+:$PYTHONPATH}"
deploy_arch="$(uname -m)"
if [[ "$deploy_arch" == arm64 ]]; then deploy_arch=aarch64; fi
export LD_LIBRARY_PATH="$deploy_root/vendor/unitree_sdk2/thirdparty/lib/$deploy_arch${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
HYBRID_DEPLOY_PYTHON="$deploy_python" bash build.sh
"$deploy_python" tools/verify_model.py
# Reject invalid policy options/config before starting the bridge or releasing a controller.
"$deploy_python" -m hybrid_deploy.run --check-only --mode "$python_mode" "$@"
if [[ "$mode" == control ]]; then
  echo "CONTROL: releasing factory controller; R2 calibrates, R2 again starts policy. L2 stops."
fi
./build/g1_control "${bridge_args[@]}" &
bridge_pid=$!
cleanup() {
  kill -TERM "$bridge_pid" 2>/dev/null || true
  wait "$bridge_pid" 2>/dev/null || true
}
trap cleanup EXIT
set +e
"$deploy_python" -m hybrid_deploy.run --mode "$python_mode" "$@"
result=$?
set -e
exit "$result"
