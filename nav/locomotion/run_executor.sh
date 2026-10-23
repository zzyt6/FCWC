#!/usr/bin/env bash
# Start our own Balance/Homie locomotion as the executor of Nav2 /cmd_vel.
# Used by g1_navigation.launch.py; can also be started alone for testing:
#   bash nav/locomotion/run_executor.sh --mode sim
set -euo pipefail
LOCOMOTION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd -- "$LOCOMOTION_DIR/../.." && pwd)"

# Same interpreter selection as Viser/run.sh (needs onnx, onnxruntime, PyYAML).
if [[ -z "${G1_LOCOMOTION_PYTHON:-}" ]]; then
  if [[ -n "${VISER_PYTHON:-}" ]]; then
    G1_LOCOMOTION_PYTHON="$VISER_PYTHON"
  elif [[ -n "${HYBRID_DEPLOY_PYTHON:-}" ]]; then
    G1_LOCOMOTION_PYTHON="$HYBRID_DEPLOY_PYTHON"
  elif [[ -x "$HOME/miniconda3/envs/CWC/bin/python" ]]; then
    G1_LOCOMOTION_PYTHON="$HOME/miniconda3/envs/CWC/bin/python"
  elif [[ -x "$WORKSPACE/Viser/.venv/bin/python" ]]; then
    G1_LOCOMOTION_PYTHON="$WORKSPACE/Viser/.venv/bin/python"
  else
    echo "No Python with onnxruntime found: run bash $WORKSPACE/Viser/setup.sh or set G1_LOCOMOTION_PYTHON." >&2
    exit 1
  fi
fi

# The deployment's C++ bridge uses Unitree SDK2's own CycloneDDS on the given
# interface, exactly as when started from Viser. Do not leak the ROS 2 DDS
# configuration or Foxy's Python paths into it.
unset CYCLONEDDS_URI RMW_IMPLEMENTATION PYTHONPATH AMENT_PREFIX_PATH LD_LIBRARY_PATH
# The C++ bridge's RUNPATH selects the DDS libraries bundled with its SDK;
# inherited ROS library paths must not override that matching pair.
export PYTHONDONTWRITEBYTECODE=1
# CPU partition: the body policy and its 200 Hz C++ bridge (children of this
# process) own CPUs 6-7. The executor moves the cart camera to CPUs 4-5
# (G1_CAMERA_CPUS), so detection cannot starve the bridge's 100 ms heartbeat.
# g1_navigation.launch.py keeps LiDAR/FAST-LIO/Nav2 on CPUs 0-3.
EXECUTOR_CPUS="${G1_EXECUTOR_CPUS:-6-7}"
if command -v taskset >/dev/null 2>&1 && [[ -n "$EXECUTOR_CPUS" ]]; then
  exec taskset -c "$EXECUTOR_CPUS" "$G1_LOCOMOTION_PYTHON" -B "$LOCOMOTION_DIR/g1_locomotion_executor.py" "$@"
fi
exec "$G1_LOCOMOTION_PYTHON" -B "$LOCOMOTION_DIR/g1_locomotion_executor.py" "$@"
