#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

source /opt/ros/foxy/setup.bash
set -u

if ! python3 -c 'import viser' >/dev/null 2>&1; then
  python3 -m pip install --user -r requirements.txt
fi

colcon build \
  --symlink-install \
  --cmake-args -DROS_EDITION=ROS2 -DCMAKE_BUILD_TYPE=Release

DEFAULT_PCD="${ROOT_DIR}/maps/cf.pcd"
DEFAULT_MAP_DIR="${ROOT_DIR}/maps/generated/cf"
if [[ -f "${DEFAULT_PCD}" ]]; then
  mkdir -p "${DEFAULT_MAP_DIR}"
  "${ROOT_DIR}/install/unitree_nav2_core/lib/unitree_nav2_core/pcd_to_nav_map" \
    --input "${DEFAULT_PCD}" \
    --output-dir "${DEFAULT_MAP_DIR}" \
    --resolution 0.03 \
    --ground-z auto \
    --obstacle-min-height 0.15 \
    --obstacle-max-height 1.80 \
    --ground-band 0.15 \
    --obstacle-radius 0.04 \
    --free-radius 0.15
else
  echo "Default map ${DEFAULT_PCD} not found; skip pre-generating the Nav2 map."
fi

echo "Build complete. Mapping: ./run_mapping.sh  Navigation: ./run_navigation.sh"
