#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

source /opt/ros/foxy/setup.bash
source "${ROOT_DIR}/install/setup.bash"
set -u

# Keep the robot-compatible CycloneDDS runtime ahead of Foxy's old 0.7 library.
NAV_DDS_LIB="${UNITREE_CYCLONEDDS_LIB:-${CYCLONEDDS_HOME:-$HOME/cyclonedds/install}/lib}"
if [[ ! -f "${NAV_DDS_LIB}/libddsc.so.0" ]]; then
  echo "CycloneDDS runtime missing: ${NAV_DDS_LIB}/libddsc.so.0; set UNITREE_CYCLONEDDS_LIB." >&2
  exit 1
fi
export LD_LIBRARY_PATH="${NAV_DDS_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PATH="/usr/bin:${PATH}"

# Unitree's lidar service is a bare-DDS publisher on the robot's wired subnet.
# Viser remains bound to 0.0.0.0 and is accessed from a remote host via wlan0.
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="${UNITREE_DDS_CONFIG:-/home/unitree/cyclonedds_ws/cyclonedds_eth0.xml}"

mkdir -p "${ROOT_DIR}/maps"

VISER_PORT="8080"
for argument in "$@"; do
  if [[ "${argument}" == viser_port:=* ]]; then
    VISER_PORT="${argument#viser_port:=}"
  fi
done

WIFI_IP="$(ip -4 -o addr show dev wlan0 2>/dev/null | awk '{split($4, a, "/"); print a[1]; exit}')"
if [[ -n "${WIFI_IP}" ]]; then
  echo "Viser Wi-Fi URL: http://${WIFI_IP}:${VISER_PORT}"
else
  echo "Viser URL: http://<robot-wifi-ip>:${VISER_PORT}"
fi
echo "Unitree DDS: eth0 via ${CYCLONEDDS_URI}"

exec ros2 launch unitree_fastlio_baseline mid360_mapping_viser.launch.py "$@"
