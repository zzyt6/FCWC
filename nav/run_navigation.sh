#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

source /opt/ros/foxy/setup.bash
source "${ROOT_DIR}/install/setup.bash"
set -u

# Sourcing Foxy puts CycloneDDS 0.7 first. It crashes while parsing discovery
# traffic on this robot; restore the installed 0.10 runtime after all setups.
NAV_DDS_LIB="${UNITREE_CYCLONEDDS_LIB:-${CYCLONEDDS_HOME:-$HOME/cyclonedds/install}/lib}"
if [[ ! -f "${NAV_DDS_LIB}/libddsc.so.0" ]]; then
  echo "CycloneDDS runtime missing: ${NAV_DDS_LIB}/libddsc.so.0; set UNITREE_CYCLONEDDS_LIB." >&2
  exit 1
fi
export LD_LIBRARY_PATH="${NAV_DDS_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# ROS Python nodes use Ubuntu's interpreter even from an activated Viser venv.
export PATH="/usr/bin:${PATH}"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="${UNITREE_DDS_CONFIG:-/home/unitree/cyclonedds_ws/cyclonedds_eth0.xml}"
export G1_NETWORK_INTERFACE="${G1_NETWORK_INTERFACE:-eth0}"

DEFAULT_PCD="${ROOT_DIR}/maps/cf.pcd"
DEFAULT_MAP_DIR="${ROOT_DIR}/maps/generated/cf"
if [[ ! -f "${DEFAULT_MAP_DIR}/map.yaml" ]]; then
  if [[ ! -f "${DEFAULT_PCD}" ]]; then
    echo "Default map ${DEFAULT_PCD} not found. Copy a PCD into maps/ or pass map_yaml:=<file>." >&2
    exit 1
  fi
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
fi

HTTP_PORT="8080"
ROSBRIDGE_PORT="9090"
for argument in "$@"; do
  case "${argument}" in
    http_port:=*) HTTP_PORT="${argument#http_port:=}" ;;
    rosbridge_port:=*) ROSBRIDGE_PORT="${argument#rosbridge_port:=}" ;;
  esac
done

WIFI_IP="$(ip -4 -o addr show dev wlan0 2>/dev/null | awk '{split($4, a, "/"); print a[1]; exit}')"
if [[ -n "${WIFI_IP}" ]]; then
  echo "G1 Canvas navigation: http://${WIFI_IP}:${HTTP_PORT}"
  echo "rosbridge WebSocket: ws://${WIFI_IP}:${ROSBRIDGE_PORT}"
else
  echo "G1 Canvas navigation: http://<robot-wifi-ip>:${HTTP_PORT}"
fi
echo "Unitree DDS / locomotion executor: ${G1_NETWORK_INTERFACE} (Balance/Homie, nav/locomotion)"
echo "Do not run Viser/run.sh at the same time: both own the low-level bridge."
echo "Motion gate starts DISABLED. Load map, localize and start the policy before enabling it."

exec ros2 launch unitree_nav2_web g1_navigation.launch.py "$@"
