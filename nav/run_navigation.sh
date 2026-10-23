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

# A run that ended without its shutdown (terminal closed, kill -9, power loss)
# leaves the direct-Livox journal behind, and livox_source then refuses to
# switch the lidar, so navigation waits forever for LiDAR data (2026-10-23).
# Restore the recorded original lidar parameters/service first; stop here if
# that fails instead of running on an unknown lidar configuration.
LIVOX_JOURNAL="${HOME}/.cache/cwc_navigation/livox_source.json"
if [[ -f "${LIVOX_JOURNAL}" ]]; then
  echo "Found an unfinished lidar switch from a previous run (${LIVOX_JOURNAL}); restoring the original lidar first..."
  if ! /usr/bin/python3 -m unitree_nav2_web.livox_source --interface "${G1_NETWORK_INTERFACE}" --restore; then
    echo "Lidar restore failed. Check that no other navigation/Livox driver is running, then run:" >&2
    echo "  /usr/bin/python3 -m unitree_nav2_web.livox_source --interface ${G1_NETWORK_INTERFACE} --restore" >&2
    exit 1
  fi
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
echo "CPU partition: navigation ${G1_NAV_CPUS:-0-3} (nice ${G1_NAV_NICE:-5}), camera ${G1_CAMERA_CPUS:-4-5}, body policy/bridge ${G1_EXECUTOR_CPUS:-6-7}."

# The Wi-Fi DDS bridge (~/dds_bridge, systemd user services) forwards the
# robot's 500 Hz topics and used ~1 CPU core even at rest, unpinned, across the
# camera and body-policy cores; 2026-10-21 pushes slowed the 50 Hz policy then.
# Its wifi_to_robot side would also publish onto the robot DDS network. Pause
# the active units for this run and start exactly those again on exit.
# G1_KEEP_DDS_BRIDGE=1 leaves them running.
DDS_BRIDGE_UNITS=(dds-robot-to-wifi-pub dds-robot-to-wifi-sub dds-wifi-to-robot-pub dds-wifi-to-robot-sub)
PAUSED_DDS_UNITS=()
restore_dds_bridge() {
  if ((${#PAUSED_DDS_UNITS[@]})); then
    echo "Restarting Wi-Fi DDS bridge: ${PAUSED_DDS_UNITS[*]}"
    systemctl --user start "${PAUSED_DDS_UNITS[@]}" || echo "Could not restart ${PAUSED_DDS_UNITS[*]}; run: systemctl --user start ${PAUSED_DDS_UNITS[*]}" >&2
  fi
}
if [[ "${G1_KEEP_DDS_BRIDGE:-0}" != "1" ]]; then
  for unit in "${DDS_BRIDGE_UNITS[@]}"; do
    if systemctl --user is-active --quiet "${unit}" 2>/dev/null; then
      PAUSED_DDS_UNITS+=("${unit}")
    fi
  done
  if ((${#PAUSED_DDS_UNITS[@]})); then
    trap restore_dds_bridge EXIT
    # Ctrl-C reaches ros2 launch directly; this shell waits for it, then restores.
    trap ':' INT TERM
    echo "Pausing Wi-Fi DDS bridge for navigation: ${PAUSED_DDS_UNITS[*]} (G1_KEEP_DDS_BRIDGE=1 keeps it)"
    systemctl --user stop "${PAUSED_DDS_UNITS[@]}"
  fi
fi

# The launch supervisor itself stays off the body-policy cluster; each node
# applies its own prefix and the executor re-pins itself to its cluster.
# Not exec: the EXIT trap above must run after launch has stopped.
status=0
taskset -c "${G1_NAV_CPUS:-0-3}" ros2 launch unitree_nav2_web g1_navigation.launch.py "$@" || status=$?
exit "${status}"
