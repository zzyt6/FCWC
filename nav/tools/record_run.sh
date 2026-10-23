#!/usr/bin/env bash
# Record a real navigation run (second terminal, while run_navigation.sh runs):
#   bash nav/tools/record_run.sh [label]      # Ctrl-C after the run -> nav/log/runs/<time>_<label>/
# Same ROS 2 / DDS environment as run_navigation.sh so the topics are visible.
set -eo pipefail
NAV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/foxy/setup.bash
source "${NAV_DIR}/install/setup.bash"
set -u
NAV_DDS_LIB="${UNITREE_CYCLONEDDS_LIB:-${CYCLONEDDS_HOME:-$HOME/cyclonedds/install}/lib}"
export LD_LIBRARY_PATH="${NAV_DDS_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PATH="/usr/bin:${PATH}"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="${UNITREE_DDS_CONFIG:-/home/unitree/cyclonedds_ws/cyclonedds_eth0.xml}"
exec /usr/bin/python3 "${NAV_DIR}/tools/record_run.py" "$@"
