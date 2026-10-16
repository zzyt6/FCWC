#!/usr/bin/env bash
set -euo pipefail
deploy_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$deploy_root"
deploy_python="${HYBRID_DEPLOY_PYTHON:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
if [[ -z "$deploy_python" || ! -x "$deploy_python" ]]; then
  echo "Activate the CWC Conda environment (conda activate CWC), or set HYBRID_DEPLOY_PYTHON to an existing environment's Python." >&2
  exit 1
fi
echo "Installing into existing environment: $deploy_python"
"$deploy_python" -m pip install -r requirements.txt
"$deploy_python" tools/check_platform.py
HYBRID_DEPLOY_PYTHON="$deploy_python" bash build.sh
"$deploy_python" tools/verify_model.py
echo "Installed. Observe: bash run.sh shadow <robot-interface>"
echo "Motor control: bash run.sh control <robot-interface>"
