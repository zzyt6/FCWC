#!/usr/bin/env bash
set -euo pipefail
VISER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VISER_BASE_PYTHON="${VISER_BASE_PYTHON:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
if [[ -z "$VISER_BASE_PYTHON" && -x "$HOME/miniconda3/envs/CWC/bin/python" ]]; then
  VISER_BASE_PYTHON="$HOME/miniconda3/envs/CWC/bin/python"
fi
if [[ -z "$VISER_BASE_PYTHON" ]]; then
  VISER_BASE_PYTHON=python3
  "$VISER_BASE_PYTHON" -m venv --system-site-packages "$VISER_DIR/.venv"
  VISER_BASE_PYTHON="$VISER_DIR/.venv/bin/python"
fi
"$VISER_BASE_PYTHON" -m pip install 'numpy<2' 'onnx==1.17.0' 'onnxruntime>=1.16,<2' 'viser==1.1.1' 'trimesh>=4' lxml networkx scipy pillow PyYAML
# Avoid trimesh[easy]'s unavailable embreex wheel on aarch64.
"$VISER_BASE_PYTHON" -m pip install --no-deps 'yourdfpy==0.0.60'
HYBRID_DEPLOY_PYTHON="$VISER_BASE_PYTHON" bash "$VISER_DIR/../G1Deploy_BalanceHomie/build.sh"
"$VISER_BASE_PYTHON" -c 'import onnx, onnxruntime, viser, yourdfpy; print("Balance/Homie + outer Viser ready")'
