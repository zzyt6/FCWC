#!/usr/bin/env bash
set -euo pipefail
deploy_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
deploy_python="${HYBRID_DEPLOY_PYTHON:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
if [[ -z "$deploy_python" || ! -x "$deploy_python" ]]; then
  echo "Activate the CWC Conda environment (conda activate CWC), or set HYBRID_DEPLOY_PYTHON to an existing environment's Python." >&2
  exit 1
fi
cmake -S "$deploy_root" -B "$deploy_root/build" -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$deploy_python"
cmake --build "$deploy_root/build" -j "${HYBRID_BUILD_JOBS:-2}"
# Ubuntu 20.04 ships CTest 3.16, which does not support --test-dir.
(cd "$deploy_root/build" && ctest --output-on-failure)
