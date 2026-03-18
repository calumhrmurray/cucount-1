#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
venv_dir="${VENV_DIR:-$repo_dir/.venv-cc-lyon}"
output_dir="${OUTPUT_DIR:-$repo_dir/examples/output/first_steps}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is required to build cucount on cc-lyon."
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required to build cucount on cc-lyon."
  exit 1
fi

"$python_bin" -m venv "$venv_dir"
source "$venv_dir/bin/activate"

python -m pip install --upgrade pip wheel
python -m pip install numpy matplotlib astropy pybind11 scikit-build-core

# Build the numpy backend only for this first-step validation workflow.
BUILD_FFI_CUCOUNT=0 python -m pip install --no-build-isolation -e "$repo_dir"

mkdir -p "$output_dir"

python "$repo_dir/examples/angular_correlation_comparison.py" \
  --output-dir "$output_dir" \
  --output-prefix "desi_unions_observed" \
  --max-random-files "${MAX_RANDOM_FILES_SKY:-4}" \
  --max-sources "${MAX_SOURCES:-5000000}" \
  --nthreads "${NTHREADS:-1}"
