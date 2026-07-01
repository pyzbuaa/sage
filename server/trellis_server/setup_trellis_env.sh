#!/usr/bin/env bash

# One-time dependency setup for an existing Conda environment. This script
# intentionally does not install Conda or create an environment.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRELLIS_DIR="${TRELLIS_DIR:-$SCRIPT_DIR/TRELLIS}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-trellis}"

if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    CONDA_BIN="$CONDA_EXE"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
else
    echo "Error: Conda was not found. Install Conda before running this script." >&2
    exit 1
fi

if ! CONDA_ENV_PREFIX=$("$CONDA_BIN" run -n "$CONDA_ENV_NAME" \
    python -c 'import sys; print(sys.prefix)' 2>/dev/null); then
    echo "Error: Conda environment '$CONDA_ENV_NAME' does not exist or is unavailable." >&2
    echo "Create it outside this script, then run this setup again." >&2
    exit 1
fi

PYTHON="$CONDA_ENV_PREFIX/bin/python"
if [[ "$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.10" ]]; then
    echo "Error: Conda environment '$CONDA_ENV_NAME' must use Python 3.10." >&2
    exit 1
fi

if [[ ! -d "$TRELLIS_DIR/.git" ]]; then
    git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git "$TRELLIS_DIR"
else
    git -C "$TRELLIS_DIR" submodule update --init --recursive
fi

"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124
"$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements-trellis.txt"
"$PYTHON" -m pip install kaolin \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
"$PYTHON" -m pip install \
    git+https://github.com/NVlabs/nvdiffrast.git \
    --no-build-isolation
"$PYTHON" -m pip install \
    "git+https://github.com/autonomousvision/mip-splatting.git#subdirectory=submodules/diff-gaussian-rasterization/" \
    --no-build-isolation

echo "TRELLIS environment is ready: $CONDA_ENV_NAME ($CONDA_ENV_PREFIX)"
