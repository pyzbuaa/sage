#!/usr/bin/env bash

# One-time setup: create the flux Conda environment (if missing) and install dependencies.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-flux}"

if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    CONDA_BIN="$CONDA_EXE"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
else
    echo "Error: Conda was not found. Install Conda before running this script." >&2
    exit 1
fi

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
    echo "Creating Conda environment '$CONDA_ENV_NAME' (Python 3.10)..."
    "$CONDA_BIN" create -n "$CONDA_ENV_NAME" python=3.10 -y
fi

if ! CONDA_ENV_PREFIX=$("$CONDA_BIN" run -n "$CONDA_ENV_NAME" \
    python -c 'import sys; print(sys.prefix)' 2>/dev/null); then
    echo "Error: Conda environment '$CONDA_ENV_NAME' is unavailable." >&2
    exit 1
fi

PYTHON="$CONDA_ENV_PREFIX/bin/python"
PY_MINOR=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PY_MINOR" != "3.10" ]]; then
    echo "Error: Conda environment '$CONDA_ENV_NAME' must use Python 3.10 (found $PY_MINOR)." >&2
    exit 1
fi

"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install \
    torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124
"$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements-flux.txt"

if ! "$PYTHON" -c "from diffusers import FluxPipeline" 2>/dev/null; then
    echo "Error: diffusers import failed after install. Check torch/diffusers versions." >&2
    exit 1
fi

echo ""
echo "Flux environment is ready: $CONDA_ENV_NAME ($CONDA_ENV_PREFIX)"
echo ""
echo "Before starting the server:"
echo "  1. Accept the model license at https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev"
echo "  2. huggingface-cli login   # if not already logged in"
echo "  3. cd $SCRIPT_DIR && ./start_flux_server.sh"
echo "     # optional: FLUX_GPU_ID=1 ./start_flux_server.sh"
