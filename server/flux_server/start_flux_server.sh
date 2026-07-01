#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-flux}"
CENTRAL_PORT="${FLUX_CENTRAL_PORT:-8090}"
WORKER_PORT="${FLUX_WORKER_PORT:-8091}"
# Physical GPU index on this machine (only this GPU is used).
FLUX_GPU_ID="${FLUX_GPU_ID:-0}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-9000}"

if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    CONDA_BIN="$CONDA_EXE"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
else
    echo "Error: Conda was not found. Install Conda outside this startup script." >&2
    exit 1
fi

if ! CONDA_ENV_PREFIX=$("$CONDA_BIN" run -n "$CONDA_ENV_NAME" \
    python -c 'import sys; print(sys.prefix)' 2>/dev/null); then
    echo "Error: Conda environment '$CONDA_ENV_NAME' does not exist." >&2
    echo "Run: cd $SCRIPT_DIR && ./setup_flux_env.sh" >&2
    exit 1
fi

PYTHON="$CONDA_ENV_PREFIX/bin/python"

SYSTEM_LIBSTDCPP=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
if [[ -f "$SYSTEM_LIBSTDCPP" ]]; then
    export LD_PRELOAD="$SYSTEM_LIBSTDCPP${LD_PRELOAD:+:$LD_PRELOAD}"
fi

if ! "$PYTHON" -c 'import diffusers, flask, flask_cors, psutil, requests, torch; from diffusers import FluxPipeline' >/dev/null 2>&1; then
    echo "Error: Flux dependencies missing or incompatible (diffusers must be <0.33 with torch 2.4)." >&2
    echo "Run: cd $SCRIPT_DIR && ./setup_flux_env.sh" >&2
    exit 1
fi

cd "$SCRIPT_DIR"

GPU_COUNT=$("$PYTHON" -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
if (( GPU_COUNT == 0 )); then
    echo "Error: Flux server requires at least one NVIDIA GPU." >&2
    exit 1
fi

if (( FLUX_GPU_ID < 0 || FLUX_GPU_ID >= GPU_COUNT )); then
    echo "Error: FLUX_GPU_ID=$FLUX_GPU_ID is out of range (visible GPUs: 0..$((GPU_COUNT - 1)))." >&2
    exit 1
fi

WORKER_PID=""
cleanup() {
    if [[ -n "$WORKER_PID" ]]; then
        kill "$WORKER_PID" 2>/dev/null || true
        wait "$WORKER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Using GPU $FLUX_GPU_ID only (set FLUX_GPU_ID to change)."
echo "Starting Flux worker on port $WORKER_PORT..."
CUDA_VISIBLE_DEVICES=$FLUX_GPU_ID "$PYTHON" worker_server.py --port "$WORKER_PORT" --gpu "$FLUX_GPU_ID" &
WORKER_PID=$!

WORKER_URL="http://localhost:$WORKER_PORT"
echo "Waiting for worker to load model and become ready (timeout ${STARTUP_TIMEOUT}s)..."
DEADLINE=$((SECONDS + STARTUP_TIMEOUT))

until "$PYTHON" -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$WORKER_PORT/health', timeout=2).read()" \
    >/dev/null 2>&1; do
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
        echo "Error: worker exited before becoming ready." >&2
        exit 1
    fi
    if (( SECONDS >= DEADLINE )); then
        echo "Error: worker did not become ready within ${STARTUP_TIMEOUT}s." >&2
        exit 1
    fi
    sleep 3
done
echo "Worker is ready on port $WORKER_PORT."

echo ""
echo "Starting Flux central distributor on port $CENTRAL_PORT..."
echo "Set server/key.json FLUX_SERVER_URL to: http://<this-host>:$CENTRAL_PORT"
echo ""

"$PYTHON" central_server.py "$WORKER_URL" --port "$CENTRAL_PORT"
