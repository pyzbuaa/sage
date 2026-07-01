#!/usr/bin/env bash
# Forward to the maintained TRELLIS launcher (default: GPUs 4,5,6,7, 4 workers).
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/trellis_server/start_trellis_server.sh" "$@"
