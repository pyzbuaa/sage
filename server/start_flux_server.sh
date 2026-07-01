#!/usr/bin/env bash
# Legacy entry point — delegates to server/flux_server/start_flux_server.sh
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/flux_server/start_flux_server.sh" "$@"
