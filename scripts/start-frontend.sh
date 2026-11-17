#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/../frontend"
export NEXT_TELEMETRY_DISABLED=1
exec npm run dev
