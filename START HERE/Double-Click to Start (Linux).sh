#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../DO NOT TOUCH" && pwd)"
cd "$SCRIPT_DIR"
# Run through bash: a downloaded ZIP has no executable bit, so ./run.sh
# would fail with "Permission denied".
exec bash ./run.sh
