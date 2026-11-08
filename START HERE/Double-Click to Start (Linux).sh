#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../DO NOT TOUCH" && pwd)"
cd "$SCRIPT_DIR"
exec ./run.sh
