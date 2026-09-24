#!/usr/bin/env bash
# Subordinate implementation; complete experiments enter through run.sh.
set -euo pipefail
exec "${PYTHON:-python3}" -m src.sir_cli "$@"
