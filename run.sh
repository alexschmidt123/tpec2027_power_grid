#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-python3}" -m src.online_cli "$@"
