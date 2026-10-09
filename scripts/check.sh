#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
"${PYTHON:-python3}" -m unittest discover -s tools/tests -v
