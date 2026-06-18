#!/usr/bin/env bash
# Instala el motor opcional FaceFusion (Linux/macOS). Wrapper de install_facefusion.py.
#   bash scripts/install_facefusion.sh
set -e
cd "$(dirname "$0")/.."
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
python scripts/install_facefusion.py
