#!/usr/bin/env bash
# ============================================================================
# Fuser · desinstalación completa (Linux / macOS)
#
#   bash scripts/uninstall.sh                 # desinstala y CONSERVA tus datos
#   bash scripts/uninstall.sh --dry-run       # solo muestra qué borraría
#   bash scripts/uninstall.sh --purge-data    # borra también caras/salidas/entrenamientos
#   bash scripts/uninstall.sh --remove-repo   # borra también esta carpeta
#
# Usa el Python del sistema a propósito: el .venv es una de las cosas que se
# borran, así que no se activa ni se depende de él.
# ============================================================================
set -e

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$PROJ/scripts/uninstall.py"

# Situarse FUERA del proyecto: no se puede borrar el directorio actual.
cd "$(dirname "$PROJ")"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    elif command -v python >/dev/null 2>&1; then
        PYTHON=python
    else
        echo "[ERROR] No encuentro Python. Instalalo, o borrá a mano: $PROJ" >&2
        exit 1
    fi
fi

exec "$PYTHON" "$SCRIPT" "$@"
