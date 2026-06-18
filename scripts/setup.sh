#!/usr/bin/env bash
# ============================================================================
# Fuser · instalación local automática (Linux / macOS)
#
#   bash scripts/setup.sh                 # GPU (CUDA) + FaceFusion (alta calidad)
#   bash scripts/setup.sh --cpu           # versión CPU (probar UI; sin FaceFusion)
#   bash scripts/setup.sh --no-facefusion # GPU pero sin instalar FaceFusion
#
# Crea .venv, instala dependencias, descarga modelos, instala el motor FaceFusion
# y ejecuta el diagnóstico. Deja TODO listo: solo queda `python app.py`.
# ============================================================================
set -e
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
REQ="requirements.txt"
WITH_FF=1
for arg in "$@"; do
    case "$arg" in
        --cpu) REQ="requirements-cpu.txt"; WITH_FF=0; echo ">> Modo CPU";;
        --no-facefusion) WITH_FF=0;;
    esac
done

echo ">> Creando entorno virtual en .venv ..."
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo ">> Actualizando pip ..."
python -m pip install --upgrade pip

echo ">> Instalando dependencias ($REQ) ..."
pip install -r "$REQ"

echo ">> Descargando modelos recomendados ..."
python scripts/download_models.py || echo "(se reintentará en el primer uso)"

if [ "$WITH_FF" = "1" ]; then
    echo ">> Instalando el motor FaceFusion (alta calidad) ..."
    python scripts/install_facefusion.py || echo "(FaceFusion opcional: continúo; puedes reintentar luego)"
fi

echo ">> Diagnóstico de entorno:"
python scripts/check_env.py || true

echo ""
echo "============================================================"
echo " Listo. Para usar la app:"
echo "   source .venv/bin/activate"
echo "   python app.py        →  http://127.0.0.1:7860"
echo "============================================================"
