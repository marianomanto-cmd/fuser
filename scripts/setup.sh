#!/usr/bin/env bash
# ============================================================================
# Fuser · instalación local automática (Linux / macOS)
#
#   bash scripts/setup.sh          # instala dependencias GPU (CUDA)
#   bash scripts/setup.sh --cpu    # instala versión CPU (probar la UI / sin GPU)
#
# Crea un entorno virtual en .venv, instala dependencias, descarga los modelos
# recomendados y ejecuta el diagnóstico de entorno.
# ============================================================================
set -e
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
REQ="requirements.txt"
if [ "$1" = "--cpu" ]; then
    REQ="requirements-cpu.txt"
    echo ">> Modo CPU: usando $REQ"
fi

echo ">> Creando entorno virtual en .venv ..."
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo ">> Actualizando pip ..."
python -m pip install --upgrade pip

echo ">> Instalando dependencias ($REQ) ..."
pip install -r "$REQ"

echo ">> Descargando modelos recomendados (inswapper_128 + gfpgan_1.4) ..."
python scripts/download_models.py || echo "(la descarga se reintentará en el primer uso)"

echo ">> Diagnóstico de entorno:"
python scripts/check_env.py || true

echo ""
echo "============================================================"
echo " Listo. Para usar la app:"
echo "   source .venv/bin/activate"
echo "   python app.py"
echo " Abre http://127.0.0.1:7860"
echo "============================================================"
