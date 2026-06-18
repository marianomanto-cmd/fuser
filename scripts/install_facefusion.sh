#!/usr/bin/env bash
# ============================================================================
# Fuser · instalación del motor OPCIONAL FaceFusion (Linux / macOS)
#
#   bash scripts/install_facefusion.sh
#
# - Clona FaceFusion en  vendor/facefusion  (Fuser lo auto-detecta ahí).
# - Instala sus dependencias en el MISMO entorno virtual (.venv) que Fuser.
# - Restaura los pines críticos de Fuser (gradio 5, numpy<2) por si FaceFusion
#   los cambia.
# - Verifica que 'import facefusion' funciona desde Fuser.
#
# Requiere: git y, idealmente, el .venv de Fuser ya creado (scripts/setup.sh).
# ============================================================================
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo ">> Usando entorno .venv"
else
    echo ">> AVISO: no encuentro .venv; instalaré en el Python actual."
fi

mkdir -p vendor
if [ ! -d vendor/facefusion/.git ]; then
    echo ">> Clonando FaceFusion en vendor/facefusion ..."
    git clone --depth 1 https://github.com/facefusion/facefusion vendor/facefusion
else
    echo ">> FaceFusion ya presente; actualizando ..."
    (cd vendor/facefusion && git pull --ff-only || true)
fi

echo ">> Instalando dependencias de FaceFusion en el entorno activo ..."
cd vendor/facefusion
if [ -f requirements.txt ]; then
    pip install -r requirements.txt || \
      echo "(Si falla, prueba el instalador propio de FaceFusion: 'python install.py --onnxruntime cuda')"
else
    echo "(No hay requirements.txt. Ejecuta el instalador de FaceFusion:"
    echo "   cd vendor/facefusion && python install.py --onnxruntime cuda )"
fi
cd "$ROOT"

echo ">> Restaurando dependencias críticas de Fuser (gradio 5, numpy<2) ..."
pip install -U "gradio>=5,<6" "numpy<2" >/dev/null 2>&1 || true

echo ">> Verificando que FaceFusion es importable desde Fuser ..."
python -c "from fuser.engines.facefusion_engine import is_available; print('FaceFusion importable:', is_available())"

echo ""
echo "============================================================"
echo " Listo. En la UI, elige el motor 'FaceFusion (Alta Calidad)'."
echo " Si la verificación dio False, revisa INSTALL.md (sección FaceFusion)."
echo "============================================================"
