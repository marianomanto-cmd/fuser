@echo off
REM ============================================================================
REM  Fuser - instalacion del motor OPCIONAL FaceFusion (Windows)
REM
REM    scripts\install_facefusion.bat
REM
REM  - Clona FaceFusion en  vendor\facefusion  (Fuser lo auto-detecta ahi).
REM  - Instala sus dependencias en el MISMO entorno virtual (.venv) que Fuser.
REM  - Restaura los pines criticos de Fuser (gradio 5, numpy<2).
REM  - Verifica que 'import facefusion' funciona desde Fuser.
REM ============================================================================
setlocal
cd /d "%~dp0\.."

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    echo ^>^> Usando entorno .venv
) else (
    echo ^>^> AVISO: no encuentro .venv; instalare en el Python actual.
)

if not exist vendor mkdir vendor
if not exist vendor\facefusion\.git (
    echo ^>^> Clonando FaceFusion en vendor\facefusion ...
    git clone --depth 1 https://github.com/facefusion/facefusion vendor\facefusion
) else (
    echo ^>^> FaceFusion ya presente; actualizando ...
    pushd vendor\facefusion & git pull --ff-only & popd
)

echo ^>^> Instalando dependencias de FaceFusion en el entorno activo ...
pushd vendor\facefusion
if exist requirements.txt (
    pip install -r requirements.txt
) else (
    echo (No hay requirements.txt. Ejecuta: cd vendor\facefusion ^&^& python install.py --onnxruntime cuda^)
)
popd

echo ^>^> Restaurando dependencias criticas de Fuser (gradio 5, numpy^<2) ...
pip install -U "gradio>=5,<6" "numpy<2"

echo ^>^> Verificando que FaceFusion es importable desde Fuser ...
python -c "from fuser.engines.facefusion_engine import is_available; print('FaceFusion importable:', is_available())"

echo.
echo ============================================================
echo  Listo. En la UI, elige el motor 'FaceFusion (Alta Calidad)'.
echo  Si la verificacion dio False, revisa INSTALL.md (seccion FaceFusion).
echo ============================================================
endlocal
