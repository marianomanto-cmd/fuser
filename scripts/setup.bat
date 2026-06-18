@echo off
REM ============================================================================
REM  Fuser - instalacion local automatica (Windows)
REM
REM    scripts\setup.bat          (instala dependencias GPU / CUDA)
REM    scripts\setup.bat --cpu    (instala version CPU, para probar la UI)
REM
REM  Crea un entorno virtual en .venv, instala dependencias, descarga los
REM  modelos recomendados y ejecuta el diagnostico de entorno.
REM ============================================================================
setlocal
cd /d "%~dp0\.."

set REQ=requirements.txt
if "%1"=="--cpu" (
    set REQ=requirements-cpu.txt
    echo ^>^> Modo CPU: usando %REQ%
)

echo ^>^> Creando entorno virtual en .venv ...
python -m venv .venv
call .venv\Scripts\activate.bat

echo ^>^> Actualizando pip ...
python -m pip install --upgrade pip

echo ^>^> Instalando dependencias (%REQ%) ...
pip install -r %REQ%

echo ^>^> Descargando modelos recomendados (inswapper_128 + gfpgan_1.4) ...
python scripts\download_models.py

echo ^>^> Diagnostico de entorno:
python scripts\check_env.py

echo.
echo ============================================================
echo  Listo. Para usar la app:
echo    .venv\Scripts\activate
echo    python app.py
echo  Abre http://127.0.0.1:7860
echo ============================================================
endlocal
