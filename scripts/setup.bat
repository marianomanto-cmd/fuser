@echo off
REM ============================================================================
REM  Fuser - instalacion local automatica (Windows)
REM
REM    scripts\setup.bat                  (GPU/CUDA + FaceFusion)
REM    scripts\setup.bat --cpu            (version CPU; sin FaceFusion)
REM    scripts\setup.bat --no-facefusion  (GPU pero sin FaceFusion)
REM
REM  Deja TODO listo: solo queda `python app.py`.
REM ============================================================================
setlocal
cd /d "%~dp0\.."

set REQ=requirements.txt
set WITH_FF=1
set RUN_DEMO=0
if "%1"=="--cpu" ( set REQ=requirements-cpu.txt & set WITH_FF=0 & echo ^>^> Modo CPU )
if "%1"=="--no-facefusion" ( set WITH_FF=0 )
if "%1"=="--demo" ( set RUN_DEMO=1 )
if "%2"=="--demo" ( set RUN_DEMO=1 )

echo ^>^> Creando entorno virtual en .venv ...
python -m venv .venv
call .venv\Scripts\activate.bat

echo ^>^> Actualizando pip ...
python -m pip install --upgrade pip

echo ^>^> Instalando dependencias (%REQ%) ...
pip install -r %REQ%

echo ^>^> Descargando modelos recomendados ...
python scripts\download_models.py

if "%WITH_FF%"=="1" (
    echo ^>^> Instalando el motor FaceFusion (alta calidad) ...
    python scripts\install_facefusion.py
)

echo ^>^> Diagnostico de entorno:
python scripts\check_env.py

if "%RUN_DEMO%"=="1" (
    echo ^>^> Ejecutando la prueba automática ...
    python scripts\run_demo.py
)

echo.
echo ============================================================
echo  Listo. Para usar la app:
echo    .venv\Scripts\activate
echo    python app.py        ^>  http://127.0.0.1:7860
echo.
echo  PRIMERA PRUEBA recomendada (descarga stock y prueba features):
echo    python scripts\run_demo.py        ( resultados en la carpeta prueba\ )
echo ============================================================
endlocal
