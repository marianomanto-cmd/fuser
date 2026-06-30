@echo off
setlocal
title Fuser - servidor (cierra esta ventana para detener la app)
cd /d "%~dp0"

rem Forzar UTF-8 para que la app no se caiga al imprimir caracteres Unicode en consola.
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo.
  echo [ERROR] No se encontro el entorno virtual de Python:
  echo     %PY%
  echo.
  echo La instalacion aun no termino o fallo. Completa la instalacion
  echo de Fuser antes de usar este acceso directo.
  echo.
  pause
  exit /b 1
)

echo ============================================================
echo   FUSER - Face Swap de Video
echo ------------------------------------------------------------
echo   Iniciando el servidor local...
echo   La aplicacion se abrira sola en una ventana aparte.
echo.
echo   * NO cierres esta ventana mientras uses la app.
echo   * Para DETENER la app, cierra esta ventana.
echo ============================================================
echo.

rem Abre la UI en su propia ventana cuando el servidor este listo (en segundo plano).
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0open_fuser_window.ps1"

rem Ejecuta la app capturando TODO (stdout+stderr nativo, sin buffer) a un log,
rem para poder diagnosticar crashes de DirectML que no dejan rastro de otro modo.
"%PY%" -u app.py > "%~dp0fuser_console.log" 2>&1

echo.
echo ------------------------------------------------------------
echo La app se ha detenido. Pulsa una tecla para cerrar la ventana.
pause >nul
