@echo off
REM ============================================================================
REM  Fuser - desinstalacion completa (Windows)
REM
REM    scripts\uninstall.bat                 desinstala y CONSERVA tus datos
REM    scripts\uninstall.bat --dry-run       solo muestra que borraria
REM    scripts\uninstall.bat --purge-data    borra tambien caras/salidas/entrenamientos
REM    scripts\uninstall.bat --remove-repo   borra tambien esta carpeta
REM
REM  Se puede ejecutar con doble clic. Usa el Python del sistema a proposito:
REM  el .venv es una de las cosas que se borran.
REM ============================================================================
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "PROJ=%~dp0.."
REM Situarse FUERA del proyecto: no se puede borrar la carpeta actual.
cd /d "%PROJ%\.."

set "PY=python"
where python >nul 2>&1 || set "PY=py"
where %PY% >nul 2>&1
if errorlevel 1 (
  echo.
  echo [ERROR] No encuentro Python en el PATH.
  echo Instalalo desde https://www.python.org/downloads/ o borra a mano la carpeta:
  echo     %PROJ%
  echo.
  pause
  exit /b 1
)

"%PY%" "%~dp0uninstall.py" %*
set "RC=%ERRORLEVEL%"

echo.
pause
exit /b %RC%
