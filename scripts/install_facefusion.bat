@echo off
REM Instala el motor opcional FaceFusion (Windows). Wrapper de install_facefusion.py.
REM   scripts\install_facefusion.bat
setlocal
cd /d "%~dp0\.."
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
python scripts\install_facefusion.py
endlocal
