@echo off
setlocal

cd /d "%~dp0"
title Quant Dashboard - Streamlit

set "PYTHON_EXE=C:\Users\Frank\AppData\Local\Programs\Python\Python314\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

echo Starting Quant Dashboard at http://localhost:8501
echo Close this window to stop the server.
echo.

"%PYTHON_EXE%" -m streamlit run "%~dp0dashboard.py" ^
  --server.headless true ^
  --server.port 8501 ^
  --server.fileWatcherType none ^
  --server.runOnSave false

if errorlevel 1 (
  echo.
  echo Dashboard stopped with an error.
  pause
)

endlocal
