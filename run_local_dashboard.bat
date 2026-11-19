@echo off
cd /d "C:\Users\Frank\quant pro wth pre market"
set "PYTHON_EXE=C:\Users\Frank\AppData\Local\Programs\Python\Python314\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
"%PYTHON_EXE%" -m streamlit run "dashboard.py" --server.headless true --server.port 8501 --server.fileWatcherType none --server.runOnSave false
pause
