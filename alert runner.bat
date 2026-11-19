@echo off
setlocal

cd /d "%~dp0"
title Quant Alert Runner - Telegram

set "PYTHON_EXE=C:\Users\Frank\AppData\Local\Programs\Python\Python314\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

echo Quant Alert Runner: Telegram alerts for US indices and Binance perpetuals every 5 minutes,
echo saves the Top 200 watch list to watchlist_history.csv every hour,
echo saves BTC/ETH gamma exposure (GEX) to gex_history.csv every hour,
echo saves SOL/XRP/AVAX/HYPE/TRX implied volatility to iv_snapshots.csv every hour,
echo records each session's Highest Flow Z Rise/Drop to session_flow_history.csv every 5 minutes,
echo and forward-tests Zone Battle + Flow signals on every timeframe to zone_flow_tracker_log.csv.
echo Close this window to stop it.
echo Log: %~dp0alert_runner.log
echo.

"%PYTHON_EXE%" "%~dp0alert_runner.py" %*

if errorlevel 1 (
  echo.
  echo Alert runner stopped with an error. See alert_runner.log
  pause
)

endlocal
