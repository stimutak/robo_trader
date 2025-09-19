#!/usr/bin/env bash
set -euo pipefail

echo "[reboot] Killing existing processes..."
pkill -9 -f "runner_async" || true
pkill -9 -f "app.py" || true
pkill -9 -f "websocket_server" || true

echo "[reboot] Ensuring local repo code is used (PYTHONPATH=.)"
export PYTHONPATH="${PYTHONPATH:-.}:."

echo "[reboot] Starting WebSocket server..."
nohup python3 -m robo_trader.websocket_server > websocket.log 2>&1 &
WS_PID=$!

echo "[reboot] Starting trader..."
export LOG_FILE="$PWD/robo_trader.log"
nohup python3 -m robo_trader.runner_async \
  --symbols AAPL,NVDA,TSLA,IXHL,NUAI,BZAI,ELTP,OPEN,CEG,TRV,PLTR,UPST,TEM,HTFL,SDGR,APLD,SOFI,CORZ,WULF,QQQ,QLD,BBIO,IMRX,CRGY \
  > trading_runner.log 2>&1 &
RUN_PID=$!

echo "[reboot] Starting dashboard on port 5555..."
export DASH_PORT=5555
nohup python3 app.py > dashboard.log 2>&1 &
DASH_PID=$!

echo "[reboot] Started PIDs: WS=${WS_PID} RUNNER=${RUN_PID} DASH=${DASH_PID}"
echo "[reboot] Tailing trader log (Ctrl+C to stop tail)"
sleep 1
tail -n +1 -f trading_runner.log | sed -u 's/^/[trader] /'

