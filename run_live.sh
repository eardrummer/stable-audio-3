#!/bin/bash

# Launches the Stable Audio 3 LIVE loop app (input -> continuous audio-to-audio).
# Shares the backend and frontend project with run_studio.sh; the only
# difference is which page you land on (live.html instead of index.html).

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WEBAPP_DIR="$SCRIPT_DIR/optimized/mlx/webapp"
BACKEND_DIR="$WEBAPP_DIR/backend"
FRONTEND_DIR="$WEBAPP_DIR/frontend"
VENV_DIR="$SCRIPT_DIR/optimized/mlx/.venv"

echo "=================================================="
echo "      STABLE AUDIO 3 — LIVE LOOP WORKSTATION      "
echo "=================================================="

cleanup() {
    echo ""
    echo "[Launcher] Stopping live servers..."
    if [ ! -z "$BACKEND_PID" ]; then
        echo "[Launcher] Terminating Backend (PID $BACKEND_PID)..."
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        echo "[Launcher] Terminating Frontend (PID $FRONTEND_PID)..."
        kill $FRONTEND_PID 2>/dev/null
    fi
    echo "[Launcher] Live servers stopped cleanly. Goodbye!"
    exit 0
}

trap cleanup SIGINT SIGTERM

check_port() {
    local port=$1
    local name=$2
    local pid=$(lsof -t -i:$port)
    if [ ! -z "$pid" ]; then
        echo -e "\033[33m[Warning] Port $port ($name) is already in use by process $pid.\033[0m"
        read -p "Would you like to terminate this process? (y/n): " confirm
        if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
            echo "Killing process $pid..."
            kill -9 $pid
            sleep 1
        else
            echo "Aborting. Please free port $port and try again."
            exit 1
        fi
    fi
}

check_port 8000 "FastAPI Backend"
check_port 5173 "Vite Frontend"

# Resolve Python Executable Dynamically
PYTHON_CMD=""
if [ -f "$VENV_DIR/bin/python" ] && "$VENV_DIR/bin/python" -c "import sys" 2>/dev/null; then
    PYTHON_CMD="$VENV_DIR/bin/python"
elif command -v uv &> /dev/null; then
    PYTHON_CMD="uv run --directory $SCRIPT_DIR/optimized/mlx python"
elif command -v python3 &> /dev/null && python3 -c "import mlx.core" 2>/dev/null; then
    PYTHON_CMD="python3"
else
    echo -e "\033[31m[Error] Python environment not found or broken at $VENV_DIR.\033[0m"
    echo -e "To set up this computer for the first time, run:"
    echo -e "  \033[36mcd optimized/mlx && ./install.sh\033[0m"
    echo -e "  \033[36mcd optimized/mlx/webapp/frontend && npm install\033[0m"
    exit 1
fi

# The live app needs scipy (48k -> 44.1k resampling) and mido/python-rtmidi
# (MIDI knob control) on top of the studio's dependencies.
if ! $PYTHON_CMD -c "import uvicorn, fastapi, scipy, mido, rtmidi" 2>/dev/null; then
    echo -e "\033[33m[Launcher] Installing missing backend dependencies...\033[0m"
    if [ -f "$VENV_DIR/bin/python" ]; then
        if command -v uv &> /dev/null; then
            VIRTUAL_ENV="$VENV_DIR" uv pip install -r "$SCRIPT_DIR/optimized/mlx/requirements.txt"
        else
            "$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/optimized/mlx/requirements.txt"
        fi
    fi
fi

# 1. Start Backend
echo "[Launcher] Starting FastAPI Backend using ($PYTHON_CMD)..."
cd "$BACKEND_DIR" || exit 1

$PYTHON_CMD -m uvicorn main:app --host 127.0.0.1 --port 8000 > "$BACKEND_DIR/backend.log" 2>&1 &
BACKEND_PID=$!

sleep 2.5
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "\033[31m[Error] Backend failed to start.\033[0m"
    echo "Check logs at: $BACKEND_DIR/backend.log"
    echo "--------------------------------------------------"
    tail -n 20 "$BACKEND_DIR/backend.log"
    echo "--------------------------------------------------"
    exit 1
fi
echo -e "\033[32m[Launcher] Backend started successfully! (PID $BACKEND_PID, logging to backend.log)\033[0m"

# Report detected MIDI inputs so you know the controller was seen.
MIDI_PORTS=$(curl -s http://127.0.0.1:8000/api/live/midi_ports 2>/dev/null)
if [ ! -z "$MIDI_PORTS" ]; then
    echo -e "[Launcher] MIDI inputs: \033[36m$MIDI_PORTS\033[0m"
fi

# 2. Start Frontend
echo "[Launcher] Starting React Frontend..."
cd "$FRONTEND_DIR" || exit 1

# Live Code mode plays Strudel patterns as the input source. Its sample banks
# are vendored locally so a performance never depends on the network; synth
# sounds work without them, but drums and piano do not.
if [ ! -d "$FRONTEND_DIR/public/strudel-samples" ]; then
    echo -e "\033[33m[Launcher] Strudel sample banks not vendored yet.\033[0m"
    echo -e "  Run \033[36mnpm run fetch-samples\033[0m in $FRONTEND_DIR for offline drums/piano (~1.2 MB)."
fi

npm run dev &
FRONTEND_PID=$!

echo "=================================================="
echo -e "\033[32m🎛  STABLE AUDIO 3 LIVE LOOP IS ONLINE!\033[0m"
echo -e "👉 Live App:        \033[36mhttp://localhost:5173/live.html\033[0m"
echo -e "👉 Studio (same backend): \033[36mhttp://localhost:5173/\033[0m"
echo -e "👉 Press \033[31mCtrl + C\033[0m in this terminal to stop both servers safely"
echo "=================================================="

wait $FRONTEND_PID
