#!/usr/bin/env bash
# ==============================================================================
# TikTok Wizard — macOS & Linux Automated Setup & Launcher
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "====================================================================="
echo "                    TikTok Wizard Launcher"
echo "====================================================================="
echo ""

# ------------------------------------------------------------------------------
# 1. Scan for Python
# ------------------------------------------------------------------------------
PY_CMD=""
if command -v python3 &> /dev/null; then
    PY_CMD="python3"
elif command -v python &> /dev/null; then
    PY_CMD="python"
fi

if [ -n "$PY_CMD" ]; then
    echo "[OK] Python is already installed ($($PY_CMD --version))."
else
    echo "[!] Python not detected. Attempting automatic installation..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            echo "[*] Installing Python 3 via Homebrew..."
            brew install python
            PY_CMD="python3"
        fi
    elif [ -f /etc/debian_version ]; then
        if command -v apt-get &> /dev/null; then
            echo "[*] Installing Python 3 via apt..."
            sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
            PY_CMD="python3"
        fi
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y python3
        PY_CMD="python3"
    elif command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm python python-pip
        PY_CMD="python3"
    fi
fi

if [ -z "$PY_CMD" ]; then
    echo ""
    echo "====================================================================="
    echo "[ERROR] Python 3.10+ is required to run TikTok Wizard."
    echo "Please install Python 3:"
    echo "  - macOS: brew install python  OR from https://www.python.org/"
    echo "  - Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    echo "  - Arch: sudo pacman -S python python-pip"
    echo "====================================================================="
    echo ""
    exit 1
fi

# ------------------------------------------------------------------------------
# 2. Scan for FFmpeg
# ------------------------------------------------------------------------------
if command -v ffmpeg &> /dev/null; then
    echo "[OK] FFmpeg is already installed."
else
    echo "[!] FFmpeg not detected. Attempting automatic installation..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            echo "[*] Installing ffmpeg via Homebrew..."
            brew install ffmpeg
        fi
    elif [ -f /etc/debian_version ]; then
        if command -v apt-get &> /dev/null; then
            echo "[*] Installing ffmpeg via apt..."
            sudo apt-get update && sudo apt-get install -y ffmpeg
        fi
    elif command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm ffmpeg
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y ffmpeg
    fi

    if command -v ffmpeg &> /dev/null; then
        echo "[OK] FFmpeg installed successfully."
    else
        echo "[!] Notice: ffmpeg was not detected. Video rendering requires ffmpeg."
    fi
fi

# ------------------------------------------------------------------------------
# 3. Setup Virtual Environment & Dependencies
# ------------------------------------------------------------------------------
if [ ! -d "venv" ]; then
    echo ""
    echo "[*] First time setup: Creating Python virtual environment..."
    $PY_CMD -m venv venv
    ./venv/bin/pip install --upgrade pip
    echo "[*] Installing required audio alignment libraries..."
    ./venv/bin/pip install -r requirements.txt
    echo "[*] Setup complete!"
    echo ""
fi

# ------------------------------------------------------------------------------
# 4. Run CLI
# ------------------------------------------------------------------------------
./venv/bin/python3 src/cli.py
