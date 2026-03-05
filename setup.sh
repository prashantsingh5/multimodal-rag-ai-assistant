#!/usr/bin/env bash
# setup.sh — Linux/macOS bootstrap for Multimodal RAG AI Assistant
# Run once: bash setup.sh
# After that, always activate with: source venv/bin/activate

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "=== Multimodal RAG AI Assistant — Environment Setup ==="

# --- 1. Create venv ---
if [ ! -d "$ROOT/venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv "$ROOT/venv"
    echo "      venv created."
else
    echo "[1/4] Virtual environment already exists — skipping."
fi

# --- 2. Activate ---
echo "[2/4] Activating virtual environment..."
source "$ROOT/venv/bin/activate"

# --- 3. Install deps ---
echo "[3/4] Installing dependencies..."
pip install --upgrade pip
pip install -r "$ROOT/requirements.txt"

# --- 4. Runtime dirs & .env ---
echo "[4/4] Creating runtime directories and .env..."
mkdir -p "$ROOT/assets/vectorstore"

if [ ! -f "$ROOT/.env" ]; then
    cp "$ROOT/.env.example" "$ROOT/.env"
    echo "      .env created from .env.example — please fill in your API keys."
fi

echo ""
echo "=== Setup complete! ==="
echo "API server:   uvicorn api.server:app --reload"
echo "Frontend:     python frontend/app.py"
echo "Activate:     source venv/bin/activate"
echo ""
