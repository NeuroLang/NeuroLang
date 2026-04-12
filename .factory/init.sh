#!/bin/bash
set -e

cd /Users/dwasserm/sources/NeuroLang

# Install Python dependencies (idempotent)
if [ -f ".venv/bin/python" ]; then
    echo "Python venv exists, syncing dependencies..."
    uv sync --extra dev --extra server 2>/dev/null || true
else
    echo "Creating Python venv and installing dependencies..."
    uv venv
    uv sync --extra dev --extra server
fi

# Install frontend dependencies if the React app exists
SPARKLIS_DIR="neurolang/utils/server/neurolang-sparklis"
if [ -d "$SPARKLIS_DIR" ] && [ -f "$SPARKLIS_DIR/package.json" ]; then
    echo "Installing frontend dependencies..."
    cd "$SPARKLIS_DIR"
    npm install
    cd /Users/dwasserm/sources/NeuroLang
fi

echo "Init complete."
