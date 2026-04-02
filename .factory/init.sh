#!/bin/bash
set -e

# Ensure uv is available
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed. Install from https://docs.astral.sh/uv/"
    exit 1
fi

echo "uv is available: $(uv --version)"
echo "Python versions available:"
uv python list 2>/dev/null | head -10 || echo "  python3.12: $(which python3.12 2>/dev/null || echo 'not found')"

# Install the project in dev mode if pyproject.toml exists with project metadata
if grep -q '^\[project\]' pyproject.toml 2>/dev/null; then
    echo "Installing project with dev extras via uv..."
    uv pip install -e ".[dev]" 2>/dev/null || echo "Install skipped (may need migration first)"
else
    echo "pyproject.toml not yet migrated, skipping install"
fi
