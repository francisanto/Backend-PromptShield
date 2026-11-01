#!/usr/bin/env bash
# PromptShield AI — Interactive Workspace Launcher
# =================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Check for virtual environment
if [ -d ".venv" ]; then
    PYTHON_EXEC=".venv/bin/python"
elif [ -d "venv" ]; then
    PYTHON_EXEC="venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

# Ensure dependencies are available
if ! "$PYTHON_EXEC" -c "import fastapi, joblib, sklearn" 2>/dev/null; then
    echo "Warning: Some dependencies might be missing in $PYTHON_EXEC."
    echo "Installing requirements..."
    "$PYTHON_EXEC" -m pip install -r requirements.txt
fi

# Run the unified interactive terminal workspace
exec "$PYTHON_EXEC" cli.py "$@"
