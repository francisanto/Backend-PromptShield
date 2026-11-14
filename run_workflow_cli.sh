#!/usr/bin/env bash
# PromptShield AI — Complete Application Workflow CLI Launcher
# ============================================================

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

# Run the dedicated unified workflow terminal CLI
exec "$PYTHON_EXEC" workflow_cli.py "$@"

