#!/usr/bin/env bash
# Helper script: convert shell scripts to Unix line endings and create a WSL-native venv
# Usage: run this from WSL. It will:
#  - convert all .sh files in the repository to LF (in-place)
#  - create a Python virtual environment in your WSL home
#  - activate it and install requirements-gpu.txt with --no-cache-dir

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
echo "Repository root detected: $REPO_ROOT"

if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "Running inside WSL. Proceeding."
else
    echo "Warning: This script is intended for WSL / Linux. If you run on native Windows (PowerShell/cmd), stop and run the PowerShell instructions instead."
fi

echo "\n1) Converting .sh files to Unix (LF) line endings"
# Find .sh files and strip CR (\r) characters in-place
sh_files=$(find "$REPO_ROOT" -type f -name '*.sh')
if [ -z "$sh_files" ]; then
    echo "No .sh files found under $REPO_ROOT"
else
    for f in $sh_files; do
        echo "Converting: $f"
        # Make a backup in case something goes wrong
        cp -n "$f" "$f.bak" 2>/dev/null || true
        # Remove CR characters
        sed -i 's/\r$//' "$f"
        # Ensure executable bit for scripts in pr-2/topic-4
        if [[ "$f" == *"pr-2/topic-4/"* || "$f" == *"pr-2/topic-4"* ]]; then
            chmod +x "$f" || true
        fi
    done
    echo "Conversion complete. Backups with .bak suffix were created where they did not already exist."
fi

echo "\n2) Setting up a WSL-native Python virtual environment"
VENV_DIR="$HOME/lora-venv"
echo "Virtualenv path: $VENV_DIR"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

REQ="$REPO_ROOT/pr-2/topic-4/requirements-gpu.txt"
if [ ! -f "$REQ" ]; then
    echo "Could not find requirements file: $REQ"
    echo "Please run the pip install command manually and point to the correct path."
    exit 1
fi

echo "Installing packages from $REQ into venv (this may take a while)."
pip install --no-cache-dir -r "$REQ"

echo "\nSetup finished. To use the environment in this shell run:\n  source $VENV_DIR/bin/activate"
echo "If you need to inspect any .bak files, they were created alongside converted scripts."

echo "Done. If pip reported Errno 5 previously, installing into this venv (on WSL native FS) should avoid that problem."
