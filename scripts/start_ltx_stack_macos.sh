#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
CUSTOM_NODE="$ROOT_DIR/custom_nodes/ComfyUI-LTXVideo-mlx"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing Python environment: $PYTHON" >&2
    exit 1
fi

if [[ ! -f "$CUSTOM_NODE/__init__.py" ]]; then
    echo "Missing custom node: $CUSTOM_NODE" >&2
    exit 1
fi

export HF_HUB_OFFLINE=1
cd "$ROOT_DIR"
exec "$PYTHON" main.py --listen 127.0.0.1 --port 8188 --preview-method auto
