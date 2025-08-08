#!/bin/bash

# Shader Predictive Compiler Launcher Script
# This script handles launching the application with proper environment setup

APP_DIR="$(dirname "$(readlink -f "$0")")"
PYTHONPATH="$APP_DIR/src:$PYTHONPATH"

# Set up environment variables for optimal performance
export PYTHONPATH
export MESA_VK_DEVICE_SELECT="1002:163f"  # Steam Deck GPU
export RADV_PERFTEST="aco,nggc"           # AMD GPU optimizations

# Check if running from installed location
if [[ "$APP_DIR" == "/opt/shader-predict-compile" ]]; then
    # Production mode - installed via install.sh
    cd "$APP_DIR"
    exec python3 ui/main_window.py "$@"
else
    # Development mode - running from source directory
    cd "$APP_DIR"
    exec python3 ui/main_window.py "$@"
fi