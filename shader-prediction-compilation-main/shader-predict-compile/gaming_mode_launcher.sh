#!/bin/bash

# Gaming Mode UI Launcher for Shader Predictive Compiler
# This provides a controller-friendly interface when launched from Gaming Mode

cd "/home/deck/shader-predict-compile"

# Set environment for Gaming Mode
export GAMING_MODE=1
export DISPLAY=:0
export QT_SCALE_FACTOR=1.5
export GDK_SCALE=1.5

# Check if we're in Gaming Mode (gamescope running)
if pgrep -x gamescope > /dev/null; then
    echo "Launching in Gaming Mode..."
    
    # Launch with Gaming Mode optimized settings
    python3 src/gaming_mode_ui.py "$@"
else
    echo "Launching in Desktop Mode..."
    
    # Launch normal GUI
    python3 ui/main_window.py "$@"
fi
