#!/bin/bash

# Auto-detect launcher for Shader Predictive Compiler
# Automatically detects Steam game libraries and handles gaming mode

cd "$(dirname "$0")"

# Set up environment
export SHADER_PREDICT_HOME="$(pwd)"
export PYTHONPATH="$SHADER_PREDICT_HOME/src:$PYTHONPATH"

# Logging
LOG_FILE="$HOME/.cache/shader-predict-compile/launcher.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check dependencies
check_dependencies() {
    local missing=()
    
    # Check Python
    if ! command -v python3 &>/dev/null; then
        missing+=("python3")
    fi
    
    # Check GTK
    if ! python3 -c "import gi; gi.require_version('Gtk', '3.0')" &>/dev/null 2>&1; then
        missing+=("python3-gi (GTK bindings)")
    fi
    
    # Check optional dependencies
    if ! python3 -c "import psutil" &>/dev/null 2>&1; then
        log "Warning: psutil not installed - system monitoring will be limited"
    fi
    
    if ! python3 -c "import numpy" &>/dev/null 2>&1; then
        log "Warning: numpy not installed - advanced analysis features will be limited"
    fi
    
    if [ ${#missing[@]} -ne 0 ]; then
        log "Error: Missing required dependencies: ${missing[*]}"
        echo "Please run the installer first: ./install"
        exit 1
    fi
}

log "Starting Shader Predictive Compiler Auto-Launcher"
check_dependencies

# Check if we're in Gaming Mode (gamescope running)
if pgrep -x gamescope > /dev/null; then
    log "Gaming Mode detected - launching Gaming Mode UI"
    export GAMING_MODE=1
    export DISPLAY=:0
    export QT_SCALE_FACTOR=1.5
    export GDK_SCALE=1.5
    
    # Gaming Mode specific optimizations
    export SDL_VIDEODRIVER=x11
    export XDG_SESSION_TYPE=x11
    
    # Launch Gaming Mode UI
    if python3 src/gaming_mode_ui.py "$@"; then
        log "Gaming Mode UI launched successfully"
    else
        log "Gaming Mode UI launch failed, falling back to desktop mode"
        python3 ui/main_window.py "$@"
    fi
else
    log "Desktop Mode detected - launching Desktop UI"
    # Check available UIs and launch the best one
    if [ -f "ui/main_window.py" ]; then
        python3 ui/main_window.py "$@"
    elif [ -f "src/gaming_mode_ui.py" ]; then
        log "Main UI not found, using Gaming Mode UI in desktop mode"
        python3 src/gaming_mode_ui.py "$@"
    else
        log "No UI found, running service status check"
        python3 src/background_service.py --status
    fi
fi

log "Launcher execution completed"