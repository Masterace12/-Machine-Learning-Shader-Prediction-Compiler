#!/bin/bash
#
# Service Wrapper for Shader Prediction Compiler
# Manages the main Python application with proper environment and monitoring
#

set -euo pipefail

# Environment setup
export SHADER_PREDICT_HOME="${SHADER_PREDICT_HOME:-/home/deck/.local/share/shader-predict-compile}"
export SHADER_PREDICT_CONFIG="${SHADER_PREDICT_CONFIG:-/home/deck/.config/shader-predict-compile}"
export SHADER_PREDICT_CACHE="${SHADER_PREDICT_CACHE:-/home/deck/.cache/shader-predict-compile}"
export PYTHONPATH="${SHADER_PREDICT_HOME}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Steam Deck specific
export STEAMDECK_MODE="${STEAMDECK_MODE:-1}"
export THERMAL_AWARE="${THERMAL_AWARE:-1}"
export BATTERY_AWARE="${BATTERY_AWARE:-1}"

# Logging configuration
readonly LOG_DIR="${SHADER_PREDICT_HOME}/logs"
readonly LOG_FILE="${LOG_DIR}/service-$(date +%Y%m%d).log"
readonly PID_FILE="/tmp/shader-predict.pid"

# Create directories if needed
mkdir -p "$LOG_DIR"
mkdir -p "$SHADER_PREDICT_CONFIG"
mkdir -p "$SHADER_PREDICT_CACHE"

# Logging function
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handler
handle_error() {
    local line_no=$1
    local exit_code=$2
    log_message "ERROR: Script failed at line $line_no with exit code $exit_code"
    cleanup
    exit "$exit_code"
}

# Cleanup function
cleanup() {
    log_message "Cleaning up..."
    
    # Remove PID file
    rm -f "$PID_FILE"
    
    # Stop any child processes
    if [[ -n "${MAIN_PID:-}" ]]; then
        kill -TERM "$MAIN_PID" 2>/dev/null || true
    fi
    
    # Flush logs
    sync
}

# Signal handlers
trap cleanup EXIT
trap 'handle_error ${LINENO} $?' ERR
trap 'log_message "Received SIGTERM"; cleanup; exit 0' TERM
trap 'log_message "Received SIGINT"; cleanup; exit 0' INT

# Check dependencies
check_dependencies() {
    log_message "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &>/dev/null; then
        log_message "ERROR: Python 3 not found"
        exit 1
    fi
    
    # Check main application
    if [[ ! -f "${SHADER_PREDICT_HOME}/src/main.py" ]]; then
        log_message "ERROR: Main application not found at ${SHADER_PREDICT_HOME}/src/main.py"
        exit 1
    fi
    
    # Check configuration
    if [[ ! -f "${SHADER_PREDICT_CONFIG}/settings.json" ]]; then
        log_message "WARNING: Configuration not found, using defaults"
        # Create default configuration
        cat > "${SHADER_PREDICT_CONFIG}/settings.json" << 'EOF'
{
    "mode": "auto",
    "ml_enabled": true,
    "p2p_enabled": true,
    "thermal_monitoring": true,
    "battery_aware": true,
    "max_cpu_percent": 10,
    "max_memory_mb": 500,
    "cache_size_gb": 2,
    "gamemode_throttle": true,
    "auto_start": false,
    "log_level": "info"
}
EOF
    fi
    
    log_message "Dependencies check passed"
}

# Initialize environment
initialize_environment() {
    log_message "Initializing environment..."
    
    # Set process priority
    renice 15 $$ >/dev/null 2>&1 || true
    
    # Set I/O priority
    ionice -c 3 -p $$ >/dev/null 2>&1 || true
    
    # Create cache directories
    mkdir -p "${SHADER_PREDICT_CACHE}/ml_models"
    mkdir -p "${SHADER_PREDICT_CACHE}/shader_cache"
    mkdir -p "${SHADER_PREDICT_CACHE}/p2p_cache"
    
    # Check available memory
    local available_mem=$(free -m | awk '/^Mem:/{print $7}')
    if [[ $available_mem -lt 500 ]]; then
        log_message "WARNING: Low memory available (${available_mem}MB)"
        export SHADER_PREDICT_LOW_MEMORY=1
    fi
    
    # Check thermal state
    if [[ -f /sys/class/thermal/thermal_zone0/temp ]]; then
        local cpu_temp=$(cat /sys/class/thermal/thermal_zone0/temp)
        cpu_temp=$((cpu_temp / 1000))
        log_message "Current CPU temperature: ${cpu_temp}°C"
        
        if [[ $cpu_temp -gt 80 ]]; then
            log_message "WARNING: High CPU temperature, enabling thermal throttling"
            export SHADER_PREDICT_THERMAL_THROTTLE=1
        fi
    fi
    
    log_message "Environment initialized"
}

# Health check function
health_check() {
    # Check if main process is running
    if [[ -n "${MAIN_PID:-}" ]] && kill -0 "$MAIN_PID" 2>/dev/null; then
        # Send health check signal
        kill -USR1 "$MAIN_PID" 2>/dev/null || true
        return 0
    else
        return 1
    fi
}

# Main service loop
run_service() {
    log_message "Starting Shader Prediction Compiler service..."
    
    # Write PID file
    echo $$ > "$PID_FILE"
    
    # Start main Python application
    cd "$SHADER_PREDICT_HOME"
    
    # Launch with proper resource constraints
    exec -a shader-predict \
        python3 -u src/main.py \
        --config "${SHADER_PREDICT_CONFIG}/settings.json" \
        --cache-dir "$SHADER_PREDICT_CACHE" \
        --log-file "$LOG_FILE" \
        --daemon \
        2>&1 | tee -a "$LOG_FILE" &
    
    MAIN_PID=$!
    log_message "Main process started with PID $MAIN_PID"
    
    # Notify systemd that we're ready
    systemd-notify --ready 2>/dev/null || true
    
    # Monitor loop
    local health_check_interval=30
    local health_check_counter=0
    
    while true; do
        # Wait for process or signals
        if ! wait -n "$MAIN_PID"; then
            log_message "Main process exited"
            break
        fi
        
        # Periodic health check
        if [[ $((health_check_counter % health_check_interval)) -eq 0 ]]; then
            if health_check; then
                systemd-notify --status="Healthy" WATCHDOG=1 2>/dev/null || true
            else
                log_message "Health check failed"
                systemd-notify --status="Unhealthy" 2>/dev/null || true
            fi
        fi
        
        health_check_counter=$((health_check_counter + 1))
        sleep 1
    done
}

# Main execution
main() {
    log_message "=== Service wrapper starting ==="
    log_message "Version: 2.0.0"
    log_message "User: $(whoami)"
    log_message "Home: $SHADER_PREDICT_HOME"
    
    # Run initialization
    check_dependencies
    initialize_environment
    
    # Start service
    run_service
    
    log_message "=== Service wrapper exiting ==="
}

# Run main function
main "$@"