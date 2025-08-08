#!/bin/bash
#
# Flatpak Launcher for ML Shader Prediction Compiler
# Handles Steam Deck specific environment setup within Flatpak sandbox
#

set -e

# Flatpak environment setup
export SHADER_PREDICT_FLATPAK=1
export SHADER_PREDICT_HOME="/app/lib/shader-predict-ml"
export SHADER_PREDICT_VERSION="3.0.0"
export PYTHONPATH="/app/lib/shader-predict-ml/src:${PYTHONPATH:-}"

# Steam Deck detection within Flatpak
detect_steam_deck() {
    local is_deck=false
    local deck_model="unknown"
    
    # Check DMI information (accessible through Flatpak)
    if [[ -f /sys/devices/virtual/dmi/id/product_name ]]; then
        local product_name=$(cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo "")
        case "$product_name" in
            *"Jupiter"*)
                is_deck=true
                deck_model="LCD"
                ;;
            *"Galileo"*)
                is_deck=true
                deck_model="OLED"
                ;;
        esac
    fi
    
    # Check for SteamOS
    if [[ -f /etc/os-release ]]; then
        source /etc/os-release
        if [[ "${ID:-}" == "steamos" ]]; then
            is_deck=true
        fi
    fi
    
    if [[ "$is_deck" == true ]]; then
        export STEAM_DECK=1
        export STEAM_DECK_MODEL="$deck_model"
        export STEAMDECK_MODE=1
        echo "Steam Deck $deck_model detected in Flatpak"
    else
        export STEAM_DECK=0
        export STEAM_DECK_MODEL="unknown"
        echo "Non-Steam Deck system detected"
    fi
}

# Thermal monitoring setup
setup_thermal_monitoring() {
    if [[ "$STEAM_DECK" == "1" ]]; then
        export SHADER_PREDICT_THERMAL_AWARE=1
        export SHADER_PREDICT_THERMAL_LIMIT=80
        
        # Check current thermal state
        if [[ -f /sys/class/thermal/thermal_zone0/temp ]]; then
            local temp=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo "0")
            temp=$((temp / 1000))
            
            if [[ $temp -gt 75 ]]; then
                export SHADER_PREDICT_THERMAL_THROTTLE=1
                echo "High temperature detected: ${temp}°C - enabling thermal protection"
            fi
        fi
    fi
}

# Battery monitoring setup
setup_battery_monitoring() {
    if [[ "$STEAM_DECK" == "1" ]]; then
        export SHADER_PREDICT_BATTERY_AWARE=1
        export SHADER_PREDICT_BATTERY_THRESHOLD=20
        
        # Check battery level
        if [[ -f /sys/class/power_supply/BAT1/capacity ]]; then
            local battery=$(cat /sys/class/power_supply/BAT1/capacity 2>/dev/null || echo "100")
            
            if [[ $battery -lt 20 ]]; then
                export SHADER_PREDICT_BATTERY_SAVE=1
                echo "Low battery detected: ${battery}% - enabling power saving"
            fi
        fi
    fi
}

# Gaming mode detection
detect_gaming_mode() {
    if pgrep -x "gamescope" >/dev/null 2>&1; then
        export SHADER_PREDICT_GAMING_MODE=1
        export SHADER_PREDICT_CPU_LIMIT=5
        export SHADER_PREDICT_MEMORY_LIMIT=200
        echo "Gaming mode detected - applying conservative resource limits"
    else
        export SHADER_PREDICT_GAMING_MODE=0
        export SHADER_PREDICT_CPU_LIMIT=10
        export SHADER_PREDICT_MEMORY_LIMIT=400
    fi
}

# Setup configuration directories
setup_directories() {
    # Flatpak maps these to ~/.var/app/com.shaderpredict.MLCompiler/
    local config_dir="$HOME/.config/shader-predict-ml"
    local cache_dir="$HOME/.cache/shader-predict-ml"  
    local data_dir="$HOME/.local/share/shader-predict-ml"
    
    mkdir -p "$config_dir"/{games,profiles,security,ml}
    mkdir -p "$cache_dir"/{shaders,ml_models,p2p,temp,thermal}
    mkdir -p "$data_dir"/{training,compiled,profiles,exports}
    
    export SHADER_PREDICT_CONFIG="$config_dir"
    export SHADER_PREDICT_CACHE="$cache_dir"
    export SHADER_PREDICT_DATA="$data_dir"
    
    # Create default configuration if it doesn't exist
    if [[ ! -f "$config_dir/config.json" ]]; then
        create_default_config "$config_dir/config.json"
    fi
}

# Create default configuration for Flatpak environment
create_default_config() {
    local config_file="$1"
    
    cat > "$config_file" << EOF
{
    "version": "3.0.0",
    "flatpak": {
        "enabled": true,
        "sandbox_mode": true,
        "steam_integration": true
    },
    "steam_deck": {
        "enabled": $STEAM_DECK,
        "model": "$STEAM_DECK_MODEL",
        "thermal_limit_celsius": 80,
        "battery_threshold_percent": 20,
        "gaming_mode_cpu_limit": 5,
        "desktop_mode_cpu_limit": 10,
        "memory_limit_mb": 300
    },
    "ml_prediction": {
        "enabled": true,
        "fallback_mode": true,
        "model_complexity": "light",
        "training_enabled": false
    },
    "p2p_network": {
        "enabled": false,
        "flatpak_limitations": "Network access limited in sandbox"
    },
    "performance": {
        "parallel_compilation": true,
        "max_worker_threads": 1,
        "thermal_throttling": true,
        "battery_aware": true,
        "sandboxed_execution": true
    },
    "security": {
        "flatpak_sandbox": true,
        "signature_verification": true,
        "privacy_protection": true,
        "restricted_filesystem_access": true
    },
    "logging": {
        "level": "INFO",
        "max_size_mb": 5,
        "max_files": 2,
        "flatpak_journal": true
    }
}
EOF
    
    echo "Created default Flatpak configuration at $config_file"
}

# Main launcher logic
main() {
    echo "ML Shader Prediction Compiler - Flatpak Launcher v3.0.0"
    echo "========================================================"
    
    # Environment detection and setup
    detect_steam_deck
    setup_thermal_monitoring
    setup_battery_monitoring  
    detect_gaming_mode
    setup_directories
    
    # Steam Deck specific optimizations
    if [[ "$STEAM_DECK" == "1" ]]; then
        export RADV_PERFTEST=aco
        export MESA_GLSL_CACHE_DISABLE=0
        export RADV_DEBUG=nocache
        export AMD_VULKAN_ICD=RADV
    fi
    
    # Change to application directory
    cd "$SHADER_PREDICT_HOME"
    
    # Handle different launch modes
    case "${1:-gui}" in
        --gui|gui)
            echo "Launching GUI mode..."
            exec python3 src/main.py --gui --flatpak --steam-deck
            ;;
        --service|service)
            echo "Launching service mode..."
            exec python3 src/main.py --service --flatpak --steam-deck
            ;;
        --status|status)
            echo "Checking status..."
            exec python3 src/main.py --status --flatpak
            ;;
        --test|test)
            echo "Running tests..."
            exec python3 src/main.py --test --flatpak --steam-deck
            ;;
        --steam-hook)
            echo "Steam integration hook activated"
            # Start background service for Steam integration
            nohup python3 src/main.py --service --flatpak --steam-mode --quiet >/dev/null 2>&1 &
            exit 0
            ;;
        --setup-steam)
            echo "Setting up Steam integration..."
            /app/share/shader-predict-ml/steam-integration/setup-steam-hooks.sh
            exit $?
            ;;
        --version|-v)
            echo "ML Shader Prediction Compiler v$SHADER_PREDICT_VERSION (Flatpak)"
            python3 src/main.py --version 2>/dev/null || echo "Core application not found"
            exit 0
            ;;
        --help|-h)
            cat << 'HELP'
ML Shader Prediction Compiler - Flatpak Edition

USAGE:
    flatpak run com.shaderpredict.MLCompiler [COMMAND]

COMMANDS:
    gui, --gui          Launch GUI interface (default)
    service, --service  Run as background service  
    status, --status    Show service status
    test, --test        Run system tests
    --setup-steam       Setup Steam integration
    --version, -v       Show version information
    --help, -h          Show this help

STEAM DECK FEATURES:
    - Automatic thermal throttling
    - Battery-aware operation
    - Gaming mode detection
    - Resource limit enforcement
    - Steam integration support

CONFIGURATION:
    Config: ~/.var/app/com.shaderpredict.MLCompiler/config/shader-predict-ml/
    Cache:  ~/.var/app/com.shaderpredict.MLCompiler/cache/shader-predict-ml/
    Data:   ~/.var/app/com.shaderpredict.MLCompiler/data/shader-predict-ml/

HELP
            exit 0
            ;;
        *)
            echo "Unknown command: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
}

# Error handling
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

# Run main function
main "$@"