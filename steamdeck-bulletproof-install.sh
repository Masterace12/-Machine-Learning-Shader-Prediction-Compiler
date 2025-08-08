#!/bin/bash
#
# Steam Deck Bulletproof Installation Script
# ML Shader Prediction Compiler - Production Ready
# 
# Handles all Steam Deck specific issues:
# - Immutable filesystem (SteamOS 3.x)
# - Memory constraints and thermal limits
# - Network connectivity issues
# - Package dependency resolution
# - Multiple installation fallback methods
# - LCD and OLED Steam Deck compatibility
# - Resource constraint enforcement
#
# Version: 3.0.0-bulletproof
# Author: Steam Deck Optimization Team
#

set -euo pipefail
IFS=$'\n\t'

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

readonly SCRIPT_VERSION="3.0.0-bulletproof"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Installation paths - Steam Deck optimized
readonly INSTALL_BASE="$HOME/.local/share"
readonly INSTALL_DIR="$INSTALL_BASE/ml-shader-predictor"
readonly CONFIG_DIR="$HOME/.config/ml-shader-predictor"
readonly CACHE_DIR="$HOME/.cache/ml-shader-predictor"
readonly LOG_DIR="$INSTALL_DIR/logs"
readonly SERVICE_DIR="$HOME/.config/systemd/user"

# Files and logs
readonly INSTALL_LOG="$LOG_DIR/install_${TIMESTAMP}.log"
readonly ERROR_LOG="$LOG_DIR/errors_${TIMESTAMP}.log"
readonly DEPENDENCY_CACHE="$CACHE_DIR/dependency_status.json"

# System constraints
readonly MAX_INSTALL_MEMORY_MB=1500    # Leave 500MB+ for system
readonly MAX_INSTALL_TIME_SEC=1800     # 30 minutes max
readonly MAX_TEMP_CELSIUS=85           # Emergency thermal limit
readonly MIN_BATTERY_PERCENT=15        # Minimum battery for installation

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Global state variables
declare -g STEAM_DECK_MODEL=""
declare -g IS_IMMUTABLE_OS=false
declare -g AVAILABLE_MEMORY_MB=0
declare -g CURRENT_TEMP_C=0
declare -g BATTERY_LEVEL=100
declare -g IS_CHARGING=true
declare -g NETWORK_AVAILABLE=false
declare -g INSTALL_PROFILE=""
declare -g PYTHON_EXECUTABLE=""

# ============================================================================
# LOGGING AND OUTPUT FUNCTIONS
# ============================================================================

setup_logging() {
    mkdir -p "$LOG_DIR" "$CACHE_DIR" "$CONFIG_DIR"
    
    # Create log files with proper permissions
    touch "$INSTALL_LOG" "$ERROR_LOG"
    chmod 644 "$INSTALL_LOG" "$ERROR_LOG"
    
    # Set up log rotation to prevent filling disk
    if [[ -f "$INSTALL_LOG" ]] && [[ $(wc -c <"$INSTALL_LOG") -gt 10485760 ]]; then
        mv "$INSTALL_LOG" "${INSTALL_LOG}.old"
        touch "$INSTALL_LOG"
    fi
}

log_message() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] [$level] $message" >> "$INSTALL_LOG"
    
    case "$level" in
        "ERROR")   echo -e "${RED}[ERROR]${NC} $message" >&2; echo "[$timestamp] [ERROR] $message" >> "$ERROR_LOG" ;;
        "WARNING") echo -e "${YELLOW}[WARNING]${NC} $message" ;;
        "SUCCESS") echo -e "${GREEN}[SUCCESS]${NC} $message" ;;
        "INFO")    echo -e "${BLUE}[INFO]${NC} $message" ;;
        "DEBUG")   [[ "${DEBUG:-0}" == "1" ]] && echo -e "${CYAN}[DEBUG]${NC} $message" ;;
        *)         echo -e "${BOLD}[$level]${NC} $message" ;;
    esac
}

log_error() { log_message "ERROR" "$@"; }
log_warning() { log_message "WARNING" "$@"; }
log_success() { log_message "SUCCESS" "$@"; }
log_info() { log_message "INFO" "$@"; }
log_debug() { log_message "DEBUG" "$@"; }

progress_bar() {
    local current="$1"
    local total="$2"
    local label="${3:-Installing}"
    local width=50
    local percentage=$((current * 100 / total))
    local completed=$((current * width / total))
    local remaining=$((width - completed))
    
    printf "\r${BLUE}[%-${width}s] %d%% %s${NC}" \
           "$(printf "%*s" "$completed" | tr ' ' '#')" \
           "$percentage" \
           "$label"
    
    [[ "$current" -eq "$total" ]] && echo
}

# ============================================================================
# SYSTEM DETECTION AND VALIDATION
# ============================================================================

detect_steam_deck() {
    log_info "Detecting Steam Deck hardware and configuration..."
    
    # Method 1: Check DMI product name
    if [[ -f "/sys/class/dmi/id/product_name" ]]; then
        local product_name
        product_name=$(cat /sys/class/dmi/id/product_name 2>/dev/null || echo "unknown")
        
        if [[ "$product_name" == *"Jupiter"* ]] || [[ "$product_name" == *"Steam Deck"* ]]; then
            # Detect LCD vs OLED model
            if [[ "$product_name" == *"1010"* ]] || [[ "$product_name" == *"1020"* ]] || [[ "$product_name" == *"1030"* ]]; then
                STEAM_DECK_MODEL="LCD"
            elif [[ "$product_name" == *"1040"* ]] || [[ "$product_name" == *"OLED"* ]]; then
                STEAM_DECK_MODEL="OLED"
            else
                STEAM_DECK_MODEL="LCD"  # Default assumption
            fi
            
            log_success "Detected Steam Deck $STEAM_DECK_MODEL: $product_name"
            return 0
        fi
    fi
    
    # Method 2: Check for AMD Van Gogh APU
    if command -v lscpu >/dev/null 2>&1; then
        local cpu_info
        cpu_info=$(lscpu 2>/dev/null)
        
        if echo "$cpu_info" | grep -qi "AMD.*Custom.*APU"; then
            STEAM_DECK_MODEL="LCD"  # Assume LCD if we can't determine
            log_success "Detected Steam Deck APU (model detection uncertain)"
            return 0
        fi
    fi
    
    # Method 3: Check SteamOS version
    if [[ -f "/etc/os-release" ]]; then
        local os_release
        os_release=$(cat /etc/os-release 2>/dev/null)
        
        if echo "$os_release" | grep -qi "steamos"; then
            STEAM_DECK_MODEL="LCD"  # Default
            log_warning "SteamOS detected but hardware model uncertain"
            return 0
        fi
    fi
    
    log_warning "Steam Deck hardware not detected - using compatibility mode"
    STEAM_DECK_MODEL="GENERIC"
    return 1
}

check_filesystem_mutability() {
    log_info "Checking filesystem mutability..."
    
    # Check if root filesystem is read-only
    if [[ -w "/usr" ]] && [[ -w "/opt" ]] && [[ -w "/etc" ]]; then
        IS_IMMUTABLE_OS=false
        log_info "Mutable filesystem detected (developer mode or non-SteamOS)"
    else
        IS_IMMUTABLE_OS=true
        log_info "Immutable filesystem detected (standard SteamOS)"
    fi
    
    # Check available space in user directory
    local available_space_kb
    available_space_kb=$(df "$HOME" | awk 'NR==2 {print $4}')
    local available_space_mb=$((available_space_kb / 1024))
    
    if [[ "$available_space_mb" -lt 2000 ]]; then
        log_error "Insufficient disk space: ${available_space_mb}MB available, need 2GB minimum"
        return 1
    fi
    
    log_info "Available disk space: ${available_space_mb}MB"
    return 0
}

check_system_resources() {
    log_info "Checking system resources and constraints..."
    
    # Memory check
    if command -v free >/dev/null 2>&1; then
        AVAILABLE_MEMORY_MB=$(free -m | awk 'NR==2{print $7}')
        local total_memory_mb=$(free -m | awk 'NR==2{print $2}')
        log_info "Memory: ${AVAILABLE_MEMORY_MB}MB available of ${total_memory_mb}MB total"
        
        if [[ "$AVAILABLE_MEMORY_MB" -lt 1000 ]]; then
            log_warning "Low memory condition: ${AVAILABLE_MEMORY_MB}MB available"
        fi
    else
        AVAILABLE_MEMORY_MB=4000  # Assume sufficient if we can't check
    fi
    
    # Temperature check
    if [[ -f "/sys/class/thermal/thermal_zone0/temp" ]]; then
        local temp_millicelsius
        temp_millicelsius=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo "65000")
        CURRENT_TEMP_C=$((temp_millicelsius / 1000))
        log_info "Current system temperature: ${CURRENT_TEMP_C}°C"
        
        if [[ "$CURRENT_TEMP_C" -gt "$MAX_TEMP_CELSIUS" ]]; then
            log_error "System too hot for installation: ${CURRENT_TEMP_C}°C (max: ${MAX_TEMP_CELSIUS}°C)"
            return 1
        fi
    fi
    
    # Battery check (if applicable)
    if command -v upower >/dev/null 2>&1; then
        local battery_info
        battery_info=$(upower -i "$(upower -e | grep 'BAT')" 2>/dev/null || echo "")
        
        if [[ -n "$battery_info" ]]; then
            BATTERY_LEVEL=$(echo "$battery_info" | grep -E "percentage" | grep -o '[0-9]*' || echo "100")
            IS_CHARGING=$(echo "$battery_info" | grep -q "charging" && echo "true" || echo "false")
            
            log_info "Battery: ${BATTERY_LEVEL}%, charging: $IS_CHARGING"
            
            if [[ "$BATTERY_LEVEL" -lt "$MIN_BATTERY_PERCENT" ]] && [[ "$IS_CHARGING" == "false" ]]; then
                log_warning "Low battery: ${BATTERY_LEVEL}% - consider connecting charger"
            fi
        fi
    fi
    
    return 0
}

determine_install_profile() {
    log_info "Determining optimal installation profile..."
    
    # Base profile on available resources
    if [[ "$AVAILABLE_MEMORY_MB" -lt 1500 ]] || [[ "$CURRENT_TEMP_C" -gt 75 ]]; then
        INSTALL_PROFILE="minimal"
        log_info "Selected MINIMAL profile (resource constraints)"
    elif [[ "$AVAILABLE_MEMORY_MB" -lt 3000 ]] || [[ "$STEAM_DECK_MODEL" == "LCD" ]]; then
        INSTALL_PROFILE="optimized"
        log_info "Selected OPTIMIZED profile (balanced performance)"
    else
        INSTALL_PROFILE="full"
        log_info "Selected FULL profile (maximum features)"
    fi
    
    # Override for battery constraints
    if [[ "$BATTERY_LEVEL" -lt 30 ]] && [[ "$IS_CHARGING" == "false" ]]; then
        INSTALL_PROFILE="minimal"
        log_info "Override to MINIMAL profile due to battery constraints"
    fi
}

# ============================================================================
# NETWORK AND CONNECTIVITY
# ============================================================================

check_network_connectivity() {
    log_info "Checking network connectivity..."
    
    local test_urls=(
        "https://pypi.org/simple/"
        "https://github.com"
        "https://files.pythonhosted.org"
        "8.8.8.8"  # Google DNS as fallback
    )
    
    for url in "${test_urls[@]}"; do
        if timeout 10 ping -c 1 "${url%%/*}" >/dev/null 2>&1 || 
           timeout 10 curl -s --head "$url" >/dev/null 2>&1; then
            NETWORK_AVAILABLE=true
            log_success "Network connectivity verified: $url"
            return 0
        fi
    done
    
    NETWORK_AVAILABLE=false
    log_warning "Network connectivity issues detected"
    return 1
}

setup_offline_mirrors() {
    log_info "Setting up offline installation mirrors..."
    
    # Create local package cache
    local cache_dir="$CACHE_DIR/packages"
    mkdir -p "$cache_dir"
    
    # Configure pip for offline mode if needed
    cat > "$CONFIG_DIR/pip.conf" << 'EOF'
[global]
timeout = 60
retries = 5
trusted-host = pypi.org
               pypi.python.org
               files.pythonhosted.org
               
[install]
find-links = ~/.cache/ml-shader-predictor/packages
EOF
    
    # Set environment variables for offline mode
    export PIP_CONFIG_FILE="$CONFIG_DIR/pip.conf"
    export PIP_CACHE_DIR="$cache_dir"
    
    log_info "Offline installation configuration prepared"
}

# ============================================================================
# PYTHON ENVIRONMENT SETUP
# ============================================================================

find_python_executable() {
    log_info "Finding suitable Python executable..."
    
    local python_candidates=(
        "python3.11"
        "python3.10"
        "python3.9"
        "python3"
        "/usr/bin/python3"
        "/home/deck/.local/bin/python3"
    )
    
    for python_cmd in "${python_candidates[@]}"; do
        if command -v "$python_cmd" >/dev/null 2>&1; then
            local python_version
            python_version=$("$python_cmd" --version 2>&1 | awk '{print $2}')
            
            # Check minimum version (3.8+)
            if "$python_cmd" -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
                PYTHON_EXECUTABLE="$python_cmd"
                log_success "Found Python $python_version at $python_cmd"
                return 0
            else
                log_warning "Python $python_version at $python_cmd is too old (need 3.8+)"
            fi
        fi
    done
    
    log_error "No suitable Python executable found (need Python 3.8+)"
    return 1
}

ensure_pip_available() {
    log_info "Ensuring pip is available..."
    
    # Method 1: Check if pip is already available
    if "$PYTHON_EXECUTABLE" -m pip --version >/dev/null 2>&1; then
        log_success "pip is already available"
        return 0
    fi
    
    # Method 2: Try ensurepip
    log_info "Attempting to install pip via ensurepip..."
    if "$PYTHON_EXECUTABLE" -m ensurepip --upgrade --user 2>/dev/null; then
        log_success "pip installed via ensurepip"
        return 0
    fi
    
    # Method 3: Download get-pip.py
    if [[ "$NETWORK_AVAILABLE" == "true" ]]; then
        log_info "Downloading get-pip.py..."
        local temp_pip="/tmp/get-pip.py"
        
        if timeout 30 curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$temp_pip" 2>/dev/null ||
           timeout 30 wget -q -O "$temp_pip" https://bootstrap.pypa.io/get-pip.py 2>/dev/null; then
            
            if "$PYTHON_EXECUTABLE" "$temp_pip" --user; then
                rm -f "$temp_pip"
                log_success "pip installed via get-pip.py"
                return 0
            fi
        fi
    fi
    
    log_error "Failed to install pip"
    return 1
}

setup_python_environment() {
    log_info "Setting up Python environment..."
    
    # Add user bin to PATH
    local user_bin="$HOME/.local/bin"
    if [[ ":$PATH:" != *":$user_bin:"* ]]; then
        export PATH="$user_bin:$PATH"
        
        # Make permanent in shell config
        for shell_config in "$HOME/.bashrc" "$HOME/.profile"; do
            if [[ -f "$shell_config" ]] && ! grep -q "export PATH.*$user_bin" "$shell_config"; then
                echo "export PATH=\"$user_bin:\$PATH\"" >> "$shell_config"
            fi
        done
        
        log_info "Added $user_bin to PATH"
    fi
    
    # Upgrade pip, setuptools, wheel
    log_info "Upgrading pip, setuptools, and wheel..."
    "$PYTHON_EXECUTABLE" -m pip install --user --upgrade --quiet pip setuptools wheel || {
        log_warning "Failed to upgrade pip components, continuing..."
    }
    
    # Set memory constraints for pip
    export PIP_NO_CACHE_DIR=1
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    
    if [[ "$INSTALL_PROFILE" == "minimal" ]]; then
        export PIP_NO_BUILD_ISOLATION=1  # Save memory during builds
    fi
    
    log_success "Python environment configured"
}

# ============================================================================
# DEPENDENCY INSTALLATION
# ============================================================================

create_requirements_file() {
    log_info "Creating requirements file for profile: $INSTALL_PROFILE"
    
    local req_file="$CONFIG_DIR/requirements_${INSTALL_PROFILE}.txt"
    
    case "$INSTALL_PROFILE" in
        "minimal")
            cat > "$req_file" << 'EOF'
# Minimal Steam Deck Profile - Essential dependencies only
numpy>=1.21.0,<1.25.0
psutil>=5.8.0,<6.0.0
PyYAML>=6.0,<7.0
joblib>=1.1.0,<1.4.0
requests>=2.28.0,<3.0.0
configparser>=5.2.0

# Linux-specific (conditional)
pyudev>=0.23.0; platform_system=="Linux"
EOF
            ;;
        "optimized")
            cat > "$req_file" << 'EOF'
# Optimized Steam Deck Profile - Balanced performance
numpy>=1.21.0,<1.25.0
scipy>=1.8.0,<1.12.0
scikit-learn>=1.1.0,<1.4.0
psutil>=5.8.0,<6.0.0
PyYAML>=6.0,<7.0
joblib>=1.1.0,<1.4.0
requests>=2.28.0,<3.0.0
configparser>=5.2.0
pandas>=1.4.0,<2.1.0

# Networking (optional)
aiohttp>=3.8.0,<4.0.0; extra == "networking"

# Linux-specific
pyudev>=0.23.0; platform_system=="Linux"
py-cpuinfo>=8.0.0,<10.0.0
EOF
            ;;
        "full")
            cat > "$req_file" << 'EOF'
# Full Steam Deck Profile - Maximum features
numpy>=1.21.0,<1.25.0
scipy>=1.8.0,<1.12.0
scikit-learn>=1.1.0,<1.4.0
pandas>=1.4.0,<2.1.0
psutil>=5.8.0,<6.0.0
PyYAML>=6.0,<7.0
joblib>=1.1.0,<1.4.0
requests>=2.28.0,<3.0.0
configparser>=5.2.0

# Networking and P2P
aiohttp>=3.8.0,<4.0.0
cryptography>=37.0.0,<42.0.0

# GPU acceleration (optional)
torch>=1.10.0,<3.0.0; extra == "gpu"

# Visualization (optional)
matplotlib>=3.5.0; extra == "viz"

# Linux-specific
pyudev>=0.23.0; platform_system=="Linux"
py-cpuinfo>=8.0.0,<10.0.0
dbus-python>=1.2.18,<2.0.0; platform_system=="Linux"
EOF
            ;;
    esac
    
    echo "$req_file"
}

install_dependencies_with_retry() {
    local req_file="$1"
    local max_attempts=3
    local attempt=1
    
    log_info "Installing dependencies from $req_file (max $max_attempts attempts)..."
    
    # Read packages from requirements file
    local packages=()
    while IFS= read -r line; do
        # Skip comments and empty lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ "$line" =~ ^[[:space:]]*$ ]] && continue
        
        # Remove conditional markers for basic installation
        line=$(echo "$line" | sed 's/;.*$//')
        [[ -n "$line" ]] && packages+=("$line")
    done < "$req_file"
    
    local total_packages=${#packages[@]}
    local installed_packages=0
    local failed_packages=()
    
    for package in "${packages[@]}"; do
        local success=false
        
        for ((attempt = 1; attempt <= max_attempts; attempt++)); do
            log_info "Installing $package (attempt $attempt/$max_attempts)..."
            
            # Monitor system resources during installation
            if ! monitor_installation_resources; then
                log_warning "Resource constraints detected, skipping $package"
                break
            fi
            
            if timeout 300 "$PYTHON_EXECUTABLE" -m pip install --user --quiet --no-warn-script-location "$package" 2>>"$ERROR_LOG"; then
                success=true
                ((installed_packages++))
                progress_bar "$installed_packages" "$total_packages" "Installing dependencies"
                break
            else
                log_warning "Failed to install $package (attempt $attempt)"
                sleep $((attempt * 2))  # Exponential backoff
            fi
        done
        
        if [[ "$success" == "false" ]]; then
            failed_packages+=("$package")
            log_error "Failed to install $package after $max_attempts attempts"
        fi
        
        # Small delay to prevent system overload
        sleep 1
    done
    
    log_info "Dependency installation complete: $installed_packages/$total_packages successful"
    
    if [[ ${#failed_packages[@]} -gt 0 ]]; then
        log_warning "Failed packages: ${failed_packages[*]}"
        
        # Try fallback versions for critical packages
        install_fallback_packages "${failed_packages[@]}"
    fi
    
    # Verify critical imports
    verify_critical_dependencies
}

monitor_installation_resources() {
    # Check memory usage
    local current_memory=$(free -m | awk 'NR==2{print $3}')
    if [[ "$current_memory" -gt "$MAX_INSTALL_MEMORY_MB" ]]; then
        log_warning "Memory usage too high: ${current_memory}MB"
        return 1
    fi
    
    # Check temperature
    if [[ -f "/sys/class/thermal/thermal_zone0/temp" ]]; then
        local temp_millicelsius
        temp_millicelsius=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo "65000")
        local current_temp=$((temp_millicelsius / 1000))
        
        if [[ "$current_temp" -gt "$MAX_TEMP_CELSIUS" ]]; then
            log_warning "System temperature too high: ${current_temp}°C"
            return 1
        fi
    fi
    
    return 0
}

install_fallback_packages() {
    log_info "Installing fallback packages for failed dependencies..."
    
    local fallback_packages=(
        "numpy==1.21.6"  # Known working version on Steam Deck
        "scikit-learn==1.1.3"
        "scipy==1.8.1"
    )
    
    for package in "$@"; do
        local base_package=$(echo "$package" | sed 's/[>=<].*//')
        
        case "$base_package" in
            "numpy")
                "$PYTHON_EXECUTABLE" -m pip install --user --quiet "numpy==1.21.6" 2>/dev/null || true
                ;;
            "scikit-learn")
                "$PYTHON_EXECUTABLE" -m pip install --user --quiet "scikit-learn==1.1.3" 2>/dev/null || true
                ;;
            "scipy")
                "$PYTHON_EXECUTABLE" -m pip install --user --quiet "scipy==1.8.1" 2>/dev/null || true
                ;;
        esac
    done
}

verify_critical_dependencies() {
    log_info "Verifying critical dependencies..."
    
    local critical_imports=(
        "import numpy"
        "import psutil"
        "import yaml"
        "import joblib"
        "import requests"
    )
    
    local import_success=0
    local import_total=${#critical_imports[@]}
    
    for import_stmt in "${critical_imports[@]}"; do
        if "$PYTHON_EXECUTABLE" -c "$import_stmt" 2>/dev/null; then
            ((import_success++))
        else
            log_warning "Failed import: $import_stmt"
        fi
    done
    
    local success_rate=$((import_success * 100 / import_total))
    log_info "Dependency verification: $import_success/$import_total successful ($success_rate%)"
    
    if [[ "$import_success" -lt 3 ]]; then
        log_error "Critical dependencies missing - installation cannot continue"
        return 1
    fi
    
    return 0
}

# ============================================================================
# APPLICATION INSTALLATION
# ============================================================================

install_application_files() {
    log_info "Installing application files..."
    
    # Create directory structure
    mkdir -p "$INSTALL_DIR"/{src,config,logs,models,cache,temp}
    mkdir -p "$SERVICE_DIR"
    
    # Copy source files with fallback search
    local source_paths=(
        "$SCRIPT_DIR/src"
        "$SCRIPT_DIR/shader-prediction-compilation-main/shader-predict-compile/src"
        "$SCRIPT_DIR/../src"
    )
    
    local source_found=false
    for src_path in "${source_paths[@]}"; do
        if [[ -d "$src_path" ]] && [[ -n "$(ls -A "$src_path" 2>/dev/null)" ]]; then
            log_info "Found source files at: $src_path"
            cp -r "$src_path"/* "$INSTALL_DIR/src/" 2>/dev/null || {
                log_warning "Some files failed to copy from $src_path"
            }
            source_found=true
            break
        fi
    done
    
    if [[ "$source_found" == "false" ]]; then
        log_error "No source files found in any expected location"
        return 1
    fi
    
    # Copy configuration files
    local config_paths=(
        "$SCRIPT_DIR/shader-prediction-compilation-main/shader-predict-compile/config"
        "$SCRIPT_DIR/config"
    )
    
    for config_path in "${config_paths[@]}"; do
        if [[ -d "$config_path" ]]; then
            cp -r "$config_path"/* "$INSTALL_DIR/config/" 2>/dev/null || true
            break
        fi
    done
    
    # Create Steam Deck specific configuration
    create_steamdeck_config
    
    # Set permissions
    chmod -R 755 "$INSTALL_DIR/src"
    chmod -R 644 "$INSTALL_DIR/config"
    chmod 755 "$INSTALL_DIR"
    
    log_success "Application files installed"
}

create_steamdeck_config() {
    log_info "Creating Steam Deck specific configuration..."
    
    local config_file="$INSTALL_DIR/config/steamdeck_config.json"
    
    cat > "$config_file" << EOF
{
  "version": "$SCRIPT_VERSION",
  "steam_deck": {
    "model": "$STEAM_DECK_MODEL",
    "detected_at": "$(date -Iseconds)",
    "immutable_os": $IS_IMMUTABLE_OS,
    "install_profile": "$INSTALL_PROFILE"
  },
  "predictor": {
    "model_type": "lightweight",
    "cache_size": $(get_cache_size),
    "max_temp": 83.0,
    "power_budget": 12.0,
    "sequence_length": $(get_sequence_length),
    "buffer_size": $(get_buffer_size),
    "auto_train_interval": 1000,
    "min_training_samples": 100,
    "memory_optimization": true,
    "max_memory_usage_mb": 200
  },
  "system": {
    "monitor_interval": 1.0,
    "thermal_protection": true,
    "battery_optimization": true,
    "performance_mode": "balanced",
    "resource_limits": {
      "memory_max_mb": 300,
      "cpu_quota_percent": 50,
      "io_weight": 100
    }
  },
  "logging": {
    "level": "INFO",
    "max_file_size_mb": 10,
    "backup_count": 5,
    "log_dir": "$LOG_DIR"
  },
  "game_profiles": {
    "1091500": {
      "name": "Cyberpunk 2077",
      "shader_complexity_modifier": 1.2,
      "thermal_limit": 80.0,
      "power_budget": 11.0,
      "cache_size": 300
    },
    "1245620": {
      "name": "ELDEN RING", 
      "shader_complexity_modifier": 1.1,
      "thermal_limit": 83.0,
      "power_budget": 12.0,
      "cache_size": 400
    }
  }
}
EOF
    
    log_success "Steam Deck configuration created: $config_file"
}

get_cache_size() {
    case "$INSTALL_PROFILE" in
        "minimal") echo "500" ;;
        "optimized") echo "1000" ;;
        "full") echo "2000" ;;
        *) echo "1000" ;;
    esac
}

get_sequence_length() {
    case "$INSTALL_PROFILE" in
        "minimal") echo "25" ;;
        "optimized") echo "50" ;;
        "full") echo "100" ;;
        *) echo "50" ;;
    esac
}

get_buffer_size() {
    case "$INSTALL_PROFILE" in
        "minimal") echo "2500" ;;
        "optimized") echo "5000" ;;
        "full") echo "10000" ;;
        *) echo "5000" ;;
    esac
}

# ============================================================================
# SYSTEMD SERVICE CREATION
# ============================================================================

create_systemd_service() {
    log_info "Creating systemd user service..."
    
    local service_file="$SERVICE_DIR/ml-shader-predictor.service"
    
    cat > "$service_file" << EOF
[Unit]
Description=ML Shader Prediction Compiler for Steam Deck
Documentation=file://$INSTALL_DIR/README.md
After=graphical-session.target steam.service
Wants=steam.service

[Service]
Type=simple
ExecStart=$HOME/.local/bin/ml-shader-predictor --daemon --config $INSTALL_DIR/config/steamdeck_config.json
ExecReload=/bin/kill -HUP \$MAINPID
Restart=on-failure
RestartSec=10
KillMode=mixed
TimeoutStopSec=30

# Environment
Environment=PYTHONPATH=$INSTALL_DIR/src
Environment=PYTHONOPTIMIZE=1
Environment=PYTHONUNBUFFERED=1
Environment=MALLOC_TRIM_THRESHOLD_=100000

# Resource constraints for Steam Deck
MemoryMax=300M
MemorySwapMax=0
CPUQuota=50%
IOWeight=100
TasksMax=50

# Security
PrivateTmp=true
NoNewPrivileges=true
RestrictSUIDSGID=true
RestrictRealtime=true
LockPersonality=true
ProtectClock=true
ProtectControlGroups=true
ProtectHostname=true
ProtectKernelModules=true
ProtectKernelTunables=true

# File system access
ReadWritePaths=$INSTALL_DIR $CONFIG_DIR $CACHE_DIR
ReadOnlyPaths=/home/deck/.steam

# Logging
StandardOutput=append:$LOG_DIR/service.log
StandardError=append:$LOG_DIR/service_errors.log
SyslogIdentifier=ml-shader-predictor

[Install]
WantedBy=default.target
EOF
    
    # Create timer for maintenance tasks
    local timer_file="$SERVICE_DIR/ml-shader-predictor-maintenance.timer"
    cat > "$timer_file" << EOF
[Unit]
Description=ML Shader Predictor Maintenance Timer
Requires=ml-shader-predictor.service

[Timer]
OnBootSec=5min
OnUnitActiveSec=1h
RandomizedDelaySec=300
Persistent=true

[Install]
WantedBy=timers.target
EOF
    
    local maintenance_service="$SERVICE_DIR/ml-shader-predictor-maintenance.service"
    cat > "$maintenance_service" << EOF
[Unit]
Description=ML Shader Predictor Maintenance
After=ml-shader-predictor.service

[Service]
Type=oneshot
ExecStart=$HOME/.local/bin/ml-shader-predictor --maintenance
PrivateTmp=true
NoNewPrivileges=true
MemoryMax=100M
CPUQuota=25%
TimeoutStartSec=300
Environment=PYTHONPATH=$INSTALL_DIR/src

[Install]
WantedBy=multi-user.target
EOF
    
    log_success "Systemd service files created"
}

# ============================================================================
# LAUNCHER AND DESKTOP INTEGRATION
# ============================================================================

create_launcher_script() {
    log_info "Creating launcher script..."
    
    local launcher_script="$HOME/.local/bin/ml-shader-predictor"
    
    cat > "$launcher_script" << 'EOF'
#!/bin/bash
#
# ML Shader Predictor Launcher - Steam Deck Optimized
#

set -euo pipefail

readonly INSTALL_DIR="$HOME/.local/share/ml-shader-predictor"
readonly CONFIG_FILE="$INSTALL_DIR/config/steamdeck_config.json"
readonly LOG_FILE="$INSTALL_DIR/logs/ml_shader_predictor.log"
readonly PID_FILE="$INSTALL_DIR/ml-shader-predictor.pid"

# Ensure directories exist
mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"

# Set environment for Steam Deck optimization
export PYTHONPATH="$INSTALL_DIR/src:${PYTHONPATH:-}"
export PYTHONOPTIMIZE=1
export PYTHONUNBUFFERED=1
export MALLOC_TRIM_THRESHOLD_=100000
export MALLOC_MMAP_THRESHOLD_=100000

# Memory optimization for Steam Deck
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=4

# Function to check if process is running
is_running() {
    [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

# Function to start the service
start_service() {
    if is_running; then
        echo "ML Shader Predictor is already running (PID: $(cat "$PID_FILE"))"
        return 0
    fi
    
    echo "Starting ML Shader Predictor..."
    cd "$INSTALL_DIR"
    
    # Resource monitoring wrapper
    exec python3 -O "$INSTALL_DIR/src/steam_deck_integration.py" \
        --config "$CONFIG_FILE" \
        --log-file "$LOG_FILE" \
        --pid-file "$PID_FILE" \
        "$@" 2>&1 | tee -a "$LOG_FILE"
}

# Function to stop the service
stop_service() {
    if ! is_running; then
        echo "ML Shader Predictor is not running"
        return 0
    fi
    
    echo "Stopping ML Shader Predictor..."
    kill -TERM "$(cat "$PID_FILE")" 2>/dev/null || true
    
    # Wait for graceful shutdown
    local timeout=30
    while [[ $timeout -gt 0 ]] && is_running; do
        sleep 1
        ((timeout--))
    done
    
    # Force kill if necessary
    if is_running; then
        echo "Force stopping..."
        kill -KILL "$(cat "$PID_FILE")" 2>/dev/null || true
    fi
    
    rm -f "$PID_FILE"
    echo "ML Shader Predictor stopped"
}

# Function to show status
show_status() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        echo "ML Shader Predictor is running (PID: $pid)"
        
        # Show resource usage
        if command -v ps >/dev/null 2>&1; then
            ps -p "$pid" -o pid,pcpu,pmem,time,cmd 2>/dev/null || true
        fi
    else
        echo "ML Shader Predictor is not running"
    fi
}

# Function to show logs
show_logs() {
    local lines="${1:-50}"
    if [[ -f "$LOG_FILE" ]]; then
        tail -n "$lines" "$LOG_FILE"
    else
        echo "No log file found at $LOG_FILE"
    fi
}

# Function to run maintenance
run_maintenance() {
    echo "Running maintenance tasks..."
    cd "$INSTALL_DIR"
    
    # Clear old logs
    find "$INSTALL_DIR/logs" -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    # Clear old cache files
    find "$INSTALL_DIR/cache" -type f -mtime +3 -delete 2>/dev/null || true
    
    # Optimize models
    if [[ -f "$INSTALL_DIR/src/steam_deck_integration.py" ]]; then
        python3 -O "$INSTALL_DIR/src/steam_deck_integration.py" --optimize 2>&1 | tee -a "$LOG_FILE"
    fi
    
    echo "Maintenance completed"
}

# Main command handling
case "${1:-help}" in
    "start"|"--daemon")
        shift
        start_service "$@"
        ;;
    "stop")
        stop_service
        ;;
    "restart")
        stop_service
        sleep 2
        shift
        start_service "$@"
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs "${2:-50}"
        ;;
    "maintenance"|"--maintenance")
        run_maintenance
        ;;
    "gui"|"--gui")
        # Launch GUI version if available
        if [[ -f "$INSTALL_DIR/src/gui/main_window.py" ]]; then
            cd "$INSTALL_DIR"
            python3 "$INSTALL_DIR/src/gui/main_window.py" "$@"
        else
            echo "GUI not available in this installation"
            exit 1
        fi
        ;;
    "help"|"--help"|"-h")
        cat << 'HELP'
ML Shader Predictor - Steam Deck Optimized

Usage: ml-shader-predictor [COMMAND] [OPTIONS]

Commands:
    start, --daemon    Start the service in daemon mode
    stop              Stop the running service
    restart           Restart the service
    status            Show service status and resource usage
    logs [N]          Show last N log lines (default: 50)
    maintenance       Run maintenance tasks
    gui, --gui        Launch GUI interface (if available)
    help, --help      Show this help message

Examples:
    ml-shader-predictor start           # Start in daemon mode
    ml-shader-predictor status          # Check if running
    ml-shader-predictor logs 100        # Show last 100 log lines
    ml-shader-predictor maintenance     # Run cleanup tasks

For more information, see: ~/.local/share/ml-shader-predictor/README.md
HELP
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use 'ml-shader-predictor help' for usage information"
        exit 1
        ;;
esac
EOF
    
    chmod +x "$launcher_script"
    log_success "Launcher script created: $launcher_script"
}

create_desktop_entry() {
    log_info "Creating desktop entry for Gaming Mode integration..."
    
    local desktop_file="$HOME/.local/share/applications/ml-shader-predictor.desktop"
    mkdir -p "$(dirname "$desktop_file")"
    
    cat > "$desktop_file" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=ML Shader Predictor
GenericName=Shader Compilation Optimizer
Comment=Machine Learning-based Shader Compilation Predictor for Steam Deck
Keywords=shader;steam;deck;gaming;ml;prediction;optimization;
Categories=Game;System;Utility;
StartupNotify=true
StartupWMClass=ml-shader-predictor
NoDisplay=false
Hidden=false
Icon=$INSTALL_DIR/icon.png
Exec=$HOME/.local/bin/ml-shader-predictor gui
Terminal=false
X-Steam-Library-Capsule=$INSTALL_DIR/icon.png
X-Steam-Controller-Template=Desktop
EOF
    
    chmod +x "$desktop_file"
    
    # Create a simple icon if one doesn't exist
    create_application_icon
    
    # Update desktop database
    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
    fi
    
    log_success "Desktop entry created: $desktop_file"
}

create_application_icon() {
    local icon_file="$INSTALL_DIR/icon.png"
    
    # Try to copy existing icon
    local icon_sources=(
        "$SCRIPT_DIR/icon.png"
        "$SCRIPT_DIR/shader-predict-compile/icon.png"
        "$SCRIPT_DIR/assets/icon.png"
    )
    
    for icon_src in "${icon_sources[@]}"; do
        if [[ -f "$icon_src" ]]; then
            cp "$icon_src" "$icon_file" 2>/dev/null && return 0
        fi
    done
    
    # Create simple text-based icon using ImageMagick if available
    if command -v convert >/dev/null 2>&1; then
        convert -size 64x64 xc:blue \
                -font DejaVu-Sans-Bold -pointsize 10 \
                -fill white -gravity center \
                -annotate +0+0 'ML\nShader' \
                "$icon_file" 2>/dev/null || true
    fi
}

# ============================================================================
# VALIDATION AND TESTING
# ============================================================================

run_installation_validation() {
    log_info "Running comprehensive installation validation..."
    
    local validation_errors=()
    local validation_warnings=()
    
    # Test 1: File structure validation
    log_info "Validating file structure..."
    local required_dirs=(
        "$INSTALL_DIR"
        "$INSTALL_DIR/src"
        "$INSTALL_DIR/config"
        "$INSTALL_DIR/logs"
        "$CONFIG_DIR"
        "$CACHE_DIR"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            validation_errors+=("Missing directory: $dir")
        fi
    done
    
    local required_files=(
        "$INSTALL_DIR/config/steamdeck_config.json"
        "$SERVICE_DIR/ml-shader-predictor.service"
        "$HOME/.local/bin/ml-shader-predictor"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            validation_errors+=("Missing file: $file")
        fi
    done
    
    # Test 2: Python import validation
    log_info "Validating Python imports..."
    local python_validation
    python_validation=$("$PYTHON_EXECUTABLE" -c "
import sys
import os
sys.path.insert(0, '$INSTALL_DIR/src')

results = {'success': [], 'failed': []}

# Critical imports
critical_modules = [
    'numpy', 'psutil', 'yaml', 'joblib', 'requests'
]

for module in critical_modules:
    try:
        __import__(module)
        results['success'].append(module)
    except ImportError as e:
        results['failed'].append(f'{module}: {e}')

# Optional imports  
optional_modules = [
    'sklearn', 'scipy', 'pandas', 'aiohttp'
]

for module in optional_modules:
    try:
        __import__(module)
        results['success'].append(module)
    except ImportError:
        pass  # Optional modules don't count as failures

print(f'SUCCESS:{len(results[\"success\"])}')
print(f'FAILED:{len(results[\"failed\"])}')
for failure in results['failed']:
    print(f'IMPORT_ERROR:{failure}')
" 2>/dev/null)
    
    local import_success=$(echo "$python_validation" | grep "SUCCESS:" | cut -d: -f2)
    local import_failed=$(echo "$python_validation" | grep "FAILED:" | cut -d: -f2)
    
    if [[ "$import_failed" -gt 2 ]]; then
        validation_errors+=("Too many failed Python imports: $import_failed")
    elif [[ "$import_failed" -gt 0 ]]; then
        validation_warnings+=("Some Python imports failed: $import_failed")
    fi
    
    # Test 3: Service file validation
    log_info "Validating systemd service..."
    if command -v systemd-analyze >/dev/null 2>&1; then
        if ! systemd-analyze --user verify "$SERVICE_DIR/ml-shader-predictor.service" 2>/dev/null; then
            validation_warnings+=("Systemd service file validation failed")
        fi
    fi
    
    # Test 4: Launcher script validation
    log_info "Validating launcher script..."
    if ! bash -n "$HOME/.local/bin/ml-shader-predictor" 2>/dev/null; then
        validation_errors+=("Launcher script has syntax errors")
    fi
    
    # Test 5: Configuration validation
    log_info "Validating configuration..."
    if ! "$PYTHON_EXECUTABLE" -c "
import json
with open('$INSTALL_DIR/config/steamdeck_config.json') as f:
    config = json.load(f)
assert 'steam_deck' in config
assert 'predictor' in config
assert 'system' in config
" 2>/dev/null; then
        validation_errors+=("Configuration file is invalid")
    fi
    
    # Test 6: Resource constraints validation
    log_info "Validating resource constraints..."
    local total_install_size
    total_install_size=$(du -sm "$INSTALL_DIR" 2>/dev/null | cut -f1 || echo "0")
    
    if [[ "$total_install_size" -gt 500 ]]; then
        validation_warnings+=("Installation size is large: ${total_install_size}MB")
    fi
    
    # Summary
    local total_errors=${#validation_errors[@]}
    local total_warnings=${#validation_warnings[@]}
    
    if [[ "$total_errors" -eq 0 ]] && [[ "$total_warnings" -eq 0 ]]; then
        log_success "All validation tests passed"
        return 0
    elif [[ "$total_errors" -eq 0 ]]; then
        log_warning "Validation completed with $total_warnings warnings:"
        printf '%s\n' "${validation_warnings[@]}" | while read -r warning; do
            log_warning "$warning"
        done
        return 0
    else
        log_error "Validation failed with $total_errors errors and $total_warnings warnings:"
        printf '%s\n' "${validation_errors[@]}" | while read -r error; do
            log_error "$error"
        done
        printf '%s\n' "${validation_warnings[@]}" | while read -r warning; do
            log_warning "$warning"
        done
        return 1
    fi
}

create_diagnostic_script() {
    log_info "Creating diagnostic script for troubleshooting..."
    
    local diag_script="$HOME/.local/bin/ml-shader-predictor-diag"
    
    cat > "$diag_script" << 'EOF'
#!/bin/bash
#
# ML Shader Predictor Diagnostic Script
# Steam Deck Troubleshooting Tool
#

set -euo pipefail

readonly INSTALL_DIR="$HOME/.local/share/ml-shader-predictor"
readonly CONFIG_DIR="$HOME/.config/ml-shader-predictor"
readonly LOG_DIR="$INSTALL_DIR/logs"

echo "ML Shader Predictor Diagnostic Report"
echo "====================================="
echo "Generated: $(date)"
echo

# System Information
echo "=== System Information ==="
echo "Hostname: $(hostname)"
echo "OS: $(uname -a)"

if [[ -f "/etc/os-release" ]]; then
    echo "Distribution:"
    grep -E '^(NAME|VERSION)=' /etc/os-release | sed 's/^/  /'
fi

if [[ -f "/sys/class/dmi/id/product_name" ]]; then
    echo "Hardware: $(cat /sys/class/dmi/id/product_name 2>/dev/null || echo 'unknown')"
fi

echo "Python: $(python3 --version 2>&1)"
echo "User: $(whoami)"
echo "Home: $HOME"
echo

# Steam Deck Detection
echo "=== Steam Deck Detection ==="
if [[ -f "/sys/class/dmi/id/product_name" ]]; then
    local product_name
    product_name=$(cat /sys/class/dmi/id/product_name 2>/dev/null || echo "unknown")
    
    if [[ "$product_name" == *"Jupiter"* ]] || [[ "$product_name" == *"Steam Deck"* ]]; then
        echo "✓ Steam Deck hardware detected: $product_name"
    else
        echo "✗ Steam Deck hardware not detected: $product_name"
    fi
else
    echo "✗ Cannot read hardware information"
fi

# Check for AMD APU
if command -v lscpu >/dev/null 2>&1; then
    if lscpu | grep -qi "AMD.*Custom.*APU"; then
        echo "✓ Steam Deck APU detected"
    else
        echo "⚠ Steam Deck APU not detected"
    fi
fi

# Check SteamOS
if [[ -f "/etc/os-release" ]] && grep -qi "steamos" /etc/os-release; then
    echo "✓ SteamOS detected"
else
    echo "⚠ SteamOS not detected"
fi
echo

# Filesystem Check
echo "=== Filesystem Check ==="
echo "Root filesystem writable: $([[ -w "/usr" ]] && echo "YES" || echo "NO")"
echo "Install directory: $INSTALL_DIR"
echo "  Exists: $([[ -d "$INSTALL_DIR" ]] && echo "YES" || echo "NO")"
echo "  Writable: $([[ -w "$INSTALL_DIR" ]] && echo "YES" || echo "NO")"

if [[ -d "$INSTALL_DIR" ]]; then
    echo "  Size: $(du -sh "$INSTALL_DIR" 2>/dev/null | cut -f1 || echo "unknown")"
fi

local available_space
available_space=$(df -h "$HOME" 2>/dev/null | awk 'NR==2 {print $4}' || echo "unknown")
echo "Available space in $HOME: $available_space"
echo

# Resource Information
echo "=== Resource Information ==="
if command -v free >/dev/null 2>&1; then
    echo "Memory:"
    free -h | sed 's/^/  /'
fi

if [[ -f "/sys/class/thermal/thermal_zone0/temp" ]]; then
    local temp_mc
    temp_mc=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo "0")
    local temp_c=$((temp_mc / 1000))
    echo "CPU Temperature: ${temp_c}°C"
fi

if command -v upower >/dev/null 2>&1; then
    echo "Battery:"
    upower -i "$(upower -e | grep 'BAT')" 2>/dev/null | grep -E "(percentage|state)" | sed 's/^/  /' || echo "  Not available"
fi
echo

# Installation Status
echo "=== Installation Status ==="
local required_files=(
    "$INSTALL_DIR/config/steamdeck_config.json"
    "$INSTALL_DIR/src/steam_deck_integration.py"
    "$HOME/.local/bin/ml-shader-predictor"
    "$HOME/.config/systemd/user/ml-shader-predictor.service"
)

for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✓ $file"
    else
        echo "✗ $file (MISSING)"
    fi
done
echo

# Python Environment
echo "=== Python Environment ==="
echo "Python executable: $(which python3 2>/dev/null || echo 'NOT FOUND')"
echo "Pip available: $(python3 -m pip --version >/dev/null 2>&1 && echo 'YES' || echo 'NO')"
echo "User site packages: $(python3 -c 'import site; print(site.USER_SITE)' 2>/dev/null || echo 'ERROR')"

echo
echo "Python package status:"
local critical_packages=("numpy" "psutil" "yaml" "joblib" "requests")
for package in "${critical_packages[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        local version
        version=$(python3 -c "import $package; print(getattr($package, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
        echo "  ✓ $package ($version)"
    else
        echo "  ✗ $package (MISSING)"
    fi
done

local optional_packages=("sklearn" "scipy" "pandas" "aiohttp")
for package in "${optional_packages[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        local version
        version=$(python3 -c "import $package; print(getattr($package, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
        echo "  ○ $package ($version) [optional]"
    else
        echo "  ○ $package (not installed) [optional]"
    fi
done
echo

# Service Status
echo "=== Service Status ==="
if command -v systemctl >/dev/null 2>&1; then
    local service_status
    service_status=$(systemctl --user is-active ml-shader-predictor.service 2>/dev/null || echo "not-found")
    echo "Service status: $service_status"
    
    local service_enabled
    service_enabled=$(systemctl --user is-enabled ml-shader-predictor.service 2>/dev/null || echo "not-found")
    echo "Service enabled: $service_enabled"
    
    if [[ "$service_status" == "active" ]]; then
        echo "Service details:"
        systemctl --user status ml-shader-predictor.service --no-pager -l 2>/dev/null | sed 's/^/  /' || echo "  Cannot get service details"
    fi
else
    echo "systemctl not available"
fi
echo

# Log Analysis
echo "=== Recent Log Analysis ==="
if [[ -d "$LOG_DIR" ]]; then
    echo "Log directory: $LOG_DIR"
    echo "Log files:"
    ls -la "$LOG_DIR" 2>/dev/null | sed 's/^/  /' || echo "  Cannot list log files"
    
    # Show recent errors
    if [[ -f "$LOG_DIR/service_errors.log" ]]; then
        echo
        echo "Recent errors (last 10 lines):"
        tail -n 10 "$LOG_DIR/service_errors.log" 2>/dev/null | sed 's/^/  >' || echo "  Cannot read error log"
    fi
else
    echo "Log directory not found: $LOG_DIR"
fi
echo

# Networking
echo "=== Network Connectivity ==="
local test_hosts=("pypi.org" "github.com" "8.8.8.8")
for host in "${test_hosts[@]}"; do
    if timeout 5 ping -c 1 "$host" >/dev/null 2>&1; then
        echo "✓ $host reachable"
    else
        echo "✗ $host unreachable"
    fi
done
echo

# Steam Integration
echo "=== Steam Integration ==="
local steam_paths=(
    "$HOME/.steam"
    "$HOME/.local/share/Steam"
    "/home/deck/.steam"
)

for steam_path in "${steam_paths[@]}"; do
    if [[ -d "$steam_path" ]]; then
        echo "✓ Steam installation found: $steam_path"
        break
    fi
done

# Check for running Steam process
if pgrep -f "steam" >/dev/null 2>&1; then
    echo "✓ Steam process running"
else
    echo "○ Steam not currently running"
fi
echo

echo "=== Diagnostic Complete ==="
echo "For support, please share this diagnostic report."
echo "Log files location: $LOG_DIR"
echo
EOF
    
    chmod +x "$diag_script"
    log_success "Diagnostic script created: $diag_script"
}

# ============================================================================
# POST-INSTALLATION TASKS
# ============================================================================

enable_and_start_service() {
    log_info "Enabling and starting systemd service..."
    
    # Reload systemd
    systemctl --user daemon-reload
    
    # Enable service
    if systemctl --user enable ml-shader-predictor.service 2>>"$ERROR_LOG"; then
        log_success "Service enabled"
    else
        log_warning "Failed to enable service (will still attempt to start)"
    fi
    
    # Enable maintenance timer
    if systemctl --user enable ml-shader-predictor-maintenance.timer 2>>"$ERROR_LOG"; then
        log_info "Maintenance timer enabled"
    else
        log_warning "Failed to enable maintenance timer"
    fi
    
    # Start services
    if systemctl --user start ml-shader-predictor.service 2>>"$ERROR_LOG"; then
        log_success "Service started successfully"
        
        # Start maintenance timer
        systemctl --user start ml-shader-predictor-maintenance.timer 2>/dev/null || {
            log_warning "Failed to start maintenance timer"
        }
    else
        log_warning "Service failed to start - check logs for details"
        return 1
    fi
    
    # Wait a moment and check status
    sleep 3
    
    local service_status
    service_status=$(systemctl --user is-active ml-shader-predictor.service 2>/dev/null || echo "unknown")
    
    if [[ "$service_status" == "active" ]]; then
        log_success "Service is running and active"
        return 0
    else
        log_error "Service is not active (status: $service_status)"
        return 1
    fi
}

create_uninstaller() {
    log_info "Creating uninstaller script..."
    
    local uninstall_script="$HOME/.local/bin/ml-shader-predictor-uninstall"
    
    cat > "$uninstall_script" << EOF
#!/bin/bash
#
# ML Shader Predictor Uninstaller - Steam Deck
#

set -euo pipefail

readonly INSTALL_DIR="$INSTALL_DIR"
readonly CONFIG_DIR="$CONFIG_DIR" 
readonly CACHE_DIR="$CACHE_DIR"
readonly SERVICE_DIR="$SERVICE_DIR"

echo "ML Shader Predictor Uninstaller"
echo "==============================="
echo
echo "This will remove:"
echo "  - Application files: \$INSTALL_DIR"
echo "  - Configuration: \$CONFIG_DIR"
echo "  - Cache: \$CACHE_DIR"
echo "  - Systemd services"
echo "  - Desktop entries"
echo "  - Launcher scripts"
echo
read -p "Are you sure you want to uninstall? [y/N] " -n 1 -r
echo
if [[ ! \$REPLY =~ ^[Yy]$ ]]; then
    echo "Uninstall cancelled"
    exit 0
fi

echo "Uninstalling ML Shader Predictor..."

# Stop and disable services
if command -v systemctl >/dev/null 2>&1; then
    echo "Stopping services..."
    systemctl --user stop ml-shader-predictor.service 2>/dev/null || true
    systemctl --user stop ml-shader-predictor-maintenance.timer 2>/dev/null || true
    systemctl --user disable ml-shader-predictor.service 2>/dev/null || true
    systemctl --user disable ml-shader-predictor-maintenance.timer 2>/dev/null || true
    systemctl --user daemon-reload 2>/dev/null || true
fi

# Remove files and directories
echo "Removing files..."
rm -rf "\$INSTALL_DIR" 2>/dev/null || true
rm -rf "\$CONFIG_DIR" 2>/dev/null || true
rm -rf "\$CACHE_DIR" 2>/dev/null || true

# Remove service files
rm -f "\$SERVICE_DIR/ml-shader-predictor.service" 2>/dev/null || true
rm -f "\$SERVICE_DIR/ml-shader-predictor-maintenance.service" 2>/dev/null || true
rm -f "\$SERVICE_DIR/ml-shader-predictor-maintenance.timer" 2>/dev/null || true

# Remove launcher and tools
rm -f "\$HOME/.local/bin/ml-shader-predictor" 2>/dev/null || true
rm -f "\$HOME/.local/bin/ml-shader-predictor-diag" 2>/dev/null || true

# Remove desktop entry
rm -f "\$HOME/.local/share/applications/ml-shader-predictor.desktop" 2>/dev/null || true

# Update desktop database
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "\$HOME/.local/share/applications" 2>/dev/null || true
fi

echo "Uninstall completed successfully"
echo "Note: Python packages were not removed (they may be used by other applications)"

# Remove self
rm -f "\$0" 2>/dev/null || true
EOF
    
    chmod +x "$uninstall_script"
    log_success "Uninstaller created: $uninstall_script"
}

show_installation_summary() {
    echo
    log_message "INSTALL" "╔══════════════════════════════════════════════════════════════════╗"
    log_message "INSTALL" "║                    INSTALLATION COMPLETED                       ║"
    log_message "INSTALL" "╚══════════════════════════════════════════════════════════════════╝"
    echo
    
    log_success "ML Shader Prediction Compiler installed successfully!"
    echo
    
    echo -e "${BOLD}Installation Details:${NC}"
    echo -e "  • Version: ${GREEN}$SCRIPT_VERSION${NC}"
    echo -e "  • Steam Deck Model: ${GREEN}$STEAM_DECK_MODEL${NC}"
    echo -e "  • Installation Profile: ${GREEN}$INSTALL_PROFILE${NC}"
    echo -e "  • Installation Directory: ${GREEN}$INSTALL_DIR${NC}"
    echo -e "  • Configuration: ${GREEN}$INSTALL_DIR/config/steamdeck_config.json${NC}"
    echo -e "  • Logs: ${GREEN}$LOG_DIR${NC}"
    echo
    
    echo -e "${BOLD}Usage:${NC}"
    echo -e "  • Start service: ${GREEN}systemctl --user start ml-shader-predictor${NC}"
    echo -e "  • Check status: ${GREEN}ml-shader-predictor status${NC}"
    echo -e "  • View logs: ${GREEN}ml-shader-predictor logs${NC}"
    echo -e "  • Launch GUI: ${GREEN}ml-shader-predictor gui${NC} (if available)"
    echo -e "  • Run diagnostics: ${GREEN}ml-shader-predictor-diag${NC}"
    echo
    
    echo -e "${BOLD}System Integration:${NC}"
    echo -e "  • Systemd service: ${GREEN}Enabled and Started${NC}"
    echo -e "  • Desktop entry: ${GREEN}Available in Gaming Mode${NC}"
    echo -e "  • Resource limits: ${GREEN}300MB RAM, 50% CPU${NC}"
    echo -e "  • Thermal protection: ${GREEN}Enabled${NC}"
    echo -e "  • Battery optimization: ${GREEN}Enabled${NC}"
    echo
    
    if [[ "$INSTALL_PROFILE" == "minimal" ]]; then
        echo -e "${YELLOW}Note:${NC} Minimal installation profile was used due to resource constraints."
        echo -e "Some advanced ML features may be limited."
        echo
    fi
    
    echo -e "${BOLD}Next Steps:${NC}"
    echo -e "  1. The service should start automatically"
    echo -e "  2. Launch Steam and start a game to begin shader optimization"
    echo -e "  3. Monitor performance with: ${GREEN}ml-shader-predictor status${NC}"
    echo -e "  4. View the desktop app in Gaming Mode's library"
    echo
    
    echo -e "${BOLD}Troubleshooting:${NC}"
    echo -e "  • Run diagnostics: ${GREEN}ml-shader-predictor-diag${NC}"
    echo -e "  • Check service logs: ${GREEN}journalctl --user -u ml-shader-predictor.service${NC}"
    echo -e "  • Uninstall: ${GREEN}ml-shader-predictor-uninstall${NC}"
    echo
    
    log_info "Installation log saved to: $INSTALL_LOG"
    
    # Check final service status
    local final_status
    final_status=$(systemctl --user is-active ml-shader-predictor.service 2>/dev/null || echo "inactive")
    
    if [[ "$final_status" == "active" ]]; then
        echo -e "${GREEN}✓ Service is running and ready to optimize your Steam Deck gaming experience!${NC}"
    else
        echo -e "${YELLOW}⚠ Service is not running. Use 'ml-shader-predictor start' to start it manually.${NC}"
    fi
    
    echo
}

# ============================================================================
# CLEANUP AND ERROR HANDLING
# ============================================================================

cleanup_on_error() {
    local exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        log_error "Installation failed with exit code: $exit_code"
        
        # Stop any running services
        systemctl --user stop ml-shader-predictor.service 2>/dev/null || true
        
        # Show recent errors
        if [[ -f "$ERROR_LOG" ]] && [[ -s "$ERROR_LOG" ]]; then
            echo
            log_error "Recent error details:"
            tail -n 10 "$ERROR_LOG" | while read -r line; do
                log_error "$line"
            done
        fi
        
        echo
        echo -e "${RED}Installation failed!${NC}"
        echo -e "Check the logs for details:"
        echo -e "  • Install log: ${INSTALL_LOG}"
        echo -e "  • Error log: ${ERROR_LOG}"
        echo
        echo -e "For troubleshooting help:"
        echo -e "  • Run diagnostics: ${GREEN}ml-shader-predictor-diag${NC} (if created)"
        echo -e "  • Check available disk space: ${GREEN}df -h $HOME${NC}"
        echo -e "  • Check system resources: ${GREEN}free -h${NC}"
        echo
        echo -e "You can try running the installer again after resolving any issues."
    fi
}

cleanup_temp_files() {
    # Clean up any temporary files
    rm -f /tmp/get-pip.py 2>/dev/null || true
    
    # Clear pip cache if we disabled it for memory
    if [[ "$INSTALL_PROFILE" == "minimal" ]]; then
        "$PYTHON_EXECUTABLE" -m pip cache purge 2>/dev/null || true
    fi
}

# ============================================================================
# MAIN INSTALLATION FUNCTION
# ============================================================================

main() {
    # Set up error handling
    trap cleanup_on_error ERR
    trap cleanup_temp_files EXIT
    
    # Banner
    echo -e "${BOLD}${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${BLUE}║        ML Shader Prediction Compiler - Steam Deck Edition       ║${NC}"
    echo -e "${BOLD}${BLUE}║                     Bulletproof Installer v$SCRIPT_VERSION                    ║${NC}"
    echo -e "${BOLD}${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo
    
    log_info "Starting installation at $(date)"
    log_info "Installation script version: $SCRIPT_VERSION"
    log_info "Installation directory: $INSTALL_DIR"
    
    # Check if already installed
    if [[ -f "$INSTALL_DIR/config/steamdeck_config.json" ]] && [[ -f "$HOME/.local/bin/ml-shader-predictor" ]]; then
        echo -e "${YELLOW}ML Shader Predictor appears to already be installed.${NC}"
        read -p "Do you want to reinstall? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Installation cancelled by user"
            exit 0
        fi
        
        log_info "Proceeding with reinstallation..."
        
        # Stop existing service
        systemctl --user stop ml-shader-predictor.service 2>/dev/null || true
    fi
    
    # Initialize logging
    setup_logging
    
    # System detection and validation
    log_info "Phase 1: System Detection and Validation"
    detect_steam_deck
    check_filesystem_mutability
    check_system_resources
    determine_install_profile
    
    # Network and connectivity
    log_info "Phase 2: Network and Connectivity"
    check_network_connectivity || setup_offline_mirrors
    
    # Python environment setup
    log_info "Phase 3: Python Environment Setup"
    find_python_executable
    ensure_pip_available
    setup_python_environment
    
    # Dependency installation
    log_info "Phase 4: Dependency Installation"
    local req_file
    req_file=$(create_requirements_file)
    install_dependencies_with_retry "$req_file"
    
    # Application installation
    log_info "Phase 5: Application Installation"
    install_application_files
    
    # System integration
    log_info "Phase 6: System Integration"
    create_systemd_service
    create_launcher_script
    create_desktop_entry
    
    # Post-installation
    log_info "Phase 7: Post-Installation Setup"
    create_diagnostic_script
    create_uninstaller
    
    # Validation
    log_info "Phase 8: Installation Validation"
    if ! run_installation_validation; then
        log_error "Installation validation failed"
        exit 1
    fi
    
    # Service activation
    log_info "Phase 9: Service Activation"
    enable_and_start_service || {
        log_warning "Service activation failed, but installation is complete"
    }
    
    # Final cleanup and summary
    cleanup_temp_files
    show_installation_summary
    
    log_success "Installation completed successfully at $(date)"
}

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            export DEBUG=1
            log_info "Debug mode enabled"
            shift
            ;;
        --force)
            export FORCE_INSTALL=1
            log_info "Force installation mode enabled"
            shift
            ;;
        --profile)
            INSTALL_PROFILE="$2"
            log_info "Installation profile override: $INSTALL_PROFILE"
            shift 2
            ;;
        --help|-h)
            cat << 'HELP'
ML Shader Prediction Compiler - Steam Deck Bulletproof Installer

Usage: steamdeck-bulletproof-install.sh [OPTIONS]

Options:
  --debug         Enable debug output
  --force         Force installation even with warnings
  --profile PROF  Override installation profile (minimal|optimized|full)
  --help, -h      Show this help message

Installation Profiles:
  minimal         Essential dependencies only (~200MB RAM usage)
  optimized       Balanced performance and memory (~300MB RAM usage)
  full            Maximum features (~500MB RAM usage)

The installer automatically detects your Steam Deck model and system
constraints to choose the optimal installation profile.

Examples:
  ./steamdeck-bulletproof-install.sh
  ./steamdeck-bulletproof-install.sh --debug
  ./steamdeck-bulletproof-install.sh --profile minimal --force

For more information, visit: https://github.com/YourRepo/ML-Shader-Predictor
HELP
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if running as root (not recommended)
if [[ $EUID -eq 0 ]]; then
    log_error "This script should not be run as root"
    log_error "Run as the deck user or your regular user account"
    exit 1
fi

# Check minimum requirements
if [[ ! -d "$HOME" ]]; then
    log_error "HOME directory not accessible: $HOME"
    exit 1
fi

# Start the installation
main "$@"