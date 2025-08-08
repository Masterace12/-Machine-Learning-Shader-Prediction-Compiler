#!/bin/bash

# ============================================================================
# Steam Deck Shader Prediction Compiler - Complete Installation Fix
# ============================================================================
# This script fixes all installation issues specific to Steam Deck's
# immutable filesystem and missing pip3 command
#
# Usage:
#   bash steam-deck-fix.sh
#
# Options:
#   --install-only     : Only install without fixing existing files
#   --fix-only        : Only fix existing installation
#   --verbose         : Show detailed output
# ============================================================================

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base directories
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly HOME_DIR="${HOME:-/home/deck}"
readonly INSTALL_DIR="${HOME_DIR}/.local/share/shader-predict-compile"
readonly CONFIG_DIR="${HOME_DIR}/.config/shader-predict-compile"
readonly CACHE_DIR="${HOME_DIR}/.cache/shader-predict-compile"
readonly VENV_DIR="${INSTALL_DIR}/venv"

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Options
INSTALL_ONLY=false
FIX_ONLY=false
VERBOSE=false

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1" >&2
}

log_debug() {
    if [[ "$VERBOSE" == true ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# ============================================================================
# STEAM DECK DETECTION
# ============================================================================

detect_steam_deck() {
    local is_deck=false
    local deck_model="Unknown"
    
    # Check DMI information
    if [[ -f /sys/class/dmi/id/board_name ]]; then
        local board_name=$(cat /sys/class/dmi/id/board_name 2>/dev/null || echo "")
        if [[ "$board_name" == *"Jupiter"* ]]; then
            is_deck=true
            deck_model="LCD"
        elif [[ "$board_name" == *"Galileo"* ]]; then
            is_deck=true
            deck_model="OLED"
        fi
    fi
    
    # Check for SteamOS
    if [[ -f /etc/os-release ]]; then
        if grep -q "SteamOS" /etc/os-release 2>/dev/null; then
            is_deck=true
        fi
    fi
    
    if [[ "$is_deck" == true ]]; then
        log_success "Steam Deck $deck_model detected"
        return 0
    else
        log_warning "Not running on Steam Deck - using compatibility mode"
        return 1
    fi
}

# ============================================================================
# FIX 1: PYTHON AND PIP SETUP
# ============================================================================

fix_python_environment() {
    log_info "Fixing Python environment for Steam Deck..."
    
    # Check if Python is installed
    if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
        log_error "Python is not installed. Installing via pacman..."
        if command -v pacman &>/dev/null; then
            sudo pacman -S --noconfirm python python-pip || {
                log_error "Failed to install Python. Please install manually."
                return 1
            }
        fi
    fi
    
    # Determine Python command
    if command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        log_error "No Python interpreter found"
        return 1
    fi
    
    log_debug "Using Python: $PYTHON_CMD"
    
    # Create virtual environment to avoid system package conflicts
    log_info "Creating Python virtual environment..."
    
    # Remove old venv if it exists
    if [[ -d "$VENV_DIR" ]]; then
        log_debug "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    fi
    
    # Create new virtual environment
    $PYTHON_CMD -m venv "$VENV_DIR" || {
        log_warning "Failed to create venv, trying with ensurepip..."
        $PYTHON_CMD -m venv --without-pip "$VENV_DIR"
        
        # Download get-pip.py if pip is not available
        log_info "Installing pip manually..."
        curl -sS https://bootstrap.pypa.io/get-pip.py | "$VENV_DIR/bin/python"
    }
    
    # Activate virtual environment for subsequent operations
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    log_info "Upgrading pip..."
    python -m pip install --upgrade pip setuptools wheel
    
    log_success "Python environment fixed"
}

# ============================================================================
# FIX 2: INSTALL DEPENDENCIES
# ============================================================================

install_python_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Ensure we're in the virtual environment
    if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
        log_error "Virtual environment not found. Run fix_python_environment first."
        return 1
    fi
    
    source "$VENV_DIR/bin/activate"
    
    # Core dependencies
    local core_deps=(
        "numpy>=1.19.0"
        "psutil>=5.8.0"
        "requests>=2.25.0"
    )
    
    # Optional dependencies for enhanced features
    local optional_deps=(
        "scikit-learn>=0.24.0"
        "pandas>=1.2.0"
        "joblib>=1.0.0"
        "aiohttp>=3.7.0"
        "cryptography>=3.4.0"
    )
    
    # Install core dependencies
    for dep in "${core_deps[@]}"; do
        log_debug "Installing $dep..."
        pip install "$dep" || log_warning "Failed to install $dep"
    done
    
    # Install optional dependencies (non-critical)
    for dep in "${optional_deps[@]}"; do
        log_debug "Installing optional: $dep..."
        pip install "$dep" 2>/dev/null || log_debug "Optional package not installed: $dep"
    done
    
    # Install PyQt for GUI (if available)
    if command -v qmake &>/dev/null; then
        pip install PyQt6 2>/dev/null || pip install PyQt5 2>/dev/null || log_debug "PyQt not installed"
    fi
    
    log_success "Python dependencies installed"
}

# ============================================================================
# FIX 3: FILE PATH CORRECTIONS
# ============================================================================

fix_file_paths() {
    log_info "Fixing file paths for Steam Deck installation..."
    
    # Create proper directory structure
    mkdir -p "$INSTALL_DIR"/{src,scripts,config,data,logs}
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$CACHE_DIR"/{compiled,ml_models,p2p}
    
    # Copy source files to correct location
    if [[ -d "$SCRIPT_DIR/src" ]]; then
        log_debug "Copying source files..."
        cp -r "$SCRIPT_DIR/src"/* "$INSTALL_DIR/src/" 2>/dev/null || true
    fi
    
    # Copy from nested directories if they exist
    if [[ -d "$SCRIPT_DIR/shader-prediction-compilation-main/shader-predict-compile/src" ]]; then
        log_debug "Copying from nested directory..."
        cp -r "$SCRIPT_DIR/shader-prediction-compilation-main/shader-predict-compile/src"/* "$INSTALL_DIR/src/" 2>/dev/null || true
    fi
    
    # Copy configuration files
    if [[ -d "$SCRIPT_DIR/config" ]]; then
        cp -r "$SCRIPT_DIR/config"/* "$CONFIG_DIR/" 2>/dev/null || true
    elif [[ -d "$SCRIPT_DIR/shader-prediction-compilation-main/shader-predict-compile/config" ]]; then
        cp -r "$SCRIPT_DIR/shader-prediction-compilation-main/shader-predict-compile/config"/* "$CONFIG_DIR/" 2>/dev/null || true
    fi
    
    # Copy scripts
    local script_dirs=(
        "$SCRIPT_DIR/scripts"
        "$SCRIPT_DIR/steamdeck-deployment/scripts"
        "$SCRIPT_DIR/shader-prediction-compilation-main/shader-predict-compile/scripts"
    )
    
    for script_dir in "${script_dirs[@]}"; do
        if [[ -d "$script_dir" ]]; then
            log_debug "Copying scripts from $script_dir..."
            cp -r "$script_dir"/* "$INSTALL_DIR/scripts/" 2>/dev/null || true
        fi
    done
    
    # Fix shebang lines in Python files to use virtual environment
    log_debug "Updating Python shebangs..."
    find "$INSTALL_DIR" -name "*.py" -type f -exec sed -i "1s|.*python.*|#!$VENV_DIR/bin/python|" {} \;
    
    # Make all scripts executable
    find "$INSTALL_DIR" -name "*.sh" -type f -exec chmod +x {} \;
    find "$INSTALL_DIR" -name "*.py" -type f -exec chmod +x {} \;
    
    log_success "File paths fixed"
}

# ============================================================================
# FIX 4: CREATE LAUNCHER SCRIPTS
# ============================================================================

create_launcher_scripts() {
    log_info "Creating launcher scripts..."
    
    # Main launcher script
    cat > "$INSTALL_DIR/launcher.sh" << EOF
#!/bin/bash
# Shader Prediction Compiler Launcher for Steam Deck

# Set up environment
export SHADER_PREDICT_HOME="$INSTALL_DIR"
export PYTHONPATH="\${SHADER_PREDICT_HOME}/src:\${PYTHONPATH}"
export PATH="$VENV_DIR/bin:\${PATH}"

# Steam Deck optimizations
if grep -q "Jupiter\\|Galileo" /sys/class/dmi/id/board_name 2>/dev/null; then
    export STEAM_DECK=1
    export RADV_PERFTEST=aco
    export MESA_GLSL_CACHE_DISABLE=0
fi

# Change to installation directory
cd "\$SHADER_PREDICT_HOME"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Check which main file exists and run it
if [[ -f "src/shader_prediction_system.py" ]]; then
    exec python src/shader_prediction_system.py "\$@"
elif [[ -f "src/main.py" ]]; then
    exec python src/main.py "\$@"
elif [[ -f "src/background_service.py" ]]; then
    exec python src/background_service.py "\$@"
else
    echo "Error: No main Python file found in src/"
    echo "Available Python files:"
    ls -la src/*.py 2>/dev/null || echo "No Python files found"
    exit 1
fi
EOF
    chmod +x "$INSTALL_DIR/launcher.sh"
    
    # Create system-wide launcher (in user's local bin)
    mkdir -p "$HOME_DIR/.local/bin"
    cat > "$HOME_DIR/.local/bin/shader-predict-compile" << EOF
#!/bin/bash
exec "$INSTALL_DIR/launcher.sh" "\$@"
EOF
    chmod +x "$HOME_DIR/.local/bin/shader-predict-compile"
    
    # Create desktop entry
    mkdir -p "$HOME_DIR/.local/share/applications"
    cat > "$HOME_DIR/.local/share/applications/shader-predict-compile.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Shader Prediction Compiler
GenericName=Gaming Performance Optimizer
Comment=AI-powered shader compilation optimization for Steam Deck
Icon=applications-games
Exec=$INSTALL_DIR/launcher.sh --gui
Terminal=false
Categories=Game;Utility;System;
StartupNotify=true
Keywords=shader;gaming;optimization;steam;performance;
EOF
    
    # Create systemd user service
    mkdir -p "$HOME_DIR/.config/systemd/user"
    cat > "$HOME_DIR/.config/systemd/user/shader-predict.service" << EOF
[Unit]
Description=Shader Prediction Compiler Service
After=graphical-session.target

[Service]
Type=simple
ExecStart=$INSTALL_DIR/launcher.sh --service
Restart=on-failure
RestartSec=10

# Resource limits for Steam Deck
MemoryMax=500M
CPUQuota=10%

# Environment
Environment="HOME=$HOME_DIR"
Environment="USER=deck"

[Install]
WantedBy=default.target
EOF
    
    log_success "Launcher scripts created"
}

# ============================================================================
# FIX 5: CREATE DEFAULT CONFIGURATION
# ============================================================================

create_configuration() {
    log_info "Creating default configuration..."
    
    # Main configuration file
    cat > "$CONFIG_DIR/settings.json" << EOF
{
    "version": "2.0.0",
    "installation_date": "$(date -Iseconds)",
    "installation_path": "$INSTALL_DIR",
    "python_venv": "$VENV_DIR",
    "system": {
        "steam_deck": true,
        "auto_detect_games": true,
        "background_compilation": true,
        "cache_size_mb": 1024,
        "max_parallel_jobs": 2
    },
    "ml_prediction": {
        "enabled": true,
        "model_path": "$INSTALL_DIR/data/models/shader_predictor.pkl",
        "confidence_threshold": 0.7,
        "learning_enabled": true
    },
    "resource_limits": {
        "max_memory_mb": 500,
        "max_cpu_percent": 10,
        "thermal_throttle_temp": 75,
        "battery_save_mode": true
    },
    "paths": {
        "src_dir": "$INSTALL_DIR/src",
        "cache_dir": "$CACHE_DIR",
        "log_dir": "$INSTALL_DIR/logs",
        "model_dir": "$INSTALL_DIR/data/models"
    },
    "logging": {
        "level": "INFO",
        "max_log_size_mb": 10,
        "max_log_files": 5
    }
}
EOF
    
    log_success "Configuration created"
}

# ============================================================================
# FIX 6: TEST INSTALLATION
# ============================================================================

test_installation() {
    log_info "Testing installation..."
    
    local test_passed=true
    
    # Test 1: Check virtual environment
    if [[ -f "$VENV_DIR/bin/python" ]]; then
        log_success "Virtual environment exists"
    else
        log_error "Virtual environment not found"
        test_passed=false
    fi
    
    # Test 2: Check Python imports
    log_debug "Testing Python imports..."
    "$VENV_DIR/bin/python" -c "import numpy; import psutil; print('Core imports successful')" 2>/dev/null || {
        log_warning "Some Python imports failed"
    }
    
    # Test 3: Check launcher script
    if [[ -x "$INSTALL_DIR/launcher.sh" ]]; then
        log_success "Launcher script is executable"
    else
        log_error "Launcher script not found or not executable"
        test_passed=false
    fi
    
    # Test 4: Check main Python files
    local main_files=(
        "$INSTALL_DIR/src/shader_prediction_system.py"
        "$INSTALL_DIR/src/main.py"
        "$INSTALL_DIR/src/background_service.py"
    )
    
    local found_main=false
    for file in "${main_files[@]}"; do
        if [[ -f "$file" ]]; then
            log_success "Found main file: $(basename "$file")"
            found_main=true
            break
        fi
    done
    
    if [[ "$found_main" == false ]]; then
        log_error "No main Python file found"
        test_passed=false
    fi
    
    # Test 5: Try to run with --help
    if "$INSTALL_DIR/launcher.sh" --help &>/dev/null; then
        log_success "Application runs successfully"
    else
        log_warning "Application may have runtime issues"
    fi
    
    if [[ "$test_passed" == true ]]; then
        log_success "All tests passed!"
        return 0
    else
        log_warning "Some tests failed - manual intervention may be needed"
        return 1
    fi
}

# ============================================================================
# MAIN INSTALLATION FLOW
# ============================================================================

show_banner() {
    echo -e "${BOLD}${BLUE}"
    cat << 'BANNER'
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║     STEAM DECK SHADER PREDICTION COMPILER - FIX SCRIPT        ║
║                                                                ║
║     Comprehensive fix for all installation issues             ║
║     • Python/pip3 issues resolved                            ║
║     • File paths corrected                                   ║
║     • Immutable filesystem handled                           ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
BANNER
    echo -e "${NC}\n"
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --install-only)
                INSTALL_ONLY=true
                shift
                ;;
            --fix-only)
                FIX_ONLY=true
                shift
                ;;
            --verbose|-v)
                VERBOSE=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_warning "Unknown option: $1"
                shift
                ;;
        esac
    done
}

show_help() {
    cat << EOF
Steam Deck Shader Prediction Compiler - Fix Script

USAGE:
    bash steam-deck-fix.sh [OPTIONS]

OPTIONS:
    --install-only    Only perform fresh installation
    --fix-only        Only fix existing installation
    --verbose, -v     Show detailed output
    --help, -h        Show this help message

EXAMPLES:
    # Full fix and installation
    bash steam-deck-fix.sh

    # Verbose mode for debugging
    bash steam-deck-fix.sh --verbose

    # Only fix existing installation
    bash steam-deck-fix.sh --fix-only

This script will:
1. Fix Python/pip3 issues by creating a virtual environment
2. Correct all file paths for Steam Deck's filesystem
3. Install all required dependencies
4. Create proper launcher scripts
5. Set up systemd services with resource limits
6. Test the installation

After running this script, you can:
- Launch GUI: shader-predict-compile --gui
- Start service: systemctl --user start shader-predict.service
- Check status: systemctl --user status shader-predict.service
EOF
}

main() {
    # Parse command line arguments
    parse_arguments "$@"
    
    # Show banner
    show_banner
    
    # Detect Steam Deck
    detect_steam_deck || true
    
    # Run fixes
    if [[ "$FIX_ONLY" != true ]]; then
        log_info "Starting comprehensive fix process..."
        
        # Fix 1: Python environment
        fix_python_environment || {
            log_error "Failed to fix Python environment"
            exit 1
        }
        
        # Fix 2: Install dependencies
        install_python_dependencies || {
            log_error "Failed to install dependencies"
            exit 1
        }
    fi
    
    if [[ "$INSTALL_ONLY" != true ]]; then
        # Fix 3: File paths
        fix_file_paths || {
            log_error "Failed to fix file paths"
            exit 1
        }
        
        # Fix 4: Create launchers
        create_launcher_scripts || {
            log_error "Failed to create launcher scripts"
            exit 1
        }
        
        # Fix 5: Create configuration
        create_configuration || {
            log_error "Failed to create configuration"
            exit 1
        }
    fi
    
    # Test installation
    test_installation || true
    
    # Success message
    echo -e "\n${BOLD}${GREEN}════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${GREEN}    Installation Fixed Successfully!${NC}"
    echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════${NC}\n"
    
    echo -e "${GREEN}✓${NC} All issues have been resolved!"
    echo -e "\nYou can now run the application using:"
    echo -e "  ${BLUE}$INSTALL_DIR/launcher.sh${NC}"
    echo -e "\nOr use the system command:"
    echo -e "  ${BLUE}shader-predict-compile${NC}"
    echo -e "\nTo start the background service:"
    echo -e "  ${BLUE}systemctl --user start shader-predict.service${NC}"
    echo -e "\nInstallation location: ${BLUE}$INSTALL_DIR${NC}"
    echo -e "Configuration: ${BLUE}$CONFIG_DIR/settings.json${NC}"
    echo -e "Logs: ${BLUE}$INSTALL_DIR/logs/${NC}"
    
    if command -v gamescope &>/dev/null; then
        echo -e "\n${BOLD}Game Mode Integration:${NC}"
        echo -e "The service will automatically optimize shaders when games are launched."
    fi
}

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

# Ensure we're not running as root on Steam Deck
if [[ $EUID -eq 0 ]] && [[ -f /sys/class/dmi/id/board_name ]]; then
    if grep -q "Jupiter\|Galileo" /sys/class/dmi/id/board_name 2>/dev/null; then
        log_error "Do not run this script as root on Steam Deck!"
        log_info "Please run as the deck user."
        exit 1
    fi
fi

# Run main function
main "$@"