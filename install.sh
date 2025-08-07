#!/bin/bash

# Shader Predictive Compiler - GitHub One-Line Installer
# Secure, cross-platform installation with comprehensive error handling
# 
# Installation Command:
#   curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh | bash
#
# Advanced Installation:
#   curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh | bash -s -- --dev --no-autostart
#
# For security-conscious users, download and inspect first:
#   curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh -o install.sh
#   less install.sh  # Inspect the script
#   chmod +x install.sh && ./install.sh

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# ============================================================================
# SECURITY AND VERIFICATION FUNCTIONS
# ============================================================================

# Wrap entire script in a function to prevent partial execution from network interruptions
main() {

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

readonly SCRIPT_VERSION="1.2.0"
readonly REPO_OWNER="YourUsername"
readonly REPO_NAME="shader-prediction-compilation"
readonly REPO_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}"
readonly RAW_URL="https://raw.githubusercontent.com/${REPO_OWNER}/${REPO_NAME}/main"
readonly API_URL="https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}"

readonly INSTALL_DIR="/opt/shader-predict-compile"
readonly CONFIG_DIR="${HOME}/.config/shader-predict-compile"
readonly CACHE_DIR="${HOME}/.cache/shader-predict-compile"
readonly TEMP_DIR="/tmp/shader-install-$$"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Installation options
INSTALL_DEV=false
ENABLE_AUTOSTART=true
ENABLE_P2P=true
ENABLE_ML=true
FORCE_REINSTALL=false
SKIP_DEPS=false

# ============================================================================
# LOGGING AND OUTPUT FUNCTIONS
# ============================================================================

log_header() {
    echo -e "\n${BOLD}${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════════${NC}\n"
}

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
    if [[ "${DEBUG:-}" == "1" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1" >&2
    fi
}

# ============================================================================
# ERROR HANDLING AND CLEANUP
# ============================================================================

cleanup() {
    local exit_code=$?
    if [[ -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR"
        log_debug "Cleaned up temporary directory: $TEMP_DIR"
    fi
    
    if [[ $exit_code -ne 0 ]]; then
        log_error "Installation failed with exit code $exit_code"
        log_info "For help, visit: ${REPO_URL}/issues"
    fi
}

trap cleanup EXIT

error_handler() {
    local line_no=$1
    local error_code=$2
    log_error "Error occurred at line $line_no: exit code $error_code"
    log_info "You can try running with DEBUG=1 for more information"
    exit "$error_code"
}

trap 'error_handler ${LINENO} $?' ERR

# ============================================================================
# SYSTEM DETECTION AND VALIDATION
# ============================================================================

detect_system() {
    log_info "Detecting system configuration..."
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
    else
        log_warning "Unsupported OS: $OSTYPE"
        OS="unknown"
    fi
    
    # Detect architecture
    ARCH=$(uname -m)
    case $ARCH in
        x86_64) ARCH="amd64" ;;
        aarch64|arm64) ARCH="arm64" ;;
        armv7l) ARCH="armv7" ;;
        *) log_warning "Unsupported architecture: $ARCH" ;;
    esac
    
    # Detect Steam Deck
    STEAM_DECK=false
    if [[ -f "/sys/devices/virtual/dmi/id/product_name" ]] || [[ -f "/sys/class/dmi/id/board_name" ]]; then
        if grep -q "Jupiter\|Galileo" /sys/class/dmi/id/board_name 2>/dev/null || \
           grep -q "Jupiter\|Galileo" /sys/devices/virtual/dmi/id/product_name 2>/dev/null; then
            STEAM_DECK=true
            # Detect LCD vs OLED model
            if grep -q "Galileo" /sys/class/dmi/id/board_name 2>/dev/null; then
                STEAM_DECK_MODEL="OLED"
            else
                STEAM_DECK_MODEL="LCD"
            fi
            log_success "Steam Deck $STEAM_DECK_MODEL model detected!"
        fi
    fi
    
    # Detect package manager
    if command -v pacman >/dev/null 2>&1; then
        PKG_MANAGER="pacman"
    elif command -v apt >/dev/null 2>&1; then
        PKG_MANAGER="apt"
    elif command -v dnf >/dev/null 2>&1; then
        PKG_MANAGER="dnf"
    elif command -v zypper >/dev/null 2>&1; then
        PKG_MANAGER="zypper"
    else
        PKG_MANAGER="unknown"
        log_warning "No known package manager found"
    fi
    
    log_info "System: $OS ($ARCH), Package Manager: $PKG_MANAGER"
    if [[ "$STEAM_DECK" == "true" ]]; then
        log_info "Steam Deck Model: $STEAM_DECK_MODEL"
    fi
}

check_requirements() {
    log_info "Checking system requirements..."
    
    local missing_tools=()
    
    # Check for essential tools
    for tool in curl wget unzip python3; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -ne 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Install with your package manager:"
        case $PKG_MANAGER in
            pacman) log_info "  sudo pacman -S ${missing_tools[*]}" ;;
            apt) log_info "  sudo apt update && sudo apt install ${missing_tools[*]}" ;;
            dnf) log_info "  sudo dnf install ${missing_tools[*]}" ;;
            *) log_info "  Please install these tools using your system's package manager" ;;
        esac
        return 1
    fi
    
    # Check Python version
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 7) else 1)" 2>/dev/null; then
        log_error "Python 3.7+ is required"
        return 1
    fi
    
    # Check available space
    local available_space
    available_space=$(df /opt 2>/dev/null | awk 'NR==2 {print $4}' || echo "1000000")
    if [[ $available_space -lt 500000 ]]; then  # 500MB
        log_warning "Low disk space detected. Need at least 500MB free."
    fi
    
    # Check network connectivity
    if ! curl -fsSL --connect-timeout 5 "https://api.github.com" >/dev/null 2>&1; then
        log_error "Cannot connect to GitHub. Please check your internet connection."
        return 1
    fi
    
    log_success "All requirements satisfied"
    return 0
}

check_existing_installation() {
    log_info "Checking for existing installation..."
    
    if [[ -d "$INSTALL_DIR" ]]; then
        if [[ "$FORCE_REINSTALL" == "true" ]]; then
            log_warning "Existing installation found, will reinstall"
            return 0
        else
            log_info "Existing installation detected at $INSTALL_DIR"
            echo -n "Do you want to reinstall? [y/N]: "
            read -r response
            case "$response" in
                [yY][eE][sS]|[yY])
                    log_info "Proceeding with reinstallation"
                    ;;
                *)
                    log_info "Installation cancelled by user"
                    exit 0
                    ;;
            esac
        fi
    fi
}

# ============================================================================
# DOWNLOAD AND VERIFICATION FUNCTIONS
# ============================================================================

verify_checksum() {
    local file="$1"
    local expected_checksum="$2"
    
    if command -v sha256sum >/dev/null 2>&1; then
        local actual_checksum
        actual_checksum=$(sha256sum "$file" | cut -d' ' -f1)
    elif command -v shasum >/dev/null 2>&1; then
        local actual_checksum
        actual_checksum=$(shasum -a 256 "$file" | cut -d' ' -f1)
    else
        log_warning "No checksum utility found, skipping verification"
        return 0
    fi
    
    if [[ "$actual_checksum" == "$expected_checksum" ]]; then
        log_success "Checksum verification passed"
        return 0
    else
        log_error "Checksum verification failed!"
        log_error "Expected: $expected_checksum"
        log_error "Actual:   $actual_checksum"
        return 1
    fi
}

download_latest_release() {
    log_info "Fetching latest release information..."
    
    mkdir -p "$TEMP_DIR"
    cd "$TEMP_DIR"
    
    # Get latest release info from GitHub API
    local release_info
    release_info=$(curl -fsSL "$API_URL/releases/latest" 2>/dev/null) || {
        log_warning "Could not fetch release info from API, using direct download"
        return 1
    }
    
    # Parse release information
    local tag_name
    tag_name=$(echo "$release_info" | grep '"tag_name"' | cut -d'"' -f4)
    local download_url="$REPO_URL/archive/refs/tags/$tag_name.tar.gz"
    
    log_info "Latest release: $tag_name"
    log_info "Downloading from: $download_url"
    
    # Download with progress bar if possible
    if curl --help 2>&1 | grep -q '\--progress-bar'; then
        curl -fsSL --progress-bar "$download_url" -o "release.tar.gz"
    else
        curl -fsSL "$download_url" -o "release.tar.gz"
    fi
    
    log_success "Downloaded release archive"
    
    # Extract
    tar -xzf "release.tar.gz" --strip-components=1
    rm "release.tar.gz"
    
    log_success "Extracted release files"
}

download_main_branch() {
    log_info "Downloading main branch..."
    
    mkdir -p "$TEMP_DIR"
    cd "$TEMP_DIR"
    
    local download_url="$REPO_URL/archive/refs/heads/main.tar.gz"
    
    # Try curl first, then wget
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$download_url" -o "main.tar.gz"
    elif command -v wget >/dev/null 2>&1; then
        wget -q "$download_url" -O "main.tar.gz"
    else
        log_error "Neither curl nor wget available"
        return 1
    fi
    
    tar -xzf "main.tar.gz" --strip-components=1
    rm "main.tar.gz"
    
    log_success "Downloaded main branch"
}

# ============================================================================
# DEPENDENCY INSTALLATION
# ============================================================================

install_system_dependencies() {
    if [[ "$SKIP_DEPS" == "true" ]]; then
        log_info "Skipping system dependency installation"
        return 0
    fi
    
    log_info "Installing system dependencies..."
    
    local deps=()
    if [[ "$STEAM_DECK" == "true" ]]; then
        deps=(python python-pip python-gobject gtk3 mesa-utils vulkan-tools)
    else
        case $PKG_MANAGER in
            pacman)
                deps=(python python-pip python-gobject gtk3 mesa-utils vulkan-tools python-requests python-psutil)
                ;;
            apt)
                deps=(python3 python3-pip python3-gi gir1.2-gtk-3.0 mesa-utils vulkan-tools python3-requests python3-psutil)
                ;;
            dnf)
                deps=(python3 python3-pip python3-gobject gtk3-devel mesa-utils vulkan-tools python3-requests python3-psutil)
                ;;
            *)
                log_warning "Unknown package manager, skipping system dependencies"
                return 0
                ;;
        esac
    fi
    
    # Install based on package manager
    case $PKG_MANAGER in
        pacman)
            if ! sudo pacman -Sy --noconfirm "${deps[@]}" 2>/dev/null; then
                log_warning "Some packages failed to install with pacman"
            fi
            ;;
        apt)
            sudo apt update >/dev/null 2>&1 || true
            if ! sudo apt install -y "${deps[@]}" 2>/dev/null; then
                log_warning "Some packages failed to install with apt"
            fi
            ;;
        dnf)
            if ! sudo dnf install -y "${deps[@]}" 2>/dev/null; then
                log_warning "Some packages failed to install with dnf"
            fi
            ;;
        *)
            log_warning "Cannot install system dependencies automatically"
            ;;
    esac
    
    log_success "System dependencies installed"
}

install_python_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Ensure pip is up to date
    python3 -m pip install --user --upgrade pip setuptools wheel >/dev/null 2>&1 || true
    
    # Core dependencies
    local python_deps=(
        "numpy>=1.19.0"
        "psutil>=5.7.0"
        "requests>=2.25.0"
        "pathlib2;python_version<'3.4'"
    )
    
    # Optional dependencies for enhanced features
    local optional_deps=(
        "PyGObject>=3.36.0"  # For GUI support
        "cryptography>=3.4.0"  # For P2P security
        "aiohttp>=3.7.0"  # For async networking
    )
    
    # Install core dependencies
    for dep in "${python_deps[@]}"; do
        if ! python3 -m pip install --user "$dep" >/dev/null 2>&1; then
            log_warning "Failed to install Python package: $dep"
        fi
    done
    
    # Install optional dependencies (non-critical)
    for dep in "${optional_deps[@]}"; do
        if ! python3 -m pip install --user "$dep" >/dev/null 2>&1; then
            log_debug "Optional package not installed: $dep"
        fi
    done
    
    log_success "Python dependencies installed"
}

# ============================================================================
# INSTALLATION FUNCTIONS
# ============================================================================

create_directories() {
    log_info "Creating installation directories..."
    
    # System directory (requires sudo)
    sudo mkdir -p "$INSTALL_DIR"
    sudo chown "$(whoami):$(whoami)" "$INSTALL_DIR" 2>/dev/null || true
    
    # User directories
    mkdir -p "$CONFIG_DIR" "$CACHE_DIR"
    mkdir -p "${CONFIG_DIR}/games" "${CACHE_DIR}/compiled" "${CACHE_DIR}/p2p"
    
    log_success "Directories created"
}

install_core_files() {
    log_info "Installing core application files..."
    
    cd "$TEMP_DIR"
    
    # Copy main application files
    if [[ -d "src" ]]; then
        cp -r src/* "$INSTALL_DIR/"
    elif [[ -f "shader_prediction_system.py" ]]; then
        cp *.py "$INSTALL_DIR/" 2>/dev/null || true
    else
        log_error "Could not find source files in downloaded archive"
        return 1
    fi
    
    # Copy configuration files
    if [[ -d "config" ]]; then
        cp -r config/* "$CONFIG_DIR/" 2>/dev/null || true
    fi
    
    # Copy security framework
    if [[ -d "security" ]]; then
        cp -r security "$INSTALL_DIR/"
    fi
    
    # Make scripts executable
    find "$INSTALL_DIR" -name "*.py" -exec chmod +x {} \; 2>/dev/null || true
    find "$INSTALL_DIR" -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
    
    log_success "Core files installed"
}

create_launcher_scripts() {
    log_info "Creating launcher scripts..."
    
    # Main launcher
    cat > "$INSTALL_DIR/launcher.sh" << 'LAUNCHER_EOF'
#!/bin/bash
# Shader Prediction Compiler Launcher
set -e

INSTALL_DIR="/opt/shader-predict-compile"
cd "$INSTALL_DIR"

# Set up environment
export PYTHONPATH="$INSTALL_DIR:$PYTHONPATH"
export PATH="$HOME/.local/bin:$PATH"

# Steam Deck optimizations
if grep -q "Jupiter\|Galileo" /sys/class/dmi/id/board_name 2>/dev/null; then
    export STEAM_DECK=1
    export RADV_PERFTEST=aco
    export MESA_GLSL_CACHE_DISABLE=0
fi

# Launch application
if [[ "$1" == "--gui" ]]; then
    python3 shader_prediction_system.py --gui
elif [[ "$1" == "--service" ]]; then
    python3 shader_prediction_system.py --service
else
    python3 shader_prediction_system.py "$@"
fi
LAUNCHER_EOF
    
    chmod +x "$INSTALL_DIR/launcher.sh"
    
    # Create system-wide launcher
    sudo tee /usr/local/bin/shader-predict-compile >/dev/null << 'SYSTEM_LAUNCHER_EOF'
#!/bin/bash
exec /opt/shader-predict-compile/launcher.sh "$@"
SYSTEM_LAUNCHER_EOF
    
    sudo chmod +x /usr/local/bin/shader-predict-compile
    
    log_success "Launcher scripts created"
}

create_desktop_integration() {
    log_info "Creating desktop integration..."
    
    # Create desktop entry
    mkdir -p "$HOME/.local/share/applications"
    cat > "$HOME/.local/share/applications/shader-predict-compile.desktop" << 'DESKTOP_EOF'
[Desktop Entry]
Version=1.0
Type=Application
Name=Shader Prediction Compiler
GenericName=Gaming Performance Optimizer
Comment=AI-powered shader compilation optimization for Steam Deck
Icon=applications-games
Exec=/opt/shader-predict-compile/launcher.sh --gui
Terminal=false
Categories=Game;Utility;System;
StartupNotify=true
Keywords=shader;gaming;optimization;steam;performance;
DESKTOP_EOF
    
    # Steam Deck specific integration
    if [[ "$STEAM_DECK" == "true" ]]; then
        # Create Steam shortcut
        mkdir -p "$HOME/.steam/steam/config"
        # Note: Steam shortcut creation would require more complex Steam library manipulation
        log_info "Steam Deck detected - desktop shortcut created"
    fi
    
    # Create autostart entry if enabled
    if [[ "$ENABLE_AUTOSTART" == "true" ]]; then
        mkdir -p "$HOME/.config/autostart"
        cat > "$HOME/.config/autostart/shader-predict-compile.desktop" << 'AUTOSTART_EOF'
[Desktop Entry]
Version=1.0
Type=Application
Name=Shader Prediction Compiler (Service)
Comment=Background shader optimization service
Icon=applications-games
Exec=/opt/shader-predict-compile/launcher.sh --service
Terminal=false
Hidden=false
NoDisplay=true
X-GNOME-Autostart-enabled=true
AUTOSTART_EOF
        log_info "Autostart enabled"
    fi
    
    log_success "Desktop integration created"
}

create_configuration() {
    log_info "Creating default configuration..."
    
    # Main configuration
    cat > "$CONFIG_DIR/config.json" << 'CONFIG_EOF'
{
    "version": "1.0.0",
    "installation_date": "",
    "system": {
        "steam_deck": false,
        "steam_deck_model": "unknown",
        "auto_detect_games": true,
        "background_compilation": true,
        "cache_size_mb": 2048,
        "max_parallel_jobs": 4
    },
    "ml_prediction": {
        "enabled": true,
        "model_path": "models/shader_predictor.pkl",
        "confidence_threshold": 0.7,
        "learning_enabled": true
    },
    "p2p_network": {
        "enabled": true,
        "port": 0,
        "max_connections": 50,
        "bandwidth_limit_kbps": 2048,
        "community_sharing": true,
        "reputation_threshold": 0.3
    },
    "security": {
        "verify_checksums": true,
        "signature_verification": true,
        "sandbox_compilation": true,
        "privacy_protection": true
    },
    "logging": {
        "level": "INFO",
        "max_log_size_mb": 10,
        "max_log_files": 5
    }
}
CONFIG_EOF
    
    # Update system-specific settings
    python3 -c "
import json
import sys
from pathlib import Path

config_file = Path('$CONFIG_DIR/config.json')
config = json.loads(config_file.read_text())

# Update system info
config['installation_date'] = '$(date -Iseconds)'
config['system']['steam_deck'] = $STEAM_DECK
config['system']['steam_deck_model'] = '${STEAM_DECK_MODEL:-unknown}'

# Adjust settings based on system
if $STEAM_DECK:
    config['system']['cache_size_mb'] = 1024  # Smaller for Steam Deck
    config['p2p_network']['bandwidth_limit_kbps'] = 1024  # Conservative WiFi

config_file.write_text(json.dumps(config, indent=2))
"
    
    log_success "Configuration created"
}

create_uninstaller() {
    log_info "Creating uninstaller..."
    
    cat > "$INSTALL_DIR/uninstall.sh" << 'UNINSTALL_EOF'
#!/bin/bash
# Shader Prediction Compiler Uninstaller

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Shader Prediction Compiler Uninstaller${NC}"
echo "======================================="
echo

echo -n "Are you sure you want to uninstall? [y/N]: "
read -r response
case "$response" in
    [yY][eE][sS]|[yY])
        ;;
    *)
        echo "Uninstallation cancelled"
        exit 0
        ;;
esac

echo -e "\n${YELLOW}Uninstalling Shader Prediction Compiler...${NC}"

# Stop any running services
pkill -f "shader_prediction_system.py" 2>/dev/null || true

# Remove files
echo "Removing installation directory..."
sudo rm -rf /opt/shader-predict-compile

echo "Removing configuration..."
rm -rf ~/.config/shader-predict-compile

echo "Removing cache..."
rm -rf ~/.cache/shader-predict-compile

echo "Removing desktop integration..."
rm -f ~/.local/share/applications/shader-predict-compile.desktop
rm -f ~/.config/autostart/shader-predict-compile.desktop

echo "Removing system launcher..."
sudo rm -f /usr/local/bin/shader-predict-compile

echo -e "\n${GREEN}✓${NC} Uninstallation complete!"
echo
echo "Your game shader caches and Steam integration remain untouched."
echo "Thank you for using Shader Prediction Compiler!"
UNINSTALL_EOF
    
    chmod +x "$INSTALL_DIR/uninstall.sh"
    
    # Create global uninstaller command
    sudo tee /usr/local/bin/uninstall-shader-predict-compile >/dev/null << 'GLOBAL_UNINSTALL_EOF'
#!/bin/bash
exec /opt/shader-predict-compile/uninstall.sh "$@"
GLOBAL_UNINSTALL_EOF
    
    sudo chmod +x /usr/local/bin/uninstall-shader-predict-compile
    
    log_success "Uninstaller created"
}

# ============================================================================
# POST-INSTALLATION FUNCTIONS
# ============================================================================

run_post_install_setup() {
    log_info "Running post-installation setup..."
    
    # Create initial shader cache directories
    mkdir -p "$CACHE_DIR"/{compiled,p2p,ml_models,temporary}
    
    # Initialize ML models if available
    if [[ -f "$INSTALL_DIR/models/download_models.py" ]]; then
        log_info "Downloading ML models..."
        cd "$INSTALL_DIR/models" && python3 download_models.py >/dev/null 2>&1 || log_warning "Model download failed"
    fi
    
    # Test installation
    log_info "Testing installation..."
    if ! "$INSTALL_DIR/launcher.sh" --test >/dev/null 2>&1; then
        log_warning "Installation test failed - some features may not work"
    else
        log_success "Installation test passed"
    fi
    
    # Start service if on Steam Deck and autostart enabled
    if [[ "$STEAM_DECK" == "true" ]] && [[ "$ENABLE_AUTOSTART" == "true" ]]; then
        log_info "Starting background service..."
        nohup "$INSTALL_DIR/launcher.sh" --service >/dev/null 2>&1 &
    fi
    
    log_success "Post-installation setup complete"
}

# ============================================================================
# COMMAND LINE ARGUMENT PARSING
# ============================================================================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dev|--development)
                INSTALL_DEV=true
                log_info "Development installation mode enabled"
                shift
                ;;
            --no-autostart)
                ENABLE_AUTOSTART=false
                log_info "Autostart disabled"
                shift
                ;;
            --no-p2p)
                ENABLE_P2P=false
                log_info "P2P features disabled"
                shift
                ;;
            --no-ml)
                ENABLE_ML=false
                log_info "ML features disabled"
                shift
                ;;
            --force)
                FORCE_REINSTALL=true
                log_info "Force reinstall enabled"
                shift
                ;;
            --skip-deps)
                SKIP_DEPS=true
                log_info "Skipping dependency installation"
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            --version|-v)
                echo "Shader Prediction Compiler Installer v$SCRIPT_VERSION"
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
    cat << 'HELP_EOF'
Shader Prediction Compiler - GitHub Installer

USAGE:
    curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh | bash
    
    # Or with options:
    curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh | bash -s -- [OPTIONS]

OPTIONS:
    --dev, --development    Install development version from main branch
    --no-autostart         Don't enable automatic startup
    --no-p2p              Disable P2P shader sharing features  
    --no-ml               Disable ML prediction features
    --force               Force reinstallation over existing install
    --skip-deps           Skip system dependency installation
    --help, -h            Show this help message
    --version, -v         Show installer version

EXAMPLES:
    # Standard installation
    curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh | bash
    
    # Development install without autostart
    curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh | bash -s -- --dev --no-autostart
    
    # Minimal install (no P2P, no ML)
    curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh | bash -s -- --no-p2p --no-ml

SECURITY:
    For security-conscious users, you can download and inspect the script first:
    
    curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh -o install.sh
    less install.sh  # Inspect the script
    chmod +x install.sh && ./install.sh

UNINSTALLATION:
    uninstall-shader-predict-compile
    
    # Or directly:
    /opt/shader-predict-compile/uninstall.sh

MORE INFO:
    Repository: https://github.com/YourUsername/shader-prediction-compilation
    Issues: https://github.com/YourUsername/shader-prediction-compilation/issues
    Documentation: https://github.com/YourUsername/shader-prediction-compilation/wiki
HELP_EOF
}

# ============================================================================
# MAIN INSTALLATION FLOW
# ============================================================================

show_banner() {
    echo -e "${BOLD}${BLUE}"
    cat << 'BANNER_EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        🎮 SHADER PREDICTION COMPILER - GITHUB INSTALLER                      ║
║                                                                              ║
║        AI-Powered Shader Optimization for Steam Deck                        ║
║        Secure • Cross-Platform • Community-Driven                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
BANNER_EOF
    echo -e "${NC}"
    
    log_info "Installer Version: $SCRIPT_VERSION"
    log_info "Repository: $REPO_URL"
    echo
}

run_installation() {
    log_header "SYSTEM ANALYSIS"
    detect_system
    check_requirements || {
        log_error "System requirements not met"
        exit 1
    }
    check_existing_installation
    
    log_header "DOWNLOADING APPLICATION"
    if [[ "$INSTALL_DEV" == "true" ]]; then
        download_main_branch
    else
        download_latest_release || download_main_branch
    fi
    
    log_header "INSTALLING DEPENDENCIES"
    install_system_dependencies
    install_python_dependencies
    
    log_header "INSTALLING APPLICATION"
    create_directories
    install_core_files
    create_launcher_scripts
    create_desktop_integration
    create_configuration
    create_uninstaller
    
    log_header "FINAL SETUP"
    run_post_install_setup
    
    log_header "INSTALLATION COMPLETE"
    show_success_message
}

show_success_message() {
    echo -e "${BOLD}${GREEN}"
    cat << 'SUCCESS_EOF'
🎉 INSTALLATION SUCCESSFUL! 🎉

Your Shader Prediction Compiler is now installed and ready to optimize your gaming experience!
SUCCESS_EOF
    echo -e "${NC}"
    
    echo
    log_success "Installation completed successfully!"
    echo
    
    echo -e "${BOLD}Quick Start:${NC}"
    echo "  🖥️  GUI Mode:      shader-predict-compile --gui"
    echo "  🔧 Service Mode:   shader-predict-compile --service"
    echo "  📊 Status:        shader-predict-compile --status"
    echo "  ❌ Uninstall:     uninstall-shader-predict-compile"
    echo
    
    if [[ "$STEAM_DECK" == "true" ]]; then
        echo -e "${BOLD}Steam Deck Users:${NC}"
        echo "  🎮 Gaming Mode: Look for 'Shader Prediction Compiler' in your Library"
        echo "  🔄 Auto-Start: Background optimization starts automatically"
        echo "  ⚡ Performance: Optimized for $STEAM_DECK_MODEL model"
        echo
    fi
    
    echo -e "${BOLD}Features Enabled:${NC}"
    echo "  🤖 ML Prediction:    $(if [[ "$ENABLE_ML" == "true" ]]; then echo "✓ Enabled"; else echo "✗ Disabled"; fi)"
    echo "  🌐 P2P Sharing:      $(if [[ "$ENABLE_P2P" == "true" ]]; then echo "✓ Enabled"; else echo "✗ Disabled"; fi)"
    echo "  🚀 Auto-Start:       $(if [[ "$ENABLE_AUTOSTART" == "true" ]]; then echo "✓ Enabled"; else echo "✗ Disabled"; fi)"
    echo
    
    echo -e "${BOLD}Support & Documentation:${NC}"
    echo "  📖 Documentation: ${REPO_URL}/wiki"
    echo "  🐛 Report Issues: ${REPO_URL}/issues"
    echo "  💬 Discussions:   ${REPO_URL}/discussions"
    echo
    
    log_info "Happy gaming! Your shaders will compile faster and games will run smoother."
}

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    log_error "Do not run this installer as root!"
    log_info "The installer will ask for sudo access when needed."
    exit 1
fi

# Parse command line arguments
parse_arguments "$@"

# Show banner
show_banner

# Run the installation
run_installation

# End of main function
}

# Call main function with all arguments to prevent partial execution
main "$@"