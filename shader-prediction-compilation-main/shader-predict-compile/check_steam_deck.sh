#!/bin/bash

# Steam Deck Compatibility Checker
# Verifies system compatibility and dependencies

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_header() { echo -e "\n${BOLD}${BLUE}$1${NC}"; }

# Banner
echo -e "${BOLD}${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              STEAM DECK COMPATIBILITY CHECK                  ║"
echo "║                                                               ║"
echo "║  Verifying system compatibility for Shader Predictive        ║"
echo "║  Compiler installation                                       ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}\n"

# Track overall status
COMPATIBLE=true
WARNINGS=0

# Check if running on Steam Deck
log_header "System Detection"

if [ -f "/sys/devices/virtual/dmi/id/product_name" ]; then
    PRODUCT=$(cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo "Unknown")
    if [[ "$PRODUCT" == *"Jupiter"* ]]; then
        log_success "Steam Deck detected: $PRODUCT"
        
        # Detect model
        MODEL="Unknown"
        if [ -f "/sys/class/drm/card0/device/apu_model" ]; then
            APU_MODEL=$(cat /sys/class/drm/card0/device/apu_model 2>/dev/null || echo "unknown")
            case $APU_MODEL in
                *"Van Gogh"*) MODEL="LCD" ;;
                *"Phoenix"*) MODEL="OLED" ;;
            esac
        fi
        log_info "Model: Steam Deck $MODEL"
    else
        log_warning "Not a Steam Deck ($PRODUCT) - compatibility may vary"
        ((WARNINGS++))
    fi
else
    log_warning "Unable to detect system type"
    ((WARNINGS++))
fi

# Check SteamOS version
if [ -f "/etc/os-release" ]; then
    . /etc/os-release
    log_info "OS: $NAME $VERSION_ID"
    if [[ "$ID" == "steamos" ]]; then
        if [[ "$VERSION_ID" < "3.7" ]]; then
            log_warning "SteamOS $VERSION_ID detected - version 3.7+ recommended"
            ((WARNINGS++))
        else
            log_success "SteamOS version compatible"
        fi
    fi
fi

# Check architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "x86_64" ]]; then
    log_success "Architecture: $ARCH (compatible)"
else
    log_error "Architecture: $ARCH (incompatible - x86_64 required)"
    COMPATIBLE=false
fi

# Check available space
log_header "Storage Check"

INSTALL_DIR="/opt/shader-predict-compile"
REQUIRED_SPACE_MB=500  # 500MB required

if [ -d "/opt" ]; then
    AVAILABLE_SPACE=$(df -BM /opt | tail -1 | awk '{print $4}' | sed 's/M//')
    if [ "$AVAILABLE_SPACE" -ge "$REQUIRED_SPACE_MB" ]; then
        log_success "Disk space: ${AVAILABLE_SPACE}MB available (${REQUIRED_SPACE_MB}MB required)"
    else
        log_error "Insufficient disk space: ${AVAILABLE_SPACE}MB available (${REQUIRED_SPACE_MB}MB required)"
        COMPATIBLE=false
    fi
else
    log_warning "Cannot check disk space - /opt not found"
    ((WARNINGS++))
fi

# Check dependencies
log_header "Dependency Check"

# Python 3
if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
    log_success "Python: $PYTHON_VERSION"
    
    # Check Python version
    PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
    PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 7 ]; then
        log_success "Python version compatible (3.7+ required)"
    else
        log_warning "Python $PYTHON_VERSION detected - 3.7+ recommended"
        ((WARNINGS++))
    fi
else
    log_error "Python 3 not found"
    COMPATIBLE=false
fi

# GTK bindings
if python3 -c "import gi; gi.require_version('Gtk', '3.0')" &>/dev/null 2>&1; then
    log_success "GTK bindings: Available"
else
    log_error "GTK bindings: Not available (python3-gi required)"
    log_info "Install with: sudo pacman -S python-gobject"
    COMPATIBLE=false
fi

# Optional dependencies
log_header "Optional Dependencies"

# psutil
if python3 -c "import psutil" &>/dev/null 2>&1; then
    PSUTIL_VERSION=$(python3 -c "import psutil; print(psutil.__version__)")
    log_success "psutil: $PSUTIL_VERSION"
else
    log_warning "psutil: Not installed (system monitoring will be limited)"
    ((WARNINGS++))
fi

# numpy
if python3 -c "import numpy" &>/dev/null 2>&1; then
    NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)")
    log_success "numpy: $NUMPY_VERSION"
else
    log_warning "numpy: Not installed (advanced analysis features will be limited)"
    ((WARNINGS++))
fi

# System tools
log_header "System Tools"

# systemctl (for service management)
if command -v systemctl &>/dev/null; then
    log_success "systemd: Available"
else
    log_warning "systemd: Not available (background service won't work)"
    ((WARNINGS++))
fi

# pacman (for package management)
if command -v pacman &>/dev/null; then
    log_success "pacman: Available"
else
    log_warning "pacman: Not available (manual dependency installation required)"
    ((WARNINGS++))
fi

# Steam-specific checks
log_header "Steam Integration"

# Check for Steam installation
if [ -d "$HOME/.steam" ] || [ -d "$HOME/.local/share/Steam" ]; then
    log_success "Steam: Installed"
    
    # Check for fossilize
    FOSSILIZE_FOUND=false
    for steam_dir in "$HOME/.steam" "$HOME/.local/share/Steam"; do
        if [ -d "$steam_dir/ubuntu12_32/fossilize" ] || [ -d "$steam_dir/ubuntu12_64/fossilize" ]; then
            FOSSILIZE_FOUND=true
            break
        fi
    done
    
    if [ "$FOSSILIZE_FOUND" = true ]; then
        log_success "Fossilize: Found"
    else
        log_warning "Fossilize: Not found (shader compilation features limited)"
        ((WARNINGS++))
    fi
else
    log_warning "Steam: Not found"
    ((WARNINGS++))
fi

# Gaming Mode check
if pgrep -x gamescope &>/dev/null; then
    log_info "Gaming Mode: Currently active"
else
    log_info "Gaming Mode: Not active (Desktop Mode)"
fi

# GPU check
log_header "GPU Detection"

if [ -f "/sys/class/drm/card0/device/vendor" ]; then
    VENDOR_ID=$(cat /sys/class/drm/card0/device/vendor)
    DEVICE_ID=$(cat /sys/class/drm/card0/device/device 2>/dev/null || echo "unknown")
    
    if [[ "$VENDOR_ID" == "0x1002" ]]; then
        log_success "GPU: AMD (compatible)"
        log_info "Device ID: $DEVICE_ID"
    else
        log_warning "GPU: Non-AMD GPU detected (may have limited compatibility)"
        ((WARNINGS++))
    fi
else
    log_warning "Unable to detect GPU"
    ((WARNINGS++))
fi

# Summary
echo
log_header "Compatibility Summary"
echo

if [ "$COMPATIBLE" = true ]; then
    if [ "$WARNINGS" -eq 0 ]; then
        echo -e "${GREEN}${BOLD}✓ FULLY COMPATIBLE${NC}"
        echo "Your system meets all requirements for Shader Predictive Compiler!"
    else
        echo -e "${YELLOW}${BOLD}⚠ COMPATIBLE WITH WARNINGS${NC}"
        echo "Your system can run Shader Predictive Compiler with some limitations."
        echo "Warnings: $WARNINGS"
    fi
    echo
    echo "Ready to install! Run: ./install"
else
    echo -e "${RED}${BOLD}✗ NOT COMPATIBLE${NC}"
    echo "Your system does not meet the minimum requirements."
    echo "Please address the errors above before installing."
fi

echo
echo "For more information, see README.md"

# Exit with appropriate code
if [ "$COMPATIBLE" = true ]; then
    exit 0
else
    exit 1
fi